# SPDX-FileCopyrightText: © 2023 abdelrahmanjamal5565@gmail.com
# SPDX-FileContributor: Kaushlendra Pratap Singh <kaushlendra-pratap.singh@siemens.com>
# SPDX-License-Identifier: LGPL-2.1-only

"""
SafaaAgent: A module for handling false positive detection in copyright notices.
"""

import hashlib
import html
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import spacy
from joblib import load, dump
from importlib.resources import files, as_file
import shutil

# Constants
PACKAGE_NAME = "safaa"
MODELS_REF = files(PACKAGE_NAME).joinpath("models")
CONFIGS_REF = files(PACKAGE_NAME).joinpath("configs")
LOCAL_MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")

# Keyed by content digest: pip --target can leave stale version metadata.
BASE_CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "safaa")

# A detector is valid only with the preprocessing version in its METADATA_FILE (none: v1).
PREPROCESSING_VERSION = 2
METADATA_FILE = "false_positive_detection_model.json"


def tree_digest(path):
    """SHA-256 over every file under path: names and contents, in a fixed order."""
    h = hashlib.sha256()
    for root, dirs, names in os.walk(path):
        dirs.sort()
        for name in sorted(names):
            full = os.path.join(root, name)
            h.update(os.path.relpath(full, path).encode("utf-8") + b"\0")
            with open(full, "rb") as fh:
                for block in iter(lambda: fh.read(1 << 20), b""):
                    h.update(block)
    return h.hexdigest()


def preprocessing_fingerprint(entity_recognizer_path, version=PREPROCESSING_VERSION):
    """Preprocessing version plus entity recognizer digest."""
    return "v%d:%s" % (version, tree_digest(entity_recognizer_path)[:16])


# Preprocessing v2 vocabulary
YEAR_RE = re.compile(r"(?<!\d)(?:19|20)\d{2}(?!\d)")
EMAIL_RE = re.compile(r"[\w.+%-]+@[\w-]+(?:\.[\w-]+)+")
SPDX_TAG_RE = re.compile(r"SPDX-FileCopyrightText\s*:?", re.IGNORECASE)
HTML_COPYRIGHT_RE = re.compile(r"&copy;|&#169;|&#xa9;", re.IGNORECASE)
COPYRIGHT_MARKER_RE = re.compile(r"\bcopyright(?:ed|s)?\b|©|\([cC]\)", re.IGNORECASE)
# Bare YEAR/XXXX only in capitals: prose "year" is no placeholder.
PLACEHOLDER_YEAR_RE = re.compile(
    r"(?i:<\s*year\s*>|\[\s*year\s*\]|\{\{?\s*year\s*\}?\}|\byyyy\b|\$\{?year\b|@var\{year\}|&lt;\s*year\s*&gt;)"
    r"|\bYEAR\b|\bXXXX\b")
PLACEHOLDER_HOLDER_RE = re.compile(
    r"(?i)<\s*(?:name|owner|author|copyright holder|organi[sz]ation)[^>]*>|\[\s*(?:name|owner|fullname|copyright"
    r" holder|yourname)[^\]]*\]|@var\{your name\}|THE PACKAGE'S COPYRIGHT HOLDER|M\. Y\. Name"
    r"|\{\{\s*(?:author|owner|holder|name)[^}]*\}\}|&lt;\s*(?:name|owner|copyright holder)[^&]*&gt;")
# "Copyright" as a code label, unless a quoted value (the statement) follows.
CODE_LABEL_RE = re.compile(
    r"\bcopyright(?:ed|s)?['\"]\s*(?:\]\s*)?(?:[);.}+]|(?:[,:]|=+>?)(?![=>])(?!\s*(?:string\(\d+\)\s*)?['\"]))",
    re.IGNORECASE)
MAX_STATEMENT_CHARS = 4096
# "see file COPYING" and similar: a pointer, not a holder.
POINTER_RE = re.compile(
    r"\bsee\s+(?:the\s+)?(?:(?:individual|accompanying|included|each)\s+)?(?:files?|copying|authors|"
    r"licen[cs]es?|notices?|readme|debian/copyright|above|below)\b", re.IGNORECASE)
# Legal forms stay with the holder ("Vestas Wind Systems A/S").
LEGAL_FORMS = (r"(?:A/S|ASA|AB|Oyj?|g?GmbH|AG|SE|KGaA|KG|S\.A\.?|S\.L\.?|S\.p\.A\.?|B\.V\.?|N\.V\.?|Inc\.?"
               r"|Incorporated|Ltd\.?|Limited|LLC|LLP|Corp\.?|Corporation|plc|PLC|Pty(?: Ltd\.?)?|e\.\s?V\.?)")
LEGAL_FORM_TAIL_RE = re.compile(r"[ \t]*,?[ \t]*" + LEGAL_FORMS + r"(?![\w/])")
LEGAL_FORM_END_RE = re.compile(LEGAL_FORMS + r"$")
# Words that follow "Copyright" in prose and headings, never as a holder.
_NOT_A_HOLDER = {
    "copyright", "copyrights", "copyrighted", "notice", "notices", "holder", "holders", "owner", "owners",
    "information", "status", "act", "treaty",
    "law", "laws", "license", "licence", "licensing", "statement", "statements", "disclaimer", "disclaimers",
    "text", "year", "years", "date", "dates", "all", "rights", "reserved", "office", "policy", "section",
    "note", "notes", "the", "this", "see", "file", "files", "page", "symbol", "sign", "header", "block",
    "line", "string", "and", "or", "of", "by", "for", "in", "on", "to", "is", "are", "not", "be", "contributors",
}
_HOLDER_LEAD_RE = re.compile(r"(?:\s|[:,.\-–]|\([cC]\)|©|\d)*")
_HOLDER_STOP_RE = re.compile(r"[,;<>()\[\]{}\"]|\.\s|\.$")
# Control characters mean binary; replacement characters only by density.
_CONTROL_RE = re.compile("[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")
_REPLACEMENT_RE = re.compile("[\ufffd\xfe\xff]")
# Licence and permission text that trails a statement in a source header.
_TRAILER_RE = re.compile(
    r"\s*(?:\ball rights? reserved\b|\bSPDX-License-Identifier\b|\blicen[cs]ed under\b|\bLicen[cs]e\s*:"
    r"|\bpermission is hereby\b"
    r"|\bredistribution and use\b|\bthis (?:file|program|library|software) is (?:free|part|distributed)\b"
    r"|\bdistributed under\b|\byou (?:can|may) redistribute\b).*$", re.IGNORECASE | re.DOTALL)
# Markup: inline elements flow within a line, others split fields; <email>/<URL> are kept.
_ENTITY_RE = re.compile(r"&(?:#\d+|#x[0-9a-fA-F]+|[A-Za-z]+);")
# Inline HTML and DocBook elements.
_INLINE = (r"(?:a|b|i|u|s|em|strong|span|font|small|big|sup|sub|img|abbr|acronym|code|tt|strike|q"
           r"|email|ulink|link|uri|emphasis|literal|phrase|quote|citetitle|trademark|productname|application"
           r"|holder|year|personname|firstname|surname|othername|honorific|lineage|orgname)")
# Perl POD escapes.
_POD_ESCAPE_RE = re.compile(r"E<(lt|gt)>")
# troff escapes, also doubled in C strings; "\&#353;" is an entity and left alone.
_TROFF_ESCAPES = {"(co": "\u00a9", "-": "-", "&": "", "e": "\\", "(ss": "\u00df",
                  "(:a": "\u00e4", "(:o": "\u00f6", "(:u": "\u00fc", "(:A": "\u00c4", "(:O": "\u00d6",
                  "(:U": "\u00dc", "(:e": "\u00eb"}
_TROFF_RE = re.compile(r"\\{1,2}(?:(\(co|\(ss|\(:[aouAOUe]|-|e)|&(?!#?\w+;)|f(?:[BIRP]|\(\w\w))")
_C_NEWLINE_RE = re.compile(r"\\n(?=[\s\"']|$)")
_END_OF_RE = re.compile(r"\s+End of (?:generated|comments?|file)\b.*$", re.IGNORECASE | re.DOTALL)
# "/ MIT Licence ..." after the holder; "Copyright 2000 MIT" is a holder and stays.
_LICENCE_NAME_TAIL_RE = re.compile(
    r"\s+[/|\u2013-]\s+(?:MIT|BSD|ISC|Apache|[AL]?GPL|MPL|EPL|CC0|CC-BY|Unlicense|Zlib|WTFPL)(?:-[\w.]+)*\b"
    r"(?=\s*(?:licen[cs]e\b|v?\d|https?:|$)).*$", re.IGNORECASE | re.DOTALL)
_M4_DNL_RE = re.compile(r"(?<![\w$-])dnl(?![\w-])")
_COPYRIGHT_ELEMENT_RE = re.compile(r"(?:<|(?:^|\s)/?)copyright(?=[\s/>])", re.IGNORECASE)
# A tag never spans a line; its name has no dot ("<www.ambiq.com>" is a URL).
_ATTRS = r"(?:[ \t]+[\w:-]+(?:=(?:\"[^\"<>\n]*\"?|'[^'<>\n]*'?|[^\s<>]+))?)*[ \t]*/?"
_NAME = r"[A-Za-z][\w:-]*(?=[\s/>]|$)"
# Without "<", only attributes with values make a tag: "Jane Doe td>" is not one.
_STRICT_ATTRS = r"(?:[ \t]+[\w:-]+=(?:\"[^\"<>\n]*\"?|'[^'<>\n]*'?|[^\s<>]+))*[ \t]*/?"
_INLINE_TAG_RE = re.compile(r"</?" + _INLINE + r"(?=[\s/>]|$)" + _ATTRS + r"(?:>|$)", re.IGNORECASE)
_TAG_RE = re.compile(r"<(?![^<>]*@)(?!\s*(?:https?|ftp)://)/?" + _NAME + _ATTRS + r"(?:>|$)")
# Tag remains ("td>") after a space or at the start, never after "." or "@" (emails).
_INLINE_REMAINS_RE = re.compile(r"(?:(?<=\s)|^)/?" + _INLINE + r"(?=[\s/>])" + _STRICT_ATTRS + r">", re.IGNORECASE)
_REMAINS_RE = re.compile(r"(?:(?<=\s)|^)/?" + _NAME + _STRICT_ATTRS + r">")
_ATTR_END_RE = re.compile(r"^[\w-]*[\"']\s*/?>|[\"']\s*/?>")
# Postal address after the holder; a year is never a house number.
_ADDRESS_TAIL_RE = re.compile(
    r"[,.]\s*(?:(?!(?:19|20)\d\d\b)\d{1,5}[A-Za-z]?,?\s+(?:[A-Z][\w'.-]*\s+){0,4}"
    r"(?:Street|St\.|Avenue|Ave\.?|Road|Rd\.|Place|Pl\.|Boulevard|Blvd\.?|Drive|Dr\.|Lane|Ln\.|Way|Square|Sq\."
    r"|Court|Ct\.|Parkway|Pkwy\.?|Highway|Hwy\.?|Strasse|Straße|Str\.)\b|(?:P\.?\s?O\.?\s*Box|Postfach)\s+\d+)"
    r".*$", re.DOTALL)
# Everything but a holder; "(c)" must not leave a "c".
_HOLDERLESS_RE = re.compile(r"\([cC]\)|copyright(?:s|ed)?|©|all rights reserved|\d+|[^\w(]+|\((?![cC]\))|_",
                            re.IGNORECASE)


def cut_at_comment_close(text):
    """Text before "*/" if it names a holder, else the whole text."""
    at = text.find("*/")
    if at < 0:
        return text
    head = text[:at].rstrip(" /*\t")
    return head if _HOLDERLESS_RE.sub("", head) else text


def holderless(text):
    """True if the statement names nobody ("Copyright -> 2004")."""
    return not _HOLDERLESS_RE.sub("", cut_at_comment_close(text))


def needs_review(text, rejected, uncertain, rejected_if_cut=None):
    """Uncertain, context-dependent, accepted without a holder, or rejected only for code after "*/"."""
    return bool(uncertain or context_dependent(text) or (not rejected and holderless(text))
                or (rejected and rejected_if_cut is False))


def context_dependent(text):
    """Placeholder year beside a real holder: only the file can decide."""
    return bool(PLACEHOLDER_YEAR_RE.search(text)) and not PLACEHOLDER_HOLDER_RE.search(text)


def looks_binary(text):
    """True for text decoded from binary data rather than written by a person."""
    if len(_CONTROL_RE.findall(text)) >= 2:
        return True
    replaced = len(_REPLACEMENT_RE.findall(text))
    return replaced >= 4 and replaced / max(len(text), 1) >= 0.2


def _names_someone(entity):
    """False for an entity without letters or made only of copyright vocabulary."""
    words = re.findall(r"[^\W\d_]+", entity.lower())
    return any(w not in _NOT_A_HOLDER for w in words)


def _holder_start(text, start):
    """Extend an entity back over up to three name words after the marker."""
    markers = [m for m in COPYRIGHT_MARKER_RE.finditer(text, 0, start) if m.end() <= start]
    if not markers:
        return start
    lead = _HOLDER_LEAD_RE.match(text, markers[-1].end()).end()
    gap = text[lead:start]
    words = gap.split()
    if (lead < start and 1 <= len(words) <= 3 and not re.search(r"[,;:<>()\[\]{}\"]", gap)
            and all(w.lower().strip(".") not in _NOT_A_HOLDER for w in words)):
        return lead
    return start


def _fallback_holder_span(text):
    """Holder span after the marker when NER finds none: ends in a legal form or is 2+ capitalised words."""
    marker = COPYRIGHT_MARKER_RE.search(text)
    if not marker:
        return None
    start = _HOLDER_LEAD_RE.match(text, marker.end()).end()
    stop = _HOLDER_STOP_RE.search(text, start)
    run = text[start:stop.start() if stop else len(text)]
    words = run.split()[:6]
    if not words:
        return None
    end = start + run.find(words[-1]) + len(words[-1])
    if LEGAL_FORM_END_RE.search(" ".join(words)):
        return start, end
    if len(words) >= 2 and all(w[0].isupper() and w.lower().strip(".") not in _NOT_A_HOLDER for w in words):
        return start, end
    return None


def _extract_if_needed(resource_ref, target_dir):
    """
    Ensures resource is available as a real directory on disk.
    Extraction occurs only once.
    """
    if os.path.exists(target_dir) and os.listdir(target_dir):
        return target_dir

    os.makedirs(target_dir, exist_ok=True)

    with as_file(resource_ref) as src_path:
        if src_path.is_dir():
            shutil.copytree(src_path, target_dir, dirs_exist_ok=True)
        else:
            shutil.copy2(src_path, target_dir)

    return target_dir


def _cache_dir(resource_ref, kind):
    with as_file(resource_ref) as src_path:
        return os.path.join(BASE_CACHE_DIR, "%s-%s" % (kind, tree_digest(src_path)[:16]))


def get_default_model_dir():
    return _extract_if_needed(MODELS_REF, _cache_dir(MODELS_REF, "models"))


def get_configs_dir():
    return _extract_if_needed(CONFIGS_REF, _cache_dir(CONFIGS_REF, "configs"))


class SafaaAgent:
    def __init__(self, use_local_model=True, model_dir=None):
        """
        Initializes the SafaaAgent with the necessary models.

        Parameters:
        use_local_model (bool): Flag to use local models if available.
        model_dir (str): Custom directory path for models. Defaults to None.
        """

        # Determine the model directory based on the provided arguments
        model_dir = (
            model_dir
            if model_dir
            else (
                LOCAL_MODEL_DIR
                if use_local_model and os.path.exists(LOCAL_MODEL_DIR)
                else get_default_model_dir()
            )
        )

        # Construct the file paths for each model
        self.false_positive_detector_path = os.path.join(
            model_dir, "false_positive_detection_model_sgd.pkl"
        )
        self.vectorizer_path = os.path.join(
            model_dir, "false_positive_detection_vectorizer.pkl"
        )
        self.entity_recognizer_path = os.path.join(
            model_dir, "entity_recognizer"
        )
        self.declutter_model_path = os.path.join(model_dir, "declutter_model")
        self.metadata_path = os.path.join(model_dir, METADATA_FILE)

        # Load the models from the constructed file paths
        self._load_models()

    def _load_models(self):
        """
        Loads models from file paths.
        """

        # Load the False Positive Detector model
        self.false_positive_detector = load(self.false_positive_detector_path)

        # Load the Vectorizer model
        self.vectorizer = load(self.vectorizer_path)

        # Load the Entity Recognizer model using spaCy
        self.entity_recognizer = spacy.load(self.entity_recognizer_path)

        # Load the Declutter model using spaCy
        self.declutter_model = spacy.load(self.declutter_model_path)

        self.metadata = {}
        if os.path.exists(self.metadata_path):
            with open(self.metadata_path, encoding="utf-8") as fh:
                self.metadata = json.load(fh)
        self.preprocessing_version = int(self.metadata.get("preprocessing_version", 1))
        if self.preprocessing_version > PREPROCESSING_VERSION:
            raise ValueError(
                f"Model in {self.metadata_path} needs preprocessing version "
                f"{self.preprocessing_version}; this safaa supports up to {PREPROCESSING_VERSION}. Upgrade safaa."
            )
        expected = self.metadata.get("preprocessing_fingerprint")
        actual = preprocessing_fingerprint(self.entity_recognizer_path, self.preprocessing_version)
        if expected and expected != actual:
            raise ValueError(
                f"The detector in {self.metadata_path} was trained with preprocessing {expected}, but this "
                f"preprocessing and entity recognizer give {actual}. "
                "Retrain the detector (scripts/train_and_release.py) after changing either."
            )

    def preprocess_data(self, data):
        """
        Preprocesses the given data by performing various text cleaning and
        transformation tasks.

        Parameters:
        data (iterable): The data to preprocess.

        Returns:
        data (list): List of preprocessed strings.
        """

        # Ensure the data is a list of strings
        data = self._ensure_list_of_strings(data)

        if self.preprocessing_version >= 2:
            return self._preprocess_v2(data)

        # Replace copyright holder entities in the data
        data = self._replace_entities(data)

        # Perform text substitutions for dates, numbers, symbols, emails, etc.
        data = self._perform_text_substitutions(data)

        return data

    def _preprocess_v2(self, data):
        """Replace holders, emails, placeholders, pointers and code labels by tokens."""
        out = []
        for text, doc in zip(data, self.entity_recognizer.pipe(data, batch_size=256)):
            spans = []  # (priority, start, end, token); lower wins
            for regex, token, prio in ((SPDX_TAG_RE, " copyright ", 0), (HTML_COPYRIGHT_RE, " COPYRIGHTSYMBOL ", 0),
                                       (CODE_LABEL_RE, " CODELABEL ", 0),
                                       (PLACEHOLDER_HOLDER_RE, " HOLDERPLACEHOLDER ", 1),
                                       (PLACEHOLDER_YEAR_RE, " YEARPLACEHOLDER ", 1), (POINTER_RE, " POINTER ", 1),
                                       (EMAIL_RE, " EMAIL ", 2)):
                spans += [(prio, m.start(), m.end(), token) for m in regex.finditer(text)]
            ents = [(3, _holder_start(text, e.start_char), e.end_char, " ENTITY ")
                    for e in doc.ents if e.label_ == "ENT" and _names_someone(e.text)]
            spans += ents
            if not ents:
                fallback = _fallback_holder_span(text)
                if fallback:
                    spans.append((4, fallback[0], fallback[1], " ENTITY "))
            chosen = []
            for span in sorted(spans):
                if all(span[2] <= c[1] or span[1] >= c[2] for c in chosen):
                    chosen.append(span)
            pieces, at = [], 0
            for _, start, end, token in sorted(chosen, key=lambda c: c[1]):
                pieces += [text[at:start], token]
                at = end
            pieces.append(text[at:])
            sentence = "".join(pieces)
            sentence = YEAR_RE.sub(" DATE ", sentence)
            sentence = re.sub(r"\d+", " ", sentence)
            sentence = re.sub(r"©|\([cC]\)", " COPYRIGHTSYMBOL ", sentence)
            sentence = re.sub(r"[^a-zA-Z0-9]", " ", sentence)
            if looks_binary(text):
                sentence += " BINARYNOISE"
            out.append(" ".join(sentence.lower().split()))
        return out

    def _ensure_list_of_strings(self, data):
        """
        Ensures the data is a list of strings.

        If the input data is not a list, attempts to convert it to a list.
        Then, ensures each element of the list is a string.

        Parameters:
        data (iterable): The data to be converted to a list of strings.

        Returns:
        list: A list of strings.
        """

        if isinstance(data, str):
            raise TypeError("expected an iterable of statements, not a single string")
        return [str(item)[:MAX_STATEMENT_CHARS] for item in data]

    def _replace_entities(self, data):
        """
        Replaces detected copyright holder entities with ' ENTITY '.

        Uses the entity_recognizer model to identify copyright holder entities,
        which are often name or organization entities, and replaces them with
        the string ' ENTITY '.

        Parameters:
        data (list): A list of strings.

        Returns:
        list: A list of strings with copyright holder entities replaced.
        """

        new_data = []
        for doc in self.entity_recognizer.pipe(data, batch_size=256):
            new_sentence = doc.text
            for entity in doc.ents:
                # If the entity is a copyright holder entity, replace it with
                # ' ENTITY '
                if entity.label_ == "ENT":
                    new_sentence = re.sub(
                        re.escape(entity.text), " ENTITY ", new_sentence
                    )
            new_data.append(new_sentence)
        return new_data

    def _perform_text_substitutions(self, data):
        """
        Performs a series of text substitutions to clean and standardize the
        data.

        This includes:
        - Replacing four-digit numbers (assumed to be years) with ' DATE '.
        - Removing all other numbers.
        - Replacing copyright symbols with ' COPYRIGHTSYMBOL '.
        - Replacing emails with ' EMAIL '.
        - Removing any special characters not already replaced or removed.
        - Converting text to lowercase.
        - Stripping extra whitespace from the text.

        Parameters:
        data (list): A list of strings.

        Returns:
        list: A list of cleaned and standardized strings.
        """

        # Define the substitution patterns and their replacements
        subs = [
            (r"\d{4}", " DATE "),
            (r"\d+", " "),
            (r"©", " COPYRIGHTSYMBOL "),
            (r"\(c\)", " COPYRIGHTSYMBOL "),
            (r"\(C\)", " COPYRIGHTSYMBOL "),
            (
                r"""(?:[a-z0-9!#$%&'*+/=?^_`{|}~-]+(?:\.[a-z0-9!#$%&'*+/=?^_`{|}~-]+)*|"
                (?:[\x01-\x08\x0b\x0c\x0e-\x1f\x21\x23-\x5b\x5d-\x7f]|
                \\[\x01-\x09\x0b\x0c\x0e-\x7f])*")@
                (?:(?:[a-z0-9](?:[a-z0-9-]*[a-z0-9])?\.)+[a-z0-9]
                (?:[a-z0-9-]*[a-z0-9])?|\[(?:(?:(2(5[0-5]|[0-4][0-9])|
                1[0-9][0-9]|[1-9]?[0-9]))\.){3}(?:(2(5[0-5]|[0-4][0-9])|
                1[0-9][0-9]|[1-9]?[0-9])|[a-z0-9-]*[a-z0-9]:(
                [\x01-\x08\x0b\x0c\x0e-\x1f\x21-\x5a\x53-\x7f]|
                \\[\x01-\x09\x0b\x0c\x0e-\x7f])+)])""",
                " EMAIL ",
            ),
            (r"[^a-zA-Z0-9]", " "),
        ]
        # Perform the substitutions for each pattern in the list
        for pattern, replacement in subs:
            data = [
                re.sub(pattern, replacement, sentence) for sentence in data
            ]
        # Convert text to lowercase and strip extra whitespace
        return [sentence.lower().strip() for sentence in data]

    def predict(self, data, threshold=0.5):
        """
        Predicts false positives in the given data.

        Parameters:
        data (iterable): The data to predict.
        threshold (float): The probability threshold for classification.
                           Defaults to 0.5.

        Returns:
        list: The predictions.
        """

        if self.metadata.get("calibration"):
            return ["f" if p > threshold else "t" for p in self._false_positive_probabilities(data)]

        # Preprocess the data before making predictions
        data = self.preprocess_data(data)

        # Vectorize the preprocessed data using the pre-trained vectorizer
        data = self.vectorizer.transform(data)

        # Check if the model supports probability prediction
        if hasattr(self.false_positive_detector, "predict_proba"):
            # Get probability predictions from the model
            predictions = self.false_positive_detector.predict_proba(data)
            # Classify based on the given threshold. If the threshhold is not
            # met, automatically sets the prediction to true
            return [
                "f" if prediction[1] >= threshold else "t"
                for prediction in predictions
            ]

        # Get binary predictions from the model if probability prediction is not
        # supported
        return [
            "f" if prediction == 1 else "t"
            for prediction in self.false_positive_detector.predict(data)
        ]

    def _false_positive_probabilities(self, data):
        """Calibrated false-positive probability; 0.5 is the decision boundary."""
        X = self.vectorizer.transform(self.preprocess_data(data))
        slope = float(self.metadata["calibration"]["slope"])
        return [1.0 / (1.0 + math.exp(-max(min(slope * s, 50.0), -50.0)))
                for s in self.false_positive_detector.decision_function(X)]

    def predict_with_scores(self, data, threshold=0.5):
        """
        Predicts false positives with a confidence and a review flag.

        Parameters:
        data (iterable): The data to predict.
        threshold (float): Probability of being a false positive above which
                           an item is predicted false ("f"). Defaults to 0.5.

        Returns:
        list of dict: {"prediction": "t" or "f",
                       "false_positive_probability": float or None,
                       "needs_review": bool}  (see needs_review)
        """
        data = self._ensure_list_of_strings(data)
        cut = {i: c for i, c in enumerate(map(cut_at_comment_close, data)) if c != data[i]}
        if not self.metadata.get("calibration"):
            predictions = self.predict(data, threshold)
            cut_rejected = dict(zip(cut, (p == "f" for p in self.predict(list(cut.values()), threshold)))) \
                if cut else {}
            return [{"prediction": p, "false_positive_probability": None,
                     "needs_review": needs_review(t, p == "f", False, cut_rejected.get(i))}
                    for i, (t, p) in enumerate(zip(data, predictions))]
        low, high = self.metadata.get("review_band", [0.5, 0.5])
        cut_rejected = dict(zip(cut, (p > threshold for p in self._false_positive_probabilities(list(cut.values()))))) \
            if cut else {}
        out = []
        for i, (t, p) in enumerate(zip(data, self._false_positive_probabilities(data))):
            out.append({"prediction": "f" if p > threshold else "t", "false_positive_probability": round(p, 4),
                        "needs_review": needs_review(t, p > threshold, low <= p <= high, cut_rejected.get(i))})
        return out

    def declutter(self, data, predictions):
        """
        Cleans up a copyright notice by removing extra text based on the
        predictions.

        A statement predicted real is never emptied and never loses its holder.

        Parameters:
        data (iterable): The data to declutter.
        predictions (list): The predictions indicating false positives.

        Returns:
        list: The decluttered data.
        """

        data = self._ensure_list_of_strings(data)
        predictions = list(predictions)
        if len(predictions) != len(data):
            raise ValueError(f"{len(data)} statements but {len(predictions)} predictions")
        out = [""] * len(data)
        keep = [i for i, p in enumerate(predictions) if p != "f"]
        texts = [self._demarkup(self._desource(cut_at_comment_close(data[i]))) for i in keep]
        docs = self.declutter_model.pipe(texts, batch_size=256)
        for i, text, doc in zip(keep, texts, docs):
            out[i] = self._decluttered(text, doc)
        return out

    @staticmethod
    def _desource(text):
        """Remove troff and C escapes, m4 words, trailers and an unmatched ")"."""
        if "\\" in text:
            text = _TROFF_RE.sub(lambda m: _TROFF_ESCAPES.get(m.group(1) or "&", ""), text)
            text = _C_NEWLINE_RE.sub("", text).rstrip()
        if "dnl" in text:
            text = " ".join(_M4_DNL_RE.sub(" ", text).split())
        for tail in (_END_OF_RE, _LICENCE_NAME_TAIL_RE):
            head = tail.sub("", text)
            if head != text and _HOLDERLESS_RE.sub("", head):
                text = head
        while text.endswith(")") and text.count(")") > text.count("("):
            text = text[:-1].rstrip()
        return text

    @staticmethod
    def _demarkup(text):
        """First marked line of the text with markup removed (plus the next if it lacks a holder)."""
        if not (_ENTITY_RE.search(text) or "<" in text or ">" in text):
            return text
        text = _POD_ESCAPE_RE.sub(lambda m: "<" if m.group(1) == "lt" else ">", text)
        named = bool(_COPYRIGHT_ELEMENT_RE.search(text))
        text = _INLINE_TAG_RE.sub(" ", text)
        text = _TAG_RE.sub("\n", text)
        text = _INLINE_REMAINS_RE.sub(" ", text)
        text = _REMAINS_RE.sub("\n", text)
        text = _ATTR_END_RE.sub(" ", text)
        lines = [" ".join(html.unescape(line).split()) for line in text.split("\n")]
        lines = [line for line in lines if line]
        at = next((k for k, line in enumerate(lines) if COPYRIGHT_MARKER_RE.search(line)), None)
        if at is None:
            # Only a <copyright> element carried the marker.
            joined = " ".join(lines)
            return "Copyright " + joined if joined and named else joined
        statement = lines[at]
        if not _HOLDERLESS_RE.sub("", COPYRIGHT_MARKER_RE.sub("", statement)) and at + 1 < len(lines):
            statement += " " + lines[at + 1]
        return statement

    @staticmethod
    def _strip_address_tail(statement):
        """The statement without a postal address after it, unless that would take the holder."""
        stripped = _ADDRESS_TAIL_RE.sub("", statement).rstrip(" ,;:-")
        return stripped if _HOLDERLESS_RE.sub("", stripped) else statement

    @staticmethod
    def _decluttered(text, doc):
        """The statements the declutter model found, or the safe fallback."""
        parts = []
        for ent in doc.ents:
            end = ent.end_char
            tail = LEGAL_FORM_TAIL_RE.match(text, end)
            if tail:
                end = tail.end()
            parts.append(text[ent.start_char:end])
        result = " ".join(parts)
        anchored = [rx for rx in (COPYRIGHT_MARKER_RE, YEAR_RE) if rx.search(text)]
        if _HOLDERLESS_RE.sub("", result) and (not anchored or any(rx.search(result) for rx in anchored)):
            return SafaaAgent._strip_address_tail(result)
        # No holder found: keep the statement, minus trailing licence text.
        trimmed = _TRAILER_RE.sub("", text).strip().rstrip(",;:-").strip()
        return trimmed if _HOLDERLESS_RE.sub("", trimmed) else text.strip()

    def train_false_positive_detector_model(self, data, labels):
        """
        Trains the false positive detector model iteratively.

        Parameters:
        data (iterable): The data to train the model on.
        labels (iterable): The labels for the training data.
        """

        # Preprocess the data before training
        preprocessed_data = self.preprocess_data(data)
        # Fit the vectorizer to the preprocessed data
        vectorized_data = self.vectorizer.transform(preprocessed_data)
        # Train the false positive detector model
        self.false_positive_detector.partial_fit(vectorized_data, labels)
        # Released calibration and results no longer apply.
        for key in ("calibration", "review_band", "decision_shift", "held_out", "held_out_novel",
                    "shipped_on_held_out", "cv", "review", "regression"):
            self.metadata.pop(key, None)
        if self.metadata:
            self.metadata["updated_incrementally"] = True

    def train_ner_model(
        self, train_path, dev_path, declutter_model=False, config_path=None, output_dir=None
    ):
        """
        Trains the named entity recognition model using the provided training and
        development datasets.

        Parameters:
        train_path (str): The path to the training data file. Should be a .spacy file
        dev_path (str): The path to the development data file. Should be a .spacy file
        declutter_model (bool): Whether to train the declutter model or the entity recognizer model. Defaults to False.
        config_path (str): The path to the configuration file. Defaults to None.
        output_dir (str): Model directory to write into. Defaults to the package's own models.
        """

        # Determine the configuration file path
        cfg_path = config_path or get_configs_dir()
        config_file_path = os.path.join(cfg_path, "train.cfg")

        # Read the configuration file
        with open(config_file_path, "r", encoding="utf-8") as file:
            cfg_contents = file.read()

        # Update the training and development data paths in the configuration
        # contents
        for path in (train_path, dev_path):
            if '"' in str(path) or "\n" in str(path):
                raise ValueError(f"unsupported character in path: {path!r}")
        updated_cfg_contents = re.sub(
            r'train\s*=\s*".*"',
            lambda _: f'train = "{train_path}"',
            re.sub(r'dev\s*=\s*".*"', lambda _: f'dev = "{dev_path}"', cfg_contents),
        )

        # Scratch dir: the config dir may be read-only; a failure leaves no partial model.
        work = tempfile.mkdtemp(prefix="safaa-train-")
        try:
            tmp_cfg_path = os.path.join(work, "train.cfg")
            with open(tmp_cfg_path, "w", encoding="utf-8") as file:
                file.write(updated_cfg_contents)
            tmp_model_path = os.path.join(work, "output")

            subprocess.run(
                [sys.executable, "-m", "spacy", "train", tmp_cfg_path, "--output", tmp_model_path],
                check=True,
            )

            new_model_dir = (
                "declutter_model" if declutter_model else "entity_recognizer"
            )
            new_model_path = os.path.join(output_dir or LOCAL_MODEL_DIR, new_model_dir)
            os.makedirs(new_model_path, exist_ok=True)

            # Move the trained model files to the new model directory
            self._move_files(
                os.path.join(tmp_model_path, "model-best"), new_model_path
            )
        finally:
            shutil.rmtree(work, ignore_errors=True)

    def _move_files(self, src_dir, dst_dir):
        """
        Internal helper method to move files from one directory to another.

        Parameters:
        src_dir (str): The source directory from where the files are to be moved.
        dst_dir (str): The destination directory where the files are to be moved.
        """

        # List all items in the source directory
        for item in os.listdir(src_dir):
            # Construct full paths for source and destination
            src_item_path = os.path.join(src_dir, item)
            dst_item_path = os.path.join(dst_dir, item)
            # Replace rather than nest (ner/ner).
            if os.path.isdir(dst_item_path):
                shutil.rmtree(dst_item_path)
            elif os.path.exists(dst_item_path):
                os.remove(dst_item_path)
            shutil.move(src_item_path, dst_item_path)

    def save(self, path=None):
        """
        Saves the trained models and vectorizer to the specified path.

        Make sure the directory you have permissions to the directory that you
        are saving to.

        Parameters:
        path (str): The path to save the models and vectorizer to. Defaults to
                    None.
        """

        # Determine the directory path to save the models and vectorizer
        save_path = path or "/home/fossy/Safaa"

        # Create the directory if it does not exist
        os.makedirs(save_path, exist_ok=True)

        # Check directory permissions
        if not os.access(save_path, os.W_OK):
            print(
                "Write permissions are not granted for the directory: "
                f"{save_path}"
            )
            return

        # Construct the full paths for the model and vectorizer files
        false_positive_detector_path = os.path.join(
            save_path, "false_positive_detection_model_sgd.pkl"
        )
        vectorizer_path = os.path.join(
            save_path, "false_positive_detection_vectorizer.pkl"
        )

        # Save the false positive detector model and vectorizer to the specified
        # paths
        dump(self.false_positive_detector, false_positive_detector_path)
        dump(self.vectorizer, vectorizer_path)
        if self.metadata:
            with open(os.path.join(save_path, METADATA_FILE), "w", encoding="utf-8") as fh:
                json.dump(self.metadata, fh, indent=2)
