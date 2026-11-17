# SPDX-FileCopyrightText: © 2026 Fossology contributors
# SPDX-License-Identifier: LGPL-2.1-only
"""SafaaAgent with the shipped models: the FOSSology contract, declutter and regressions."""

import csv
import json
import os
import re

import pandas as pd
import pytest

from safaa.Safaa import METADATA_FILE, PREPROCESSING_VERSION, SafaaAgent, context_dependent

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BOILERPLATE = {"copyright", "copyrights", "c", "all", "rights", "right", "reserved", "the", "by", "and", "of"}


@pytest.fixture(scope="module")
def agent():
    return SafaaAgent()


@pytest.fixture(scope="module")
def cases():
    with open(os.path.join(ROOT, "datasets", "regression_cases.csv"), encoding="utf-8") as fh:
        return [r for r in csv.DictReader(fh) if not context_dependent(r["text"])]


def test_fossology_contract(agent):
    """src/decider/agent/copyrightDeactivationClutterRemovalScript.py, call for call."""
    df = pd.DataFrame({"content": ["Copyright (c) 2019 Jane Doe", "copyright notice, this list of conditions"]})
    df["is_copyright"] = agent.predict(df["content"], 0.5)
    assert list(df["is_copyright"]) == ["t", "f"]
    df["decluttered_content"] = agent.declutter(df["content"], df["is_copyright"])
    assert df["decluttered_content"][0].startswith("Copyright (c) 2019 Jane Doe")
    assert df["decluttered_content"][1] == ""


def test_predict_with_scores(agent):
    texts = ["Copyright (c) 2019 Jane Doe", "Copyright (C) YEAR Free Software Foundation"]
    out = agent.predict_with_scores(texts)
    assert [o["prediction"] for o in out] == agent.predict(texts)
    assert out[1]["needs_review"], "a placeholder year beside a real holder needs the file to decide"
    if agent.metadata.get("calibration"):
        assert all(0.0 <= o["false_positive_probability"] <= 1.0 for o in out)


def test_threshold_moves_decisions_when_calibrated(agent):
    if not agent.metadata.get("calibration"):
        pytest.skip("model has no calibration")
    texts = ["Copyright (c) 2019 Jane Doe", "copyright notice, this list of conditions"]
    assert agent.predict(texts, 0.0) == ["f", "f"]
    assert agent.predict(texts, 1.0) == ["t", "t"]


def test_regression_cases(agent, cases):
    """Every hand-verified case passes, except those the model records as known failures."""
    predicted = agent.predict([c["text"] for c in cases])
    failed = {c["text"] for c, p in zip(cases, predicted) if (p == "f") != (c["label"] == "1")}
    known = set(agent.metadata.get("regression", {}).get("known_failures", []))
    assert failed <= known, "newly failing: %s" % sorted(failed - known)


def test_model_reproduces_its_recorded_results(agent):
    """The shipped model reproduces its recorded held-out results with this preprocessing."""
    held = agent.metadata.get("held_out")
    osselot = os.path.join(ROOT, "datasets", "osselot_dataset.csv")
    if not held or not os.path.exists(osselot):
        pytest.skip("model has no recorded held-out results, or no OSSelot dataset")
    import random
    import sys
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    from retrain_false_positive_detector import cap_patterns

    meta = agent.metadata
    osl = pd.read_csv(osselot, keep_default_na=False)
    osl["label"] = osl["label"].astype(int)
    pkgs = osl["packages"].str.split(";")
    names = sorted({p for ps in pkgs for p in ps})
    test_pkgs = set(random.Random(meta["seed"]).sample(names, max(1, round(len(names) * meta["test_packages_share"]))))
    assert test_pkgs == set(meta["data"]["test_packages"]), "OSSelot dataset differs from the one trained on"
    test = osl[pkgs.apply(lambda ps: any(p in test_pkgs for p in ps)) & ~osl["text"].map(context_dependent)]
    test, _ = cap_patterns(test, meta["data"]["max_per_pattern"], meta["seed"])
    predicted = agent.predict(test["text"].tolist())
    real_rejected = sum(1 for p, y in zip(predicted, test["label"]) if y == 0 and p == "f")
    noise_missed = sum(1 for p, y in zip(predicted, test["label"]) if y == 1 and p == "t")
    assert (real_rejected, noise_missed) == (held["real_rejected"], held["noise_missed"])


def test_declutter_never_loses_a_copyright(agent, cases):
    """A statement predicted real comes back non-empty with its holder."""
    real = [c["text"] for c in cases if c["label"] == "0"]
    for text, out in zip(real, agent.declutter(real, ["t"] * len(real))):
        want = {w for w in re.findall(r"\w+", text.lower()) if w not in BOILERPLATE and not w.isdigit()}
        assert out.strip(), "emptied: %r" % text
        assert want & {w for w in re.findall(r"\w+", out.lower())}, "holder lost: %r -> %r" % (text, out)


@pytest.mark.parametrize("text,kept", [
    ("Copyright (c) 2023 Vestas Wind Systems A/S", "A/S"),
    ("Copyright (c) 2018 Nordic Semiconductor ASA", "ASA"),
    ("2007 - 2013, Nicolas François", "Nicolas François"),
    ("Copyright 2009, VMware, Inc. All Rights Reserved.", "VMware"),
])
def test_declutter_keeps_holder_and_legal_form(agent, text, kept):
    assert kept in agent.declutter([text], ["t"])[0]


@pytest.mark.parametrize("text,expected", [
    ("Copyright (C) 2003-2010 Frederico Caldeira Knabben</copyright",
     "Copyright (C) 2003-2010 Frederico Caldeira Knabben"),
    ("Copyright (c) 2010 Foo Inc.</p><br />", "Copyright (c) 2010 Foo Inc."),
    ("Copyright (c) 2007, Rick Root <rick@webworksllc.com>", "Copyright (c) 2007, Rick Root <rick@webworksllc.com>"),
    ("Copyright 2019 Foo <http://foo.org>", "Copyright 2019 Foo <http://foo.org>"),
    ("Copyright 2003-2010 <a href=", "Copyright 2003-2010 <a href="),  # stripping would leave no holder
])
def test_markup_after_a_statement(agent, text, expected):
    assert agent.declutter([text], ["t"])[0] == expected


@pytest.mark.parametrize("text,expected", [
    # XML fields: only the field holding the statement
    ('copyright>Copyright (C) 2003-2010 Frederico Caldeira Knabben</copyright> initialWindow> content>_samples/a.html'
     '</content> title>FCKeditor AIR Sample</title>', "Copyright (C) 2003-2010 Frederico Caldeira Knabben"),
    ('Copyright &copy; 2003-2010 <a href="#" onclick="SendEMail();">Frederico Caldeira Knabben</a> td> td align="c">',
     "Copyright © 2003-2010 Frederico Caldeira Knabben"),                       # holder in the link text
    # a postal address is clutter
    ("Copyright (C) 1989, 1991 Free Software Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110 USA",
     "Copyright (C) 1989, 1991 Free Software Foundation, Inc."),
    ("Copyright 2008-2009 Peter Krefting E<lt>peter@softwolves.pp.se>",
     "Copyright 2008-2009 Peter Krefting <peter@softwolves.pp.se>"),            # Perl POD escape
    ("copyright><year>1998-2001</year> <holder>Jason Gunthorpe</holder></copyright>",
     "Copyright 1998-2001 Jason Gunthorpe"),                                     # DocBook
    ("Copyright (c) 2024 Ambiq Micro Inc. <www.ambiq.com>", "Copyright (c) 2024 Ambiq Micro Inc. <www.ambiq.com>"),
    ("Copyright (c) 2024 Jeferson Fernando <jeferson.santos@edge.ufal.br>",
     "Copyright (c) 2024 Jeferson Fernando <jeferson.santos@edge.ufal.br>"),
    ("Copyright 2002 Square Inc.", "Copyright 2002 Square Inc."),              # a year is not a house number
])
def test_declutter_markup_and_addresses(agent, text, expected):
    assert agent.declutter([text], ["t"])[0] == expected


def test_declutter_drops_false_positives(agent):
    assert agent.declutter(["Copyright (c) 2019 Jane Doe"], ["f"]) == [""]


def test_model_newer_than_code_is_refused(tmp_path):
    src = os.path.dirname(SafaaAgent().vectorizer_path)
    for name in os.listdir(src):
        if not name.startswith("__"):
            os.symlink(os.path.join(src, name), tmp_path / name)
    meta = tmp_path / METADATA_FILE
    if meta.exists():
        meta.unlink()
    meta.write_text(json.dumps({"preprocessing_version": PREPROCESSING_VERSION + 1}))
    with pytest.raises(ValueError, match="Upgrade safaa"):
        SafaaAgent(model_dir=str(tmp_path))


def test_move_files_replaces_instead_of_nesting(tmp_path):
    src, dst = tmp_path / "src" / "ner", tmp_path / "dst" / "ner"
    src.mkdir(parents=True)
    dst.mkdir(parents=True)
    (src / "model").write_text("new")
    (dst / "model").write_text("old")
    SafaaAgent._move_files(None, str(tmp_path / "src"), str(tmp_path / "dst"))
    assert (dst / "model").read_text() == "new"
    assert not (dst / "ner").exists()


def test_any_iterable_of_statements(agent):
    statements = ("Copyright 2020 Jane Doe", "copyright notice, this list of conditions")
    assert agent.predict(statements) == agent.predict(list(statements)) == agent.predict(s for s in statements)
    with pytest.raises(TypeError):
        agent.predict("Copyright 2020 Jane Doe")          # a string is not a list of statements


def test_declutter_refuses_mismatched_predictions(agent):
    with pytest.raises(ValueError):
        agent.declutter(["Copyright 2020 Jane Doe", "Copyright 2021 John Roe"], ["t"])


def test_markup_without_a_copyright_element_is_not_made_a_statement():
    assert SafaaAgent._demarkup("<p>Jane Doe</p><p>2020</p>") == "Jane Doe 2020"
    assert SafaaAgent._demarkup("<copyright><year>1998</year><holder>Jason</holder></copyright>") \
        == "Copyright 1998 Jason"


def test_patterns_run_in_linear_time():
    import time
    import safaa.Safaa as S
    long_text = "copyright\"" + " " * 20000 + "x"
    started = time.time()
    S.CODE_LABEL_RE.search(long_text)
    assert time.time() - started < 0.5


def test_incremental_training_drops_released_calibration(tmp_path):
    agent = SafaaAgent()
    if not agent.metadata.get("calibration"):
        pytest.skip("model has no calibration")
    agent.train_false_positive_detector_model(["Copyright 2020 Jane Doe", "copyright notice, this list"], [0, 1])
    assert "calibration" not in agent.metadata and "held_out" not in agent.metadata
    assert agent.metadata["updated_incrementally"]
    agent.save(str(tmp_path))
    saved = json.loads((tmp_path / METADATA_FILE).read_text())
    assert "calibration" not in saved and saved["preprocessing_version"] == agent.preprocessing_version


def test_model_with_another_preprocessing_fingerprint_is_refused(tmp_path):
    src = os.path.dirname(SafaaAgent().vectorizer_path)
    for name in os.listdir(src):
        if not name.startswith("__") and name != METADATA_FILE:
            os.symlink(os.path.join(src, name), tmp_path / name)
    shipped = os.path.join(src, METADATA_FILE)
    meta = json.loads(open(shipped).read()) if os.path.exists(shipped) else {}
    meta["preprocessing_version"] = PREPROCESSING_VERSION
    meta["preprocessing_fingerprint"] = "v%d:0000000000000000" % PREPROCESSING_VERSION
    (tmp_path / METADATA_FILE).write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="Retrain the detector"):
        SafaaAgent(model_dir=str(tmp_path))


@pytest.mark.parametrize("text,expected", [
    # troff escapes in man pages, single or doubled as in C strings
    ("Copyright \\(co 2009 Red Hat, Inc.", "Copyright © 2009 Red Hat, Inc."),
    ("Copyright \\\\(co 2004 Robert M. Love.", "Copyright © 2004 Robert M. Love."),
    ("Copyright (C) 1992\\-2024 The University of Delaware", "Copyright (C) 1992-2024 The University of Delaware"),
    ("Copyright © 2009-2016 Inria\\&", "Copyright © 2009-2016 Inria"),
    ("Copyright \\fBGNU\\fP Project", "Copyright GNU Project"),
    # m4 comment words in autoconf files
    ("Copyright © 2004-2008 High Performance Computing Center Stuttgart, dnl University of Stuttgart",
     "Copyright © 2004-2008 High Performance Computing Center Stuttgart, University of Stuttgart"),
    ("Copyright (C) 2001-2020 Free Software Foundation, dnl Inc.",
     "Copyright (C) 2001-2020 Free Software Foundation, Inc."),
    # kept: author credits (reviewers keep them), "$dnl" in code
    ("Copyright (C) 2010-2021 Free Software Foundation, Inc. Written by Peter Rosin <peda@lysator.liu.se>.",
     "Copyright (C) 2010-2021 Free Software Foundation, Inc. Written by Peter Rosin <peda@lysator.liu.se>."),
    ("Copyright 2019 $dnl variable Holder", "Copyright 2019 $dnl variable Holder"),
])
def test_declutter_source_format_noise(agent, text, expected):
    assert agent.declutter([text], ["t"])[0] == expected


def test_troff_zero_width_space_spares_entities():
    assert SafaaAgent._desource("za\\&#353;tita") == "za\\&#353;tita"


@pytest.mark.parametrize("text,expected", [
    ("Copyright 1998,1999 Heiko Eissfeldt */ if 4 == SIZEOF_LONG_INT define UINT4 long",
     "Copyright 1998,1999 Heiko Eissfeldt"),
    ("Copyright 1994,2000-2002 J. Schilling*/", "Copyright 1994,2000-2002 J. Schilling"),
    # the holder is only after the comment close: nothing is cut
    ('copyright */ define ARES_COPYRIGHT "2004 - 2024 Daniel Stenberg"',
     'copyright */ define ARES_COPYRIGHT "2004 - 2024 Daniel Stenberg"'),
    ("Copyright 2019 Jane Doe", "Copyright 2019 Jane Doe"),
])
def test_cut_at_comment_close(text, expected):
    from safaa.Safaa import cut_at_comment_close
    assert cut_at_comment_close(text) == expected


@pytest.mark.parametrize("text,expected", [
    ("Copyright -> 2004", True), ("Copyright + 2001", True), ("Copyright (c) 1998-2003", True),
    ("Copyright (c) 2003 Jane", False), ("Copyright (C) 2003 (Acme)", False), ("Copyright 2019 Jane Doe */ x", False),
])
def test_holderless(text, expected):
    from safaa.Safaa import holderless
    assert holderless(text) is expected


def test_needs_review_rule():
    from safaa.Safaa import needs_review
    assert needs_review("Copyright 2019 Jane Doe", rejected=False, uncertain=True)
    assert not needs_review("Copyright 2019 Jane Doe", rejected=False, uncertain=False)
    assert needs_review("Copyright -> 2004", rejected=False, uncertain=False)          # accepted, names nobody
    assert not needs_review("Copyright -> 2004", rejected=True, uncertain=False)       # rejected noise stays rejected
    assert needs_review("Copyright 1998 X */ code", rejected=True, uncertain=False, rejected_if_cut=False)
    assert not needs_review("Copyright 1998 X */ code", rejected=True, uncertain=False, rejected_if_cut=True)


def test_statement_rejected_only_for_code_after_its_comment_is_reviewed(agent):
    """cdrkit mytype.h: the notice is real, the code after "*/" is not part of it."""
    text = "Copyright 1998,1999 Heiko Eissfeldt */ if 4 == SIZEOF_LONG_INT define UINT4 long unsigned int define UINT"
    scored = agent.predict_with_scores([text])[0]
    assert scored["prediction"] == "t" or scored["needs_review"]
    assert agent.declutter([text], ["t"])[0] == "Copyright 1998,1999 Heiko Eissfeldt"


@pytest.mark.parametrize("text,expected", [
    ("Copyright (C) 1987, 1995-2006 Joerg Schilling\\n", "Copyright (C) 1987, 1995-2006 Joerg Schilling"),
    ("Copyright 1999-2006 by J\\(:org Schilling", "Copyright 1999-2006 by Jörg Schilling"),
    ("Copyright 1998 J. Schilling)", "Copyright 1998 J. Schilling"),
    ("Copyright 1998-2004 Heiko Eissfeldt */ Global Variables */", "Copyright 1998-2004 Heiko Eissfeldt"),
    ("Copyright (c) 2018 Steven Benner (http://stevenbenner.com/).",
     "Copyright (c) 2018 Steven Benner (http://stevenbenner.com/)."),               # balanced brackets kept
])
def test_declutter_cdrkit_clutter(agent, text, expected):
    assert agent.declutter([text], ["t"])[0] == expected


@pytest.mark.parametrize("text,expected", [
    ("Copyright 2026, Internet Systems Consortium End of generated man page.",
     "Copyright 2026, Internet Systems Consortium"),
    ("Copyright 2019 End of Days Productions", "Copyright 2019 End of Days Productions"),      # a holder's name
])
def test_declutter_end_of_generated_text(agent, text, expected):
    assert agent.declutter([text], ["t"])[0] == expected


@pytest.mark.parametrize("text,expected", [
    ("(c) Chris Veness 2014-2019 / MIT Licence https://github.com/chrisveness/latlon-geohash",
     "(c) Chris Veness 2014-2019"),
    ("Copyright 2019 Jane Doe - Apache 2.0", "Copyright 2019 Jane Doe"),
    ("Copyright 2019 Jane Doe / BSD-3-Clause", "Copyright 2019 Jane Doe"),
    ("Copyright (c) 2010 - MIT", "Copyright (c) 2010 - MIT"),                   # MIT is the holder
    ("Copyright 2015 Foo - GPL Team Members", "Copyright 2015 Foo - GPL Team Members"),
])
def test_declutter_licence_name_after_statement(agent, text, expected):
    assert agent.declutter([text], ["t"])[0] == expected
