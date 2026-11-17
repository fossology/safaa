#!/usr/bin/env python3

# SPDX-FileCopyrightText: © 2026 Fossology contributors
# SPDX-License-Identifier: LGPL-2.1-only
"""
Retrain the declutter model on the statements OSSelot reviewers kept (minus a
trailing "All rights reserved") plus datasets/json/declutter_dataset.jsonl, and
score shipped and candidate on the detector's held-out packages:
lost (holder missing), kept, junk (words not kept), exact.

    python scripts/retrain_declutter.py --findings <osselot findings.csv> \\
        --report <retrain report.json> --out <dir>
"""

import argparse
import json
import logging
import os
import random
import re
import shutil
import sys
import tempfile
import warnings
from datetime import datetime, timezone

import spacy
from spacy.tokens import DocBin

from safaa.Safaa import LOCAL_MODEL_DIR, SafaaAgent

LOG = logging.getLogger("retrain_declutter")
ARR_RE = re.compile(r"[\s,.;:|-]*\ball rights? reserved\b.*$", re.IGNORECASE | re.DOTALL)
BOILERPLATE = {"copyright", "copyrights", "c", "all", "rights", "right", "reserved", "spdx", "filecopyrighttext",
               "the", "by", "and", "of"}


def convention(statement):
    """A reviewer's statement as declutter should produce it."""
    return ARR_RE.sub("", statement).strip()


def collapse(text):
    """(text with whitespace runs as one space, index of each kept char in text)."""
    out, where, space = [], [], False
    for i, ch in enumerate(text):
        if ch.isspace():
            if not space and out:
                out.append(" ")
                where.append(i)
            space = True
        else:
            out.append(ch)
            where.append(i)
            space = False
    return "".join(out), where


def locate(raw, statement):
    """(start, end) of a reviewer's statement inside the raw finding, or None."""
    want = convention(statement)
    if not want:
        return None
    hay, where = collapse(raw)
    needle = collapse(want)[0].strip()
    at = hay.find(needle)
    if at < 0:
        at = hay.lower().find(needle.lower())
    if at < 0:
        return None
    return where[at], where[at + len(needle) - 1] + 1


def words(text):
    return {w for w in re.findall(r"\w+", text.lower()) if w not in BOILERPLATE}


def score(agent, rows):
    """Declutter quality on real findings, through SafaaAgent and its safety net."""
    out = agent.declutter([r["text"] for r in rows], ["t"] * len(rows))
    norm = lambda s: " ".join(re.findall(r"\w+", s.lower()))  # noqa: E731
    lost = kept = junk = exact = 0
    for o, r in zip(out, rows):
        want = words(r["matched"])
        have = words(o)
        if want and not want & have:
            lost += 1
        if want <= have:
            kept += 1
        junk += len(have - want)
        exact += norm(o) == norm(convention(r["matched"]))
    n = len(rows)
    return {"n": n, "lost": lost, "kept": round(kept / n, 4), "junk_per_statement": round(junk / n, 4),
            "exact": round(exact / n, 4)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--findings", required=True, help="findings.csv from build_osselot_dataset.py")
    ap.add_argument("--report", required=True, help="report.json of the detector retrain (held-out packages)")
    ap.add_argument("--legacy", default="datasets/json/declutter_dataset.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-per-pattern", type=int, default=20)
    ap.add_argument("--eval-size", type=int, default=8000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    warnings.filterwarnings("ignore")
    os.makedirs(args.out, exist_ok=True)
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from retrain_false_positive_detector import pattern_key  # noqa: E402

    import csv
    with open(args.report, encoding="utf-8") as fh:
        test_pkgs = set(json.load(fh)["data"]["test_packages"])
    real = [r for r in csv.DictReader(open(args.findings, encoding="utf-8")) if r["label"] == "0" and r["matched"]]
    test = [r for r in real if r["package"] in test_pkgs]
    train_rows = [r for r in real if r["package"] not in test_pkgs]

    # --- training examples: every kept statement of the file found in the finding ---
    nlp = spacy.blank("en")
    examples, seen, per_key, skipped = [], set(), {}, 0
    rng = random.Random(args.seed)
    rng.shuffle(train_rows)
    for r in train_rows:
        if r["text"] in seen:
            continue
        seen.add(r["text"])
        key = pattern_key(r["text"])
        if per_key.get(key, 0) >= args.max_per_pattern:
            continue
        doc = nlp.make_doc(r["text"])
        spans = []
        for statement in [r["matched"]] + r["curated"].split(" | "):
            at = locate(r["text"], statement)
            span = doc.char_span(at[0], at[1], "Copyright", alignment_mode="expand") if at else None
            if span is not None and all(span.end <= s.start or span.start >= s.end for s in spans):
                spans.append(span)
        if not spans:
            skipped += 1
            continue
        doc.ents = sorted(spans, key=lambda s: s.start)
        examples.append(doc)
        per_key[key] = per_key.get(key, 0) + 1
    osselot_n = len(examples)
    for line in open(args.legacy, encoding="utf-8"):
        item = json.loads(line)
        doc = nlp.make_doc(item["text"])
        spans = [doc.char_span(s, e, "Copyright", alignment_mode="expand") for s, e, _ in item["label"]]
        spans = [s for s in spans if s is not None]
        try:
            doc.ents = spans
        except ValueError:
            continue
        examples.append(doc)
    rng.shuffle(examples)
    dev_n = max(500, len(examples) // 10)
    LOG.info("examples: %d osselot (%d unalignable skipped) + %d legacy | dev %d | held-out real findings %d",
             osselot_n, skipped, len(examples) - osselot_n, dev_n, len(test))

    work = tempfile.mkdtemp(prefix="safaa-declutter-")
    try:
        train_path, dev_path = os.path.join(work, "train.spacy"), os.path.join(work, "dev.spacy")
        DocBin(docs=examples[dev_n:]).to_disk(train_path)
        DocBin(docs=examples[:dev_n]).to_disk(dev_path)
        cand_dir = os.path.join(args.out, "models")
        os.makedirs(cand_dir, exist_ok=True)
        for name in os.listdir(LOCAL_MODEL_DIR):
            src, dst = os.path.join(LOCAL_MODEL_DIR, name), os.path.join(cand_dir, name)
            if name == "declutter_model" or os.path.exists(dst) or name.startswith("__"):
                continue
            (shutil.copytree if os.path.isdir(src) else shutil.copy2)(src, dst)
        shipped = SafaaAgent()
        shipped.train_ner_model(train_path, dev_path, declutter_model=True, output_dir=cand_dir)
    finally:
        shutil.rmtree(work, ignore_errors=True)

    rows = random.Random(5).sample(test, min(args.eval_size, len(test)))
    result = {"built": datetime.now(timezone.utc).isoformat(), "held_out_real_findings": len(rows),
              "shipped": score(SafaaAgent(), rows), "candidate": score(SafaaAgent(model_dir=cand_dir), rows),
              "train_examples": len(examples) - dev_n, "osselot_examples": osselot_n}
    with open(os.path.join(args.out, "declutter_report.json"), "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)
    LOG.info("shipped   %s", result["shipped"])
    LOG.info("candidate %s", result["candidate"])


if __name__ == "__main__":
    main()
