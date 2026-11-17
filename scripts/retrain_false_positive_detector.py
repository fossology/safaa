#!/usr/bin/env python3

# SPDX-FileCopyrightText: © 2026 Fossology contributors
# SPDX-License-Identifier: LGPL-2.1-only
"""
Retrain the false-positive detector from scratch.

Data:
  --legacy   the repository dataset (copyright,falsePositive)
  --osselot  dataset.csv from build_osselot_dataset.py (text,label,count,packages)

OSSelot packages are held out whole for testing. Grouped cross-validation picks
the configuration with the best noise recall at --min-real-recall. The
candidate is written to --out.
"""

import argparse
import hashlib
import itertools
import json
import logging
import math
import os
import pickle
import random
import re
import sys
import warnings
from datetime import datetime, timezone

import pandas as pd
import sklearn
from joblib import Parallel, delayed, dump
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GroupKFold

import safaa.Safaa as safaa_module
from safaa.Safaa import (METADATA_FILE, PREPROCESSING_VERSION, SafaaAgent, context_dependent, cut_at_comment_close,
                         needs_review,
                         preprocessing_fingerprint)

LOG = logging.getLogger("retrain")

GRID = {
    "loss": ["hinge", "modified_huber", "log_loss"],
    "alpha": [1e-6, 1e-5, 1e-4],
    "class_weight": [None, "balanced"],
    "ngram_range": [(1, 2), (1, 3)],
    "min_df": [1, 2],
    "sublinear_tf": [False, True],
}


def metrics(y_true, y_pred):
    """Recall of each class, precision on noise, and the confusion counts."""
    (rr, rf), (fr, ff) = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()
    real_recall = rr / (rr + rf) if rr + rf else 1.0
    fp_recall = ff / (fr + ff) if fr + ff else 1.0
    fp_precision = ff / (rf + ff) if rf + ff else 1.0
    f1 = 2 * fp_precision * fp_recall / (fp_precision + fp_recall) if fp_precision + fp_recall else 0.0
    return {"real_recall": round(real_recall, 4), "fp_recall": round(fp_recall, 4),
            "fp_precision": round(fp_precision, 4), "fp_f1": round(f1, 4),
            "real_rejected": rf, "noise_missed": fr, "n": rr + rf + fr + ff}


def make(cfg, seed):
    vec = TfidfVectorizer(ngram_range=cfg["ngram_range"], min_df=cfg["min_df"],
                          sublinear_tf=cfg["sublinear_tf"])
    clf = SGDClassifier(loss=cfg["loss"], alpha=cfg["alpha"], class_weight=cfg["class_weight"],
                        max_iter=50, tol=1e-4, random_state=seed)
    return vec, clf


def shift_for(scores, labels, min_real_recall):
    """Threshold that keeps real-copyright recall at min_real_recall."""
    real = sorted((s for s, y in zip(scores, labels) if y == 0), reverse=True)
    allowed = int((1 - min_real_recall) * len(real))
    return real[allowed] if allowed < len(real) else float("-inf")


def silent(y_true, y_pred, flagged):
    """Errors outside the review band."""
    return {"real_rejected": sum(1 for y, p, f in zip(y_true, y_pred, flagged) if y == 0 and p == 1 and not f),
            "noise_missed": sum(1 for y, p, f in zip(y_true, y_pred, flagged) if y == 1 and p == 0 and not f),
            "flagged_share": round(sum(flagged) / max(len(flagged), 1), 4)}


def oof_scores(cfg, Xtr, ytr, folds, seed):
    """Each training text's score from the fold model that did not see it."""
    scores = [0.0] * len(ytr)
    for tr, va in folds:
        vec, clf = make(cfg, seed)
        clf.fit(vec.fit_transform([Xtr[i] for i in tr]), [ytr[i] for i in tr])
        for i, sc in zip(va, clf.decision_function(vec.transform([Xtr[i] for i in va]))):
            scores[i] = float(sc)
    return scores


def cross_validate(cfg, Xtr, ytr, folds, seed, min_real_recall):
    """(cfg, metrics) for one configuration from its out-of-fold scores."""
    scores = oof_scores(cfg, Xtr, ytr, folds, seed)
    shift = shift_for(scores, ytr, min_real_recall)
    m = metrics(ytr, [1 if sc > shift else 0 for sc in scores])
    m["shift"] = round(shift, 6)
    return cfg, m


def predict(vec, clf, texts):
    # The same rule SafaaAgent.predict applies at its default threshold.
    X = vec.transform(texts)
    if hasattr(clf, "predict_proba"):
        return [1 if p[1] >= 0.5 else 0 for p in clf.predict_proba(X)]
    return [int(p) for p in clf.predict(X)]


def preprocess(agent, texts, version):
    """SafaaAgent's preprocessing of the given version, whatever model it loaded."""
    texts = [str(t) for t in texts]
    if version >= 2:
        return agent._preprocess_v2(texts)
    return agent._perform_text_substitutions(agent._replace_entities(texts))


def calibrate(scores, labels, max_review):
    """(slope, (low, high), margin): logistic calibration and a review band of at most max_review."""
    s = [[x] for x in scores]
    slope = float(LogisticRegression(fit_intercept=False, C=1e4).fit(s, labels).coef_[0][0])
    margin = sorted(abs(x) for x in scores)[max(0, int(max_review * len(scores)) - 1)]

    def prob(x):
        return 1.0 / (1.0 + math.exp(-slope * x))
    return slope, (round(prob(-margin), 4), round(prob(margin), 4)), margin


def preprocess_cached(agent, texts, cache_path, version):
    """Run SafaaAgent's own preprocessing once per distinct text."""
    cache = {}
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as fh:
            cache = pickle.load(fh)
    todo = sorted({t for t in texts if t not in cache})
    if todo:
        LOG.info("preprocessing %d new texts (spaCy entity pass)...", len(todo))
        for t, p in zip(todo, preprocess(agent, todo, version)):
            cache[t] = p
        with open(cache_path + ".part", "wb") as fh:
            pickle.dump(cache, fh)
        os.replace(cache_path + ".part", cache_path)
    return [cache[t] for t in texts]


def pattern_key(text, words=10):
    """First words with numbers blanked, to cap near-duplicates."""
    tokens = re.findall(r"[a-z]+|\d+", text.lower())[:words]
    return " ".join("9" if t.isdigit() else t for t in tokens)


def cap_patterns(df, cap, seed):
    """At most `cap` rows per (pattern, label), chosen reproducibly."""
    if not cap:
        return df, 0
    shuffled = df.sample(frac=1.0, random_state=seed)
    keys = shuffled["text"].map(pattern_key)
    kept = shuffled[shuffled.groupby([keys, shuffled["label"]]).cumcount() < cap]
    return kept.sort_index(), len(df) - len(kept)


def apply_corrections(legacy, path):
    """Relabel or drop legacy rows; a correction that matches nothing is an error."""
    fixes = pd.read_csv(path, keep_default_na=False)
    stale = []
    for _, f in fixes.iterrows():
        hit = (legacy["text"] == f["copyright"]) & (legacy["label"] == int(float(f["old_label"])))
        if not hit.any():
            stale.append(int(f["line"]))
        elif f["action"] == "exclude":
            legacy = legacy[~hit]
        else:
            legacy.loc[hit, "label"] = int(float(f["new_label"]))
    if stale:
        raise SystemExit("corrections no longer match the dataset at lines %s; review %s" % (stale, path))
    return legacy, fixes["action"].value_counts().to_dict()


def sha256(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--legacy", default="datasets/false_positive_detection_dataset.csv",
                    help="repository dataset; pass an empty string to train on OSSelot alone")
    ap.add_argument("--corrections", default="datasets/false_positive_detection_corrections.csv",
                    help="reviewed relabels/exclusions for --legacy; pass an empty string to use none")
    ap.add_argument("--osselot", required=True, help="dataset.csv from build_osselot_dataset.py")
    ap.add_argument("--extra", action="append", default=[],
                    help="more labelled data (text,label columns); trains only, never scored. Repeatable")
    ap.add_argument("--out", required=True, help="directory for the candidate model and report")
    ap.add_argument("--test-packages", type=float, default=0.2, help="share of OSSelot packages held out")
    ap.add_argument("--min-real-recall", type=float, default=0.995)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 2), help="parallel CV workers")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--preprocessing-version", type=int, default=PREPROCESSING_VERSION,
                    help="SafaaAgent preprocessing to train with (default: the newest)")
    ap.add_argument("--regression", default="datasets/regression_cases.csv",
                    help="hand-verified cases a candidate must not newly fail; empty string to skip")
    ap.add_argument("--max-review", type=float, default=0.05,
                    help="share of texts the review band may hold, nearest the decision boundary")
    ap.add_argument("--max-per-pattern", type=int, default=20,
                    help="cap near-duplicate texts per pattern and label, in train and test alike (0: no cap)")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    warnings.filterwarnings("ignore", category=UserWarning)
    os.makedirs(args.out, exist_ok=True)

    if args.legacy:
        legacy = pd.read_csv(args.legacy)
        legacy = pd.DataFrame({"text": legacy["copyright"].astype(str),
                               "label": legacy["falsePositive"].astype(int)})
        corrected = {}
        if args.corrections:
            legacy, corrected = apply_corrections(legacy, args.corrections)
            LOG.info("legacy corrections applied: %s", corrected)
        legacy = legacy.drop_duplicates()
    else:
        legacy = pd.DataFrame({"text": pd.Series(dtype=str), "label": pd.Series(dtype=int)})
        corrected = {}
    osl = pd.read_csv(args.osselot, keep_default_na=False)
    osl["label"] = osl["label"].astype(int)
    osl["pkgs"] = osl["packages"].str.split(";")

    # --- hold out whole packages ---
    packages = sorted({p for ps in osl["pkgs"] for p in ps})
    rng = random.Random(args.seed)
    test_pkgs = set(rng.sample(packages, max(1, round(len(packages) * args.test_packages))))
    in_test = osl["pkgs"].apply(lambda ps: any(p in test_pkgs for p in ps))
    ctx = {"legacy": int(legacy["text"].map(context_dependent).sum()),
           "osselot": int(osl["text"].map(context_dependent).sum())}
    legacy = legacy[~legacy["text"].map(context_dependent)]
    osl = osl[~osl["text"].map(context_dependent)]
    in_test = in_test[osl.index]
    LOG.info("placeholder year beside a concrete holder, left out as context-dependent: %s", ctx)
    test, osl_train = osl[in_test], osl[~in_test]
    test, capped_test = cap_patterns(test, args.max_per_pattern, args.seed)

    # --- training set: legacy + OSSelot train packages, minus anything in test ---
    test_texts = set(test["text"])
    leak = legacy["text"].isin(test_texts)
    LOG.info("legacy rows identical to a held-out text, kept out of training: %d", int(leak.sum()))
    extra = pd.concat([pd.read_csv(p, keep_default_na=False)[["text", "label"]] for p in args.extra]
                      or [pd.DataFrame({"text": pd.Series(dtype=str), "label": pd.Series(dtype=int)})])
    extra["label"] = extra["label"].astype(int)
    extra = extra[~extra["text"].isin(test_texts) & ~extra["text"].map(context_dependent)]
    if len(extra):
        LOG.info("extra training rows: %d from %s", len(extra), args.extra)
    train = pd.concat([
        legacy[~leak].assign(group="legacy"),
        extra.assign(group="legacy"),
        osl_train[["text", "label"]].assign(group=osl_train["pkgs"].str[0]),
    ], ignore_index=True)
    labels_per_text = train.groupby("text")["label"].nunique()
    conflicted = set(labels_per_text[labels_per_text > 1].index)
    if conflicted:
        LOG.info("texts labelled both ways across sources, dropped: %d", len(conflicted))
        for t in sorted(conflicted)[:10]:
            LOG.info("   conflict: %r", t[:100])
    train = train[~train["text"].isin(conflicted)].drop_duplicates("text").reset_index(drop=True)
    train, capped_train = cap_patterns(train, args.max_per_pattern, args.seed)
    train = train.reset_index(drop=True)
    LOG.info("near-duplicates capped at %d per pattern: %d dropped from train, %d from test",
             args.max_per_pattern, capped_train, capped_test)
    # Legacy rows have no package; spread them over the folds at random.
    legacy_rows = train["group"] == "legacy"
    train.loc[legacy_rows, "group"] = ["legacy-%d" % (i % args.folds) for i in range(int(legacy_rows.sum()))]
    LOG.info("train: %d texts (%d legacy, %d osselot from %d packages) | test: %d texts from %d held-out packages",
             len(train), int(legacy_rows.sum()), int((~legacy_rows).sum()), len(packages) - len(test_pkgs),
             len(test), len(test_pkgs))
    LOG.info("train labels %s | test labels %s", train["label"].value_counts().to_dict(),
             test["label"].value_counts().to_dict())

    agent = SafaaAgent()  # the shipped model: scored with its own preprocessing
    version = args.preprocessing_version
    fingerprint = preprocessing_fingerprint(agent.entity_recognizer_path, version)
    # Keyed by preprocessing fingerprint and code.
    with open(safaa_module.__file__, "rb") as fh:
        code = hashlib.sha256(fh.read()).hexdigest()[:12]
    cache = os.path.join(args.out, "preprocess_cache_%s_%s.pkl" % (fingerprint.replace(":", "_"), code))
    Xtr = preprocess_cached(agent, train["text"].tolist(), cache, version)
    Xte = preprocess_cached(agent, test["text"].tolist(), cache, version)
    ytr, yte = train["label"].tolist(), test["label"].tolist()
    groups = train["group"].tolist()

    # --- grouped cross-validation on the training side only ---
    folds = list(GroupKFold(n_splits=args.folds).split(Xtr, ytr, groups))
    keys = list(GRID)
    configs = [dict(zip(keys, values)) for values in itertools.product(*(GRID[k] for k in keys))]
    LOG.info("cross-validating %d configurations x %d folds on %d workers...", len(configs), len(folds), args.jobs)
    results = Parallel(n_jobs=args.jobs)(
        delayed(cross_validate)(cfg, Xtr, ytr, folds, args.seed, args.min_real_recall) for cfg in configs)
    cfg, cv = max(results, key=lambda r: (r[1]["fp_recall"], r[1]["real_recall"], r[1]["fp_f1"]))
    LOG.info("chosen %s", cfg)
    LOG.info("CV at its threshold %s", cv)

    # --- fit on all training data; score both models on the held-out packages ---
    vec, clf = make(cfg, args.seed)
    clf.fit(vec.fit_transform(Xtr), ytr)
    # Bake the threshold into the intercept.
    clf.intercept_ = clf.intercept_ - cv["shift"]
    new_pred = predict(vec, clf, Xte)

    # --- confidence: calibrate on out-of-fold scores, measured at the same boundary ---
    shifted = [x - cv["shift"] for x in oof_scores(cfg, Xtr, ytr, folds, args.seed)]
    slope, band, margin = calibrate(shifted, ytr, args.max_review)
    te_scores = clf.decision_function(vec.transform(Xte))
    te_err = [int(sc > 0) != y for sc, y in zip(te_scores, yte)]
    # Flagged exactly as SafaaAgent.predict_with_scores flags (needs_review).
    te_texts = test["text"].tolist()
    cut = {i: c for i, c in enumerate(map(cut_at_comment_close, te_texts)) if c != te_texts[i]}
    cut_scores = clf.decision_function(vec.transform(preprocess_cached(agent, list(cut.values()), cache, version))) \
        if cut else []
    cut_rejected = {i: sc > 0 for i, sc in zip(cut, cut_scores)}
    in_band = [needs_review(t, sc > 0, abs(sc) <= margin, cut_rejected.get(i))
               for i, (t, sc) in enumerate(zip(te_texts, te_scores))]
    review = {"slope": round(slope, 6), "band": band,
              "held_out_share_flagged": round(sum(in_band) / len(in_band), 4),
              "held_out_errors_flagged": "%d of %d" % (sum(e and b for e, b in zip(te_err, in_band)), sum(te_err))}
    LOG.info("calibration and review band: %s", review)
    old_scored = agent.predict_with_scores(test["text"].tolist())
    old_pred = [1 if s["prediction"] == "f" else 0 for s in old_scored]
    # A model without calibration flags nothing, so all its errors are silent.
    old_flags = [bool(s["needs_review"]) for s in old_scored]
    seen = set(Xtr)
    novel = [i for i, x in enumerate(Xte) if x not in seen]
    report = {
        "held_out": {"shipped": metrics(yte, old_pred), "candidate": metrics(yte, new_pred)},
        "held_out_silent": {"shipped": silent(yte, old_pred, old_flags),
                            "candidate": silent(yte, new_pred, in_band)},
        "held_out_novel": {"shipped": metrics([yte[i] for i in novel], [old_pred[i] for i in novel]),
                           "candidate": metrics([yte[i] for i in novel], [new_pred[i] for i in novel])},
    }
    for scope, r in report.items():
        for name in ("shipped", "candidate"):
            LOG.info("%-15s %-9s %s", scope, name, r[name])

    # --- regression cases: hand-verified, must not newly fail ---
    regression = None
    if args.regression:
        cases = pd.read_csv(args.regression, keep_default_na=False)
        cases = cases[~cases["text"].map(context_dependent)]
        want = cases["label"].astype(int).tolist()
        old_ok = [int(p == "f") == w for p, w in zip(agent.predict(cases["text"].tolist()), want)]
        cand = predict(vec, clf, preprocess(agent, cases["text"].tolist(), version))
        new_ok = [p == w for p, w in zip(cand, want)]
        regression = {"cases": len(want), "shipped_pass": sum(old_ok), "candidate_pass": sum(new_ok),
                      "newly_failed": [t for t, o, n in zip(cases["text"], old_ok, new_ok) if o and not n],
                      "still_failing": [t for t, o, n in zip(cases["text"], old_ok, new_ok) if not o and not n]}
        LOG.info("regression cases: shipped %d/%d, candidate %d/%d, newly failed %d", sum(old_ok), len(want),
                 sum(new_ok), len(want), len(regression["newly_failed"]))
        for t in regression["newly_failed"]:
            LOG.info("   NEWLY FAILED: %r", t[:100])

    # Held-out texts where the two models disagree: exactly one of them is right.
    differ = [{"text": t, "label": y, "shipped": o, "candidate": n}
              for t, y, o, n in zip(test["text"], yte, old_pred, new_pred) if o != n]

    dump(clf, os.path.join(args.out, "false_positive_detection_model_sgd.pkl"))
    dump(vec, os.path.join(args.out, "false_positive_detection_vectorizer.pkl"))
    out = {
        "built": datetime.now(timezone.utc).isoformat(),
        "sklearn": sklearn.__version__, "python": sys.version.split()[0], "seed": args.seed,
        "config": {k: (list(v) if isinstance(v, tuple) else v) for k, v in cfg.items()},
        "min_real_recall": args.min_real_recall,
        "cv": cv,
        "data": {"legacy_sha256": sha256(args.legacy) if args.legacy else None,
                 "corrections_sha256": sha256(args.corrections) if args.legacy and args.corrections else None,
                 "corrections_applied": corrected,
                 "osselot_sha256": sha256(args.osselot),
                 "extra": {os.path.basename(p): sha256(p) for p in args.extra},
                 "train": len(train), "test": len(test), "test_novel": len(novel),
                 "conflicts_dropped": len(conflicted), "test_packages": sorted(test_pkgs),
                 "context_dependent_dropped": ctx,
                 "max_per_pattern": args.max_per_pattern,
                 "near_duplicates_dropped": {"train": capped_train, "test": capped_test}},
        **report,
        "preprocessing_version": version,
        "review": review,
        "regression": regression,
        "fixed_by_candidate": [d for d in differ if d["candidate"] == d["label"]],
        "broken_by_candidate": [d for d in differ if d["shipped"] == d["label"]],
        "cv_all": [{"config": {k: (list(v) if isinstance(v, tuple) else v) for k, v in c.items()}, **m}
                   for c, m in results],
    }
    with open(os.path.join(args.out, "report.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, default=str)
    metadata = {
        "preprocessing_version": version,
        "preprocessing_fingerprint": fingerprint,
        "calibration": {"slope": round(slope, 6)},
        "review_band": list(band),
        "built": out["built"], "sklearn": out["sklearn"], "config": out["config"],
        "decision_shift": cv["shift"], "min_real_recall": args.min_real_recall,
        # How the held-out set was drawn, so a test can rebuild it exactly.
        "seed": args.seed, "test_packages_share": args.test_packages,
        "held_out": report["held_out"]["candidate"], "held_out_novel": report["held_out_novel"]["candidate"],
        "shipped_on_held_out": report["held_out"]["shipped"],
        "cv": {k: v for k, v in cv.items() if k != "shift"},
        "review": review,
        # Cases this model is known to get wrong; the test suite fails on any other.
        "regression": {"cases": (regression or {}).get("cases"),
                       "known_failures": (regression or {}).get("still_failing", [])
                       + (regression or {}).get("newly_failed", [])},
        "data": out["data"],
    }
    with open(os.path.join(args.out, METADATA_FILE), "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2, default=str)
    LOG.info("candidate written to %s (fixed %d, broke %d held-out texts vs shipped)", args.out,
             len(out["fixed_by_candidate"]), len(out["broken_by_candidate"]))


if __name__ == "__main__":
    main()
