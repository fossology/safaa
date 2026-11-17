#!/usr/bin/env python3

# SPDX-FileCopyrightText: © 2026 Siemens AG
# SPDX-FileContributor: Kaushlendra Pratap Singh <kaushlendra-pratap.singh@siemens.com>
# SPDX-License-Identifier: LGPL-2.1-only
"""
Retrain the false-positive detector from scratch and release it only if it passes the gate.

Gate, on held-out OSSelot packages, against the current model ("silent" = outside the review band):
  - no more real copyrights rejected silently, no more noise accepted silently,
  - at most one percentage point more flagged for review,
  - no newly failing datasets/regression_cases.csv, noise F1 >= --min-f1.

Locally the candidate is left in --output-dir; in CI (GITHUB_ACTIONS=true) a passing
candidate replaces the shipped model and a PR is opened.

--csv (optional, repeatable):
  - A text column  (--text-col;  auto-detected from "text" or "copyright")
  - A label column (--label-col; auto-detected from "label" or "falsePositive")
    with binary 0/1 values (0 = real copyright, 1 = false positive)
"""

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone

import pandas as pd
import requests

LOG = logging.getLogger("train_and_release")

MODEL_REPO_PATH = "Safaa/src/safaa/models"

# Tried in order when --text-col / --label-col are not given.
TEXT_COL_CANDIDATES = ("text", "copyright")
LABEL_COL_CANDIDATES = ("label", "falsePositive")


def resolve_column(columns, requested, candidates, kind):
    """Return the column to use, or exit with an error if none is found."""
    if requested:
        if requested in columns:
            return requested
        LOG.error("%s column '%s' not found. Available: %s", kind, requested, list(columns))
        sys.exit(2)
    for name in candidates:
        if name in columns:
            return name
    LOG.error(
        "No %s column found (tried %s). Available: %s. Use --%s-col to set it.",
        kind, list(candidates), list(columns), kind,
    )
    sys.exit(2)


def run_cmd(cmd, check=True):
    LOG.debug("Running: %s", " ".join(cmd))
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if res.returncode != 0 and check:
        LOG.error("Command failed (%s): %s", res.returncode, res.stderr)
        raise RuntimeError(f"Command failed: {' '.join(cmd)}")
    return res


def is_ci():
    return os.environ.get("GITHUB_ACTIONS") == "true"


def create_branch_and_push(files_to_commit, branch_name, commit_message):
    actor = os.environ.get("GITHUB_ACTOR", "github-actions[bot]")
    email = f"{actor}@users.noreply.github.com"
    run_cmd(["git", "config", "user.name", actor])
    run_cmd(["git", "config", "user.email", email])

    run_cmd(["git", "checkout", "-b", branch_name])
    for f in files_to_commit:
        run_cmd(["git", "add", f])
    run_cmd(["git", "commit", "-m", commit_message])

    # Uses the credentials actions/checkout stored; no token in the URL.
    run_cmd(["git", "push", "-u", "origin", branch_name])


def create_pull_request(branch_name, title, body, base="main"):
    github_token = os.environ["GITHUB_TOKEN"]
    repo = os.environ["GITHUB_REPOSITORY"]
    url = f"https://api.github.com/repos/{repo}/pulls"
    headers = {
        "Authorization": f"Bearer {github_token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    payload = {"title": title, "head": branch_name, "base": base, "body": body}
    res = requests.post(url, headers=headers, json=payload, timeout=60)
    if not (200 <= res.status_code < 300):
        LOG.error("Failed to create PR: %s %s", res.status_code, res.text)
        raise RuntimeError("Failed to create PR")
    return res.json()


MODEL_FILES = ("false_positive_detection_model_sgd.pkl", "false_positive_detection_vectorizer.pkl",
               "false_positive_detection_model.json")


#: Extra share of held-out findings a candidate may flag for review.
MAX_EXTRA_REVIEW_SHARE = 0.01


def gate(report, min_f1):
    """(passed, reasons). Judged on silent errors; on all errors if the report lacks them."""
    quiet = report.get("held_out_silent")
    cand, cur = (quiet["candidate"], quiet["shipped"]) if quiet else (
        report["held_out"]["candidate"], report["held_out"]["shipped"])
    kind = "silently " if quiet else ""
    reasons = []
    if cand["real_rejected"] > cur["real_rejected"]:
        reasons.append("%srejects more real copyrights (%d > %d)" % (kind, cand["real_rejected"], cur["real_rejected"]))
    if cand["noise_missed"] > cur["noise_missed"]:
        reasons.append("%slets through more noise (%d > %d)" % (kind, cand["noise_missed"], cur["noise_missed"]))
    if quiet and cand["flagged_share"] > cur["flagged_share"] + MAX_EXTRA_REVIEW_SHARE:
        reasons.append("flags more for review (%.1f%% > %.1f%% + %.0f points)"
                       % (100 * cand["flagged_share"], 100 * cur["flagged_share"], 100 * MAX_EXTRA_REVIEW_SHARE))
    newly = (report.get("regression") or {}).get("newly_failed", [])
    if newly:
        reasons.append("newly fails %d regression case(s): %s" % (len(newly), newly[:3]))
    if report["held_out"]["candidate"]["fp_f1"] < min_f1:
        reasons.append("noise F1 %.4f below %.4f" % (report["held_out"]["candidate"]["fp_f1"], min_f1))
    return not reasons, reasons


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", action="append", default=[], help="extra labelled CSV to train on (repeatable)")
    parser.add_argument("--osselot", default="datasets/osselot_dataset.csv", help="OSSelot dataset")
    parser.add_argument("--output-dir", default="trained_model", help="where the candidate and report go")
    parser.add_argument("--text-col", default=None,
                        help="CSV column with copyright text (default: auto-detect 'text' or 'copyright')")
    parser.add_argument("--label-col", default=None,
                        help="CSV column with binary labels 0/1 (default: auto-detect 'label' or 'falsePositive')")
    parser.add_argument("--min-f1", type=float, default=0.0, help="floor on the candidate's noise F1")
    parser.add_argument("--base-branch", default="main", help="Target branch for PR in CI mode (default: main)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    os.makedirs(args.output_dir, exist_ok=True)
    extras = []
    for i, path in enumerate(args.csv):
        if not os.path.exists(path):
            LOG.error("CSV file not found: %s", path)
            sys.exit(2)
        df = pd.read_csv(path, keep_default_na=False)
        text_col = resolve_column(df.columns, args.text_col, TEXT_COL_CANDIDATES, "text")
        label_col = resolve_column(df.columns, args.label_col, LABEL_COL_CANDIDATES, "label")
        LOG.info("%s: using columns text='%s', label='%s'", path, text_col, label_col)
        if not set(df[label_col].astype(str)) <= {"0", "1"}:
            LOG.error("%s: label column '%s' must hold only 0 and 1", path, label_col)
            sys.exit(2)
        extra = os.path.join(args.output_dir, "extra_%d.csv" % i)
        pd.DataFrame({"text": df[text_col].astype(str), "label": df[label_col].astype(int)}).to_csv(extra, index=False)
        extras.append(extra)

    trainer = os.path.join(os.path.dirname(os.path.abspath(__file__)), "retrain_false_positive_detector.py")
    cmd = [sys.executable, trainer, "--osselot", args.osselot, "--out", args.output_dir]
    for extra in extras:
        cmd += ["--extra", extra]
    LOG.info("Retraining from scratch: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)

    with open(os.path.join(args.output_dir, "report.json"), encoding="utf-8") as fh:
        report = json.load(fh)
    passed, reasons = gate(report, args.min_f1)
    cand, cur = report["held_out"]["candidate"], report["held_out"]["shipped"]
    quiet = report.get("held_out_silent", {})

    def row(name, current, candidate):
        return "| %s | %s | %s |\n" % (name, current, candidate)

    table = "| On held-out OSSelot packages | Current model | Candidate |\n|---|---|---|\n"
    for name, key in (("Real copyrights rejected silently", "real_rejected"),
                      ("Noise let through silently", "noise_missed"), ("Share flagged for review", "flagged_share")):
        table += row(name, quiet.get("shipped", {}).get(key, "-"), quiet.get("candidate", {}).get(key, "-"))
    for name, key in (("Real copyrights rejected (all)", "real_rejected"), ("Noise let through (all)", "noise_missed"),
                      ("Real-copyright recall", "real_recall"), ("Noise recall", "fp_recall"),
                      ("Noise precision", "fp_precision"), ("Noise F1", "fp_f1"), ("Texts", "n")):
        table += row(name, cur[key], cand[key])
    LOG.info("\n%s", table)
    if not passed:
        LOG.error("Quality gate FAILED, model not released: %s", "; ".join(reasons))
        sys.exit(1)
    LOG.info("Quality gate passed.")

    if not is_ci():
        LOG.info("Running locally, no PR created. Candidate in %s/; to use it copy %s into %s/",
                 args.output_dir, ", ".join(MODEL_FILES), MODEL_REPO_PATH)
        return

    for name in MODEL_FILES:
        shutil.copy2(os.path.join(args.output_dir, name), os.path.join(MODEL_REPO_PATH, name))
    if not os.environ.get("GITHUB_TOKEN") or not os.environ.get("GITHUB_REPOSITORY"):
        LOG.error("GITHUB_TOKEN and GITHUB_REPOSITORY are required in CI mode.")
        sys.exit(2)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    branch_name = f"model-update-{ts}"
    create_branch_and_push([os.path.join(MODEL_REPO_PATH, n) for n in MODEL_FILES], branch_name,
                           f"chore(model): retrain false positive detector ({ts})")
    sources = [os.path.basename(args.osselot)] + [os.path.basename(c) for c in args.csv]
    body = ("## Retrained false positive detector\n\nAuto-generated by the training pipeline: retrained from "
            "scratch and gated against the current model on OSSelot packages neither was trained on.\n\n"
            + table + "\nTraining data: %s\n" % ", ".join(sources))
    pr = create_pull_request(branch_name, "Release: retrained false positive detector", body, base=args.base_branch)
    LOG.info("PR created: %s", pr.get("html_url"))


if __name__ == "__main__":
    main()
