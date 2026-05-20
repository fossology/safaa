#!/usr/bin/env python3

# SPDX-FileCopyrightText: © 2026 Siemens AG
# SPDX-FileContributor: Kaushlendra Pratap Singh <kaushlendra-pratap.singh@siemens.com>
# SPDX-License-Identifier: LGPL-2.1-only
"""
Train the Safaa false-positive detector (SGD) iteratively from a labeled CSV.

This script works in two modes:
  1. LOCAL   User provides --csv; model is saved to --output-dir. No git/PR.
  2. CI      Detected via GITHUB_ACTIONS=true env var. After training, if quality
              gate passes, the updated model binary replaces the one in
              Safaa/src/safaa/models/ and a PR is raised.

CSV requirements:
  - A text column  (--text-col,  default: "text")
  - A label column (--label-col, default: "label") with binary 0/1 values
    (0 = real copyright, 1 = false positive)
"""

import argparse
import json
import logging
import os
import subprocess
import sys
from datetime import datetime, timezone

import requests

try:
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    from safaa.Safaa import SafaaAgent
except Exception as e:
    print("Missing dependency. Run: poetry install", file=sys.stderr)
    raise

LOG = logging.getLogger("train_and_release")

MODEL_REPO_PATH = "Safaa/src/safaa/models"


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

    github_token = os.environ["GITHUB_TOKEN"]
    repo = os.environ["GITHUB_REPOSITORY"]
    push_url = f"https://x-access-token:{github_token}@github.com/{repo}.git"
    run_cmd(["git", "remote", "set-url", "origin", push_url])
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
    res = requests.post(url, headers=headers, json=payload)
    if not (200 <= res.status_code < 300):
        LOG.error("Failed to create PR: %s %s", res.status_code, res.text)
        raise RuntimeError("Failed to create PR")
    return res.json()


def main():
    parser = argparse.ArgumentParser(
        description="Train Safaa false-positive detector (SGD, iterative) from a labeled CSV"
    )
    parser.add_argument("--csv", required=True, help="Path to the labeled training CSV")
    parser.add_argument(
        "--output-dir", default="trained_model",
        help="Directory to write model artifacts when running locally (default: trained_model/)",
    )
    parser.add_argument("--text-col", default="text", help="CSV column with copyright text (default: text)")
    parser.add_argument("--label-col", default="label", help="CSV column with binary labels 0/1 (default: label)")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test set fraction (default: 0.2)")
    parser.add_argument(
        "--min-f1", type=float, default=0.0,
        help="Quality gate: minimum F1 to save/promote model (default: 0.0  disabled)",
    )
    parser.add_argument("--base-branch", default="main", help="Target branch for PR in CI mode (default: main)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    # --- Load and validate CSV ---
    if not os.path.exists(args.csv):
        LOG.error("CSV file not found: %s", args.csv)
        sys.exit(2)

    df = pd.read_csv(args.csv)
    for col in (args.text_col, args.label_col):
        if col not in df.columns:
            LOG.error("Column '%s' not found. Available: %s", col, list(df.columns))
            sys.exit(2)

    texts = df[args.text_col].astype(str).tolist()
    labels = df[args.label_col].tolist()

    if len(set(labels)) < 2:
        LOG.error("Label column must contain at least two distinct classes.")
        sys.exit(2)

    agent = SafaaAgent()

    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=args.test_size, random_state=42, stratify=labels
    )
    LOG.info("Train: %d  |  Test: %d", len(X_train), len(X_test))

    agent.train_false_positive_detector_model(X_train, y_train)
    LOG.info("Training complete (iterative SGD partial_fit).")

    # --- Evaluate ---
    y_pred_raw = agent.predict(X_test)
    label_map = {"f": 1, "t": 0}
    y_pred = [label_map[p] for p in y_pred_raw]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "train_size": len(X_train),
        "test_size": len(X_test),
        "csv_file": os.path.basename(args.csv),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    LOG.info(
        "Metrics  Accuracy: %.4f | Precision: %.4f | Recall: %.4f | F1: %.4f",
        metrics["accuracy"], metrics["precision"], metrics["recall"], metrics["f1"],
    )

    if metrics["f1"] < args.min_f1:
        LOG.error(
            "Quality gate FAILED: F1 %.4f < required %.4f. Model NOT saved.",
            metrics["f1"], args.min_f1,
        )
        sys.exit(1)

    if is_ci():
        # In CI: overwrite the model binary in the repo source tree
        output_dir = MODEL_REPO_PATH
    else:
        output_dir = args.output_dir

    os.makedirs(output_dir, exist_ok=True)
    agent.save(output_dir)

    metrics_path = os.path.join(output_dir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)

    LOG.info("Model artifacts saved to %s/", output_dir)

    if is_ci():
        github_token = os.environ.get("GITHUB_TOKEN")
        repo = os.environ.get("GITHUB_REPOSITORY")
        if not github_token or not repo:
            LOG.error("GITHUB_TOKEN and GITHUB_REPOSITORY are required in CI mode.")
            sys.exit(2)

        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        branch_name = f"model-update-{ts}"
        commit_message = f"chore: update false positive detector model ({ts})"

        model_pkl = os.path.join(MODEL_REPO_PATH, "false_positive_detection_model_sgd.pkl")
        vectorizer_pkl = os.path.join(MODEL_REPO_PATH, "false_positive_detection_vectorizer.pkl")

        pr_title = "Release: Updated False Positive Detector Model"
        pr_body = (
            "## Updated False Positive Detector Model\n\n"
            "This PR was auto-generated by the training pipeline.\n"
            "The SGD model was iteratively trained with new data.\n\n"
            "**Performance Metrics (test set):**\n\n"
            "| Metric | Value |\n|--------|-------|\n"
            f"| Accuracy | {metrics['accuracy']:.4f} |\n"
            f"| Precision | {metrics['precision']:.4f} |\n"
            f"| Recall | {metrics['recall']:.4f} |\n"
            f"| F1 Score | {metrics['f1']:.4f} |\n"
            f"| Train Size | {metrics['train_size']} |\n"
            f"| Test Size | {metrics['test_size']} |\n\n"
            f"**Source CSV:** `{metrics['csv_file']}`\n"
        )

        create_branch_and_push(
            [model_pkl, vectorizer_pkl],
            branch_name, commit_message,
        )
        pr = create_pull_request(branch_name, pr_title, pr_body, base=args.base_branch)
        LOG.info("PR created: %s", pr.get("html_url"))
    else:
        LOG.info("Running locally, no PR created. Use the artifacts in %s/", output_dir)


if __name__ == "__main__":
    main()
