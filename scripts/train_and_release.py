#!/usr/bin/env python3
"""
Train false positive detector using safaa.Safaa.SafaaAgent, evaluate, save model, and create a PR.

Expected CSV format: a column named 'text' and a column named 'label' (binary 0/1).

This script is intended to run inside GitHub Actions where the following env vars are available:
  - GITHUB_TOKEN (used to push and create PR)
  - GITHUB_REPOSITORY (owner/repo)
  - GITHUB_ACTOR (username)

Usage:
  python scripts/train_and_release.py --csv data.csv --model-path models/false_positive_detector.pkl

If labels are not present in the CSV the script will exit with an error — training requires labeled data.
"""

import argparse
import json
import logging
import os
import subprocess
import sys
from datetime import datetime

try:
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    from safaa.Safaa import SafaaAgent
except Exception as e:
    print("Missing dependency or safaa package not importable. Ensure requirements.txt is installed and project package path is available.", file=sys.stderr)
    raise

LOG = logging.getLogger("train_and_release")


def run_cmd(cmd, check=True, cwd=None, env=None):
    LOG.debug("Running: %s", " ".join(cmd))
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, cwd=cwd, env=env)
    if res.returncode != 0 and check:
        LOG.error("Command failed (%s): %s", res.returncode, res.stderr)
        raise RuntimeError(f"Command failed: {' '.join(cmd)}")
    return res


def create_branch_and_push(files_to_commit, branch_name, commit_message, github_token, repo):
    # Configure git user
    actor = os.environ.get("GITHUB_ACTOR", "github-actions[bot]")
    email = f"{actor}@users.noreply.github.com"
    run_cmd(["git", "config", "user.name", actor])
    run_cmd(["git", "config", "user.email", email])

    run_cmd(["git", "checkout", "-b", branch_name])
    for f in files_to_commit:
        run_cmd(["git", "add", f])
    run_cmd(["git", "commit", "-m", commit_message])

    # Push using token-authenticated remote URL
    push_url = f"https://x-access-token:{github_token}@github.com/{repo}.git"
    run_cmd(["git", "remote", "set-url", "origin", push_url])
    run_cmd(["git", "push", "-u", "origin", branch_name])


def create_pull_request(repo, branch_name, title, body, github_token, base="main"):
    import requests

    url = f"https://api.github.com/repos/{repo}/pulls"
    headers = {"Authorization": f"token {github_token}", "Accept": "application/vnd.github+json"}
    payload = {"title": title, "head": branch_name, "base": base, "body": body}
    res = requests.post(url, headers=headers, json=payload)
    if not (200 <= res.status_code < 300):
        LOG.error("Failed to create PR: %s %s", res.status_code, res.text)
        raise RuntimeError("Failed to create PR")
    return res.json()


def main():
    parser = argparse.ArgumentParser(description="Train false positive detector and create PR with new model")
    parser.add_argument("--csv", required=True, help="Path to CSV with 'text' and 'label' columns")
    parser.add_argument("--model-path", default="models/false_positive_detector.pkl", help="Where to save the trained model in the repo")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test set fraction")
    parser.add_argument("--base-branch", default="main", help="Target branch for the PR")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if not os.path.exists(args.csv):
        LOG.error("CSV file not found: %s", args.csv)
        sys.exit(2)

    df = pd.read_csv(args.csv)
    if "text" not in df.columns or "label" not in df.columns:
        LOG.error("CSV must contain 'text' and 'label' columns for training")
        sys.exit(2)

    texts = df["text"].astype(str).tolist()
    labels = df["label"].tolist()

    agent = SafaaAgent()
    LOG.info("Preprocessing %d examples", len(texts))
    preprocessed = agent.preprocess_data(texts)

    X_train, X_test, y_train, y_test = train_test_split(preprocessed, labels, test_size=args.test_size, random_state=42)

    LOG.info("Training model on %d examples", len(X_train))
    agent.train_false_positive_detector_model(X_train, y_train)

    LOG.info("Evaluating on %d test examples", len(X_test))
    y_pred = agent.predict(X_test)

    # Compute metrics (assume binary labels)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    metrics = {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}
    LOG.info("Metrics: %s", metrics)

    # Ensure model directory exists
    model_path = args.model_path
    model_dir = os.path.dirname(model_path)
    if model_dir and not os.path.exists(model_dir):
        os.makedirs(model_dir, exist_ok=True)

    LOG.info("Saving model to %s", model_path)
    agent.save(model_path)

    metrics_path = "model_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)

    # Git operations & PR
    github_token = os.environ.get("GITHUB_TOKEN")
    repo = os.environ.get("GITHUB_REPOSITORY")
    if not github_token or not repo:
        LOG.error("GITHUB_TOKEN and GITHUB_REPOSITORY environment variables are required to push and create PR")
        sys.exit(2)

    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    branch_name = f"model-update-{ts}"
    commit_message = f"chore: release false positive detector model ({ts})"
    pr_title = "Release: Updated False Positive Detector model"
    pr_body = (
        "## New False Positive Detector Model Release\n\n"
        "This PR introduces an updated version of the False Positive Detector model.\n\n"
        "**Performance Metrics on Test Data:**\n"
        f"- Accuracy: {metrics['accuracy']:.4f}\n"
        f"- Precision: {metrics['precision']:.4f}\n"
        f"- Recall: {metrics['recall']:.4f}\n"
        f"- F1 Score: {metrics['f1']:.4f}\n\n"
        f"The model was trained with data from the provided CSV: {os.path.basename(args.csv)}."
    )

    try:
        create_branch_and_push([model_path, metrics_path], branch_name, commit_message, github_token, repo)
        pr = create_pull_request(repo, branch_name, pr_title, pr_body, github_token, base=args.base_branch)
        LOG.info("Created PR: %s", pr.get("html_url"))
    except Exception as e:
        LOG.exception("Failed to push branch or create PR: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
