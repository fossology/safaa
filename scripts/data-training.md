<!--
SPDX-FileCopyrightText: © 2026 Siemens AG
SPDX-FileContributor: Kaushlendra Pratap Singh <kaushlendra-pratap.singh@siemens.com>
SPDX-License-Identifier: LGPL-2.1-only
-->

# Training Pipeline Documentation

## Overview

Safaa has three models:

| Model | Job | Retrained by |
|---|---|---|
| False positive detector (TF-IDF + linear SGD) | Is this scanner finding a real copyright statement? | `train_and_release.py` (CI or local) |
| Declutter model (spaCy NER) | Which part of the finding is the statement? | `retrain_declutter.py` (maintainer) |
| Entity recognizer (spaCy NER) | Where is the holder? Feeds the detector's preprocessing | `SafaaAgent.train_ner_model` |

The detector is **retrained from scratch** every time: the vocabulary is refitted,
so words the previous model never saw count. A candidate is released only if it is
better than the model it replaces, measured on OSSelot packages that neither model
was trained on.

## The data

| File | What it is |
|---|---|
| `datasets/false_positive_detection_dataset.csv` | The 2023 dataset (`copyright`, `falsePositive`) |
| `datasets/false_positive_detection_corrections.csv` | Reviewed relabels and exclusions for it, applied at training time; the original stays untouched. A correction that no longer matches its row stops the run. |
| `datasets/osselot_dataset.csv` | Copyright agent findings on 233 [OSSelot](https://www.osselot.org) packages, labelled against what OSSelot's reviewers kept for each file (CC0-1.0) |
| `datasets/regression_cases.csv` | Hand-verified cases no new model may newly fail |
| any other CSV you add | Your labelled data: trains, never scores |

Labels are binary: `0` = real copyright, `1` = false positive. Statements whose
year is a placeholder beside a real holder (`Copyright (C) YEAR Free Software
Foundation`) are kept out of training and scoring: reviewers keep them in some
files and reject them in others, so only the file can decide. `predict_with_scores`
always flags them for review.

### Your own CSV

```csv
text,label
"Copyright 2024 Siemens AG",0
"src/lib/c/tests/testlibs",1
```

The columns `text`/`label` and `copyright`/`falsePositive` are detected
automatically; for other names use `--text-col` and `--label-col`.

## The quality gate

On the held-out OSSelot packages, a candidate must:

1. **silently** reject no more real copyrights than the current model,
2. **silently** let through no more noise than the current model,
3. flag at most one percentage point more of the findings for review,
4. newly fail **none** of `datasets/regression_cases.csv`,
5. reach `--min-f1` on noise (CI default 0.9).

"Silently" means outside the review band. A finding SAFAA is unsure of is flagged
(`predict_with_scores(...)["needs_review"]`) and FOSSology's decider leaves it active
for a reviewer, so what can cost a company an attribution is a confident mistake.
Findings near the decision boundary flip between any two retrains; counting them
would block improvements by chance. All errors, flagged or not, are still reported.

The detector's decision point is tuned so that at most 0.5% of real copyrights are
rejected in cross-validation; rejecting a real copyright costs a reviewer a missed
notice, so that is the constraint and catching noise is what is maximised.

## Running locally

```bash
poetry install
python scripts/train_and_release.py                         # repository datasets only
python scripts/train_and_release.py --csv my_data.csv       # plus your data
```

The candidate, `report.json` (every metric, plus each held-out text the candidate
fixed or broke) and `false_positive_detection_model.json` land in `trained_model/`.
To use the candidate, copy the two `.pkl` files and the `.json` into
`Safaa/src/safaa/models/`, then run `poetry run pytest -q`.

## Running in GitHub Actions

Adding or changing a CSV under `datasets/` on `main` retrains automatically; so does
**Actions → "Train and Release Model" → Run workflow**, optionally naming an extra
CSV. A candidate that passes the gate is proposed as a pull request with its
metrics; one that fails stops the run with the reason.

Pull requests opened with the default `GITHUB_TOKEN` do not trigger other
workflows. Run the checks on the model PR by hand, or give the workflow a token
from a GitHub App.

## Refreshing the OSSelot dataset (maintainers)

`datasets/osselot_dataset.csv` is built from source, not downloaded:

1. Build FOSSology's copyright agent from the commit **before** its cleanup stage
   (`85b24694d^`), so the findings are what the scanner really emits:
   ```bash
   git -C fossology worktree add --detach ../safaa-work/fo-precleanup 85b24694d^
   cmake -S ../safaa-work/fo-precleanup -B ../safaa-work/fo-precleanup/build -GNinja
   cmake --build ../safaa-work/fo-precleanup/build --target copyright
   ```
2. With a [Thesmo](https://github.com/fossology/thesmo) checkout for the pinned
   package list and fetcher:
   ```bash
   python scripts/build_osselot_dataset.py \
     --agent ../safaa-work/fo-precleanup/build/src/copyright/agent/copyright \
     --conf  ../safaa-work/fo-precleanup/src/copyright/agent/copyright.conf \
     --thesmo ../thesmo --tier all --fetch --drop-trees --out ../safaa-work/osselot
   cp ../safaa-work/osselot/dataset.csv datasets/osselot_dataset.csv
   ```
   Files OSSelot recorded nothing for are not labelled; findings that match a
   reviewer's statement only partly, or look like a statement but match none, go to
   `review.csv` and never train.

## Retraining the declutter model (maintainers)

```bash
python scripts/retrain_declutter.py --findings ../safaa-work/osselot/findings.csv \
  --report trained_model/report.json --out ../safaa-work/declutter
```

It trains on the reviewers' statements located inside the raw findings, plus
`datasets/json/declutter_dataset.jsonl`, and scores the shipped and the candidate
model on the held-out packages: statements whose holder is lost (must be 0), holder
fully kept, junk words, and exact agreement with the reviewer. Adopt the candidate's
`models/declutter_model` only if it is better on all four.
