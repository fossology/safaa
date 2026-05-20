<!--
SPDX-FileCopyrightText: © 2026 Siemens AG
SPDX-FileContributor: Kaushlendra Pratap Singh <kaushlendra-pratap.singh@siemens.com>
SPDX-License-Identifier: LGPL-2.1-only
-->

# Training Pipeline Documentation

## Overview

Safaa uses an **iterative SGD (Stochastic Gradient Descent)** model for false-positive
copyright detection. Each training run fine-tunes the existing model weights with new data
using `partial_fit`  it does NOT train from scratch. This means you can continuously
improve the model by feeding it new labeled data over time.

## How It Works

```
┌─────────────────────────────┐
│  You provide a labeled CSV  │
│  (text + binary 0/1 label)  │
└──────────────┬──────────────┘
               │
       ┌───────▼────────┐
       │  train_and_     │
       │  release.py     │
       └───────┬─────────┘
               │
    ┌──────────▼──────────────────────────┐
    │  Loads existing SafaaAgent model     │
    │  ↓                                   │
    │  Stratified 80/20 train-test split   │
    │  ↓                                   │
    │  partial_fit (iterative training)    │
    │  ↓                                   │
    │  Evaluate: accuracy, precision,      │
    │            recall, F1                 │
    │  ↓                                   │
    │  Quality gate (--min-f1)             │
    │  ↓                                   │
    │  Save model artifacts                │
    └─────────────────────────────────────┘
```

## CSV Format

Your CSV must have:
- A **text column** containing the raw copyright statement string
- A **label column** containing binary integer labels:
  - `0` = real copyright (true positive)
  - `1` = false positive (noise)

Example:

```csv
text,label
"Copyright 2024 Siemens AG",0
"src/lib/c/tests/testlibs",1
```

> **Note:** If your CSV uses different column names (e.g. `copyright` / `falsePositive`),
> use `--text-col` and `--label-col` flags to specify them.

---

## Running Locally

### Prerequisites

```bash
# Clone the repo and install
git clone https://github.com/fossology/safaa.git
cd safaa
poetry install
```

### Train with default column names (`text` / `label`)

```bash
python scripts/train_and_release.py --csv path/to/your_data.csv
```

### Train with custom column names

```bash
python scripts/train_and_release.py \
    --csv datasets/false_positive_detection_dataset.csv \
    --text-col copyright \
    --label-col falsePositive
```

### All available options

| Flag | Default | Description |
|------|---------|-------------|
| `--csv` | *(required)* | Path to the labeled CSV |
| `--text-col` | `text` | Column name for the copyright string |
| `--label-col` | `label` | Column name for the binary label |
| `--output-dir` | `trained_model/` | Where to save model artifacts (local mode only) |
| `--test-size` | `0.2` | Fraction held out for evaluation |
| `--min-f1` | `0.0` | Quality gate  exits with error if F1 is below this |
| `--base-branch` | `main` | PR target branch (CI mode only) |

### Output

After training, you'll find these files in `--output-dir`:

```
trained_model/
├── false_positive_detection_model_sgd.pkl   # The SGD classifier
├── false_positive_detection_vectorizer.pkl  # The TF-IDF vectorizer
└── metrics.json                             # Evaluation results
```

To use the newly trained model with Safaa:

```bash
# Copy the .pkl files into the package's model directory
cp trained_model/*.pkl Safaa/src/safaa/models/
```

---

## Running via GitHub Actions (CI)

The pipeline runs automatically OR manually  no database access needed.

### Automatic Trigger (recommended)

Simply **add a new `.csv` file to the `datasets/` directory** on the `main` branch:

```bash
cp my_new_training_data.csv datasets/
git add datasets/my_new_training_data.csv
git commit -m "Add new training data"
git push origin main
```

The workflow detects the new CSV and automatically:
1. Installs dependencies via Poetry
2. Trains the model iteratively on the new data
3. Evaluates and checks the quality gate (F1 ≥ 0.7)
4. If passed → creates a **Pull Request** replacing the model binary in
   `Safaa/src/safaa/models/false_positive_detection_model_sgd.pkl`
5. The PR includes a metrics table so reviewers can assess model quality

### Manual Trigger (workflow_dispatch)

Go to **Actions → "Train and Release Model" → Run workflow** and fill in:

| Input | Description |
|-------|-------------|
| `csv_path` | Path to CSV in the repo (e.g. `datasets/my_data.csv`) |
| `min_f1` | Quality gate threshold (default: `0.7`) |
| `text_col` | Text column name (default: `text`) |
| `label_col` | Label column name (default: `label`) |

### What the PR looks like

The auto-generated PR includes:
- Updated `.pkl` model binary
- Updated `metrics.json` with full evaluation results
- A markdown table showing accuracy, precision, recall, F1, and dataset sizes

---

## Iterative Training Strategy

Because the model uses **SGD with `partial_fit`**:

- Each training run **builds on** the previous model state
- You do NOT need to retrain from scratch with all historical data
- New data incrementally improves the model
- You can train multiple times with different CSV batches

### Recommended workflow

1. Export labeled copyright data from your Fossology instance
2. Format as CSV with `text` and `label` columns
3. Add to `datasets/` and push → pipeline trains automatically
4. Review the PR metrics and merge if satisfied
5. Repeat as new data becomes available

---

## Quality Gate

The `--min-f1` flag prevents bad models from being promoted:

- **Local:** Script exits with code `1` and does NOT save model artifacts
- **CI:** Script exits with code `1`, no PR is created, workflow fails visibly

Set this based on your acceptable model quality. A value of `0.7` is a reasonable
starting point for copyright false-positive detection.

---

## Testing on GitHub Actions

Follow these steps to verify the full CI pipeline works end-to-end:

### Option A: Manual trigger (quickest way to test)

1. **Push your branch** (with the updated workflow + script) to the remote:
   ```bash
   git push origin your-branch-name
   ```

2. **Go to Actions tab** in the GitHub repo.

3. **Click "Run workflow"** → select your branch → fill in:
   - `csv_path`: `datasets/false_positive_detection_dataset.csv`
   - `text_col`: `copyright`
   - `label_col`: `falsePositive`
   - `min_f1`: `0.5` (use a low value for testing so the gate passes)

4. Click **"Run workflow"** and watch the logs.

5. If successful, a PR will be created against `main` with the updated model binary.

### Option B: Simulate the automatic trigger

1. Merge your workflow changes into `main` first (or test on a fork).

2. Add a CSV to `datasets/` and push directly to `main`:
   ```bash
   cp your_training_data.csv datasets/new_training_data.csv
   git add datasets/new_training_data.csv
   git commit -m "test: add training data to trigger pipeline"
   git push origin main
   ```

3. The workflow triggers automatically. Check the Actions tab for progress.

> **Note:** The automatic trigger only fires on `main`. If you're testing on a
> feature branch, use Option A (manual trigger).

### What to verify in the logs

- **Install dependencies** → Poetry installs safaa + dev deps successfully
- **Detect new CSV** (push trigger only) → Correctly identifies the new CSV file
- **Train model** → Shows train/test split sizes, metrics output
- **Quality gate** → Passes if F1 ≥ threshold
- **PR created** → Link to the auto-generated PR appears in the log

### Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| Workflow doesn't trigger on push | CSV not in `datasets/` or not on `main` | Use `workflow_dispatch` or push to `main` |
| `Column 'text' not found` | CSV uses different column names | Set `text_col` / `label_col` inputs |
| Quality gate fails | F1 below threshold | Lower `--min-f1` or provide better training data |
| PR creation fails | Missing permissions | Ensure workflow has `contents: write` and `pull-requests: write` |
| `No new CSV detected. Skipping.` | File was not added/modified in the push diff | Ensure the CSV is a new or changed file in the commit range |

---

## Fetching Data from Fossology (Optional)

If you have direct access to a Fossology PostgreSQL database, you can use the
`scripts/fetch_copyrights.py` helper to export copyright data to CSV.
See `scripts/README.md` for usage details. This is **not required** for the
training pipeline  any labeled CSV will work regardless of how it was produced.
