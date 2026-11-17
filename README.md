<!--
SPDX-FileCopyrightText: © 2023 abdelrahmanjamal5565@gmail.com

SPDX-License-Identifier: LGPL-2.1-only
-->
# Safaa

Safaa is a Python package designed for handling false positive detection in copyright notices.
Additionally, it can declutter copyright notices, removing unnecessary extra text.

## Features

- Load pre-trained models or train your own.
- Integration with scikit-learn for training and prediction.
- Integrated with spaCy for named entity recognition and decluttering tasks.
- Preprocessing tools to ensure data consistency and quality.
- Ability to handle local or default model directories.

## Installation

To install Safaa, simply use pip:

```bash
pip install safaa
```

## Usage

### Initialization

```
from safaa.Safaa import *
agent = SafaaAgent()
```
`SafaaAgent(model_dir=...)` loads pickled models: load only model directories
you trust, as loading one can run code it contains.

### Preprocessing Data
```
data = ["Your raw data here"]
preprocessed_data = agent.preprocess_data(data)
```

### Predicting False Positives
```
predictions = agent.predict(data)          # "t" = real copyright, "f" = false positive
```

With a confidence and a review flag per item:
```
for result in agent.predict_with_scores(data):
    result["prediction"]                   # "t" or "f", as predict() returns
    result["false_positive_probability"]   # calibrated, 0.0 - 1.0
    result["needs_review"]                 # True when a reviewer should decide
```
`predict(data, threshold)` uses the same probability: raise the threshold to
reject fewer statements, lower it to reject more. The default of 0.5 is the
operating point the model was tuned for.

### Decluttering Copyright Notices
```
decluttered_data = agent.declutter(data, predictions)
```
A statement predicted real is never returned empty and never loses its
holder: when the declutter model cannot find the statement, the input is
returned with only the licence text after it removed.

### Training Models
**To retrain the false positive detector** from the repository datasets, gated
against the current model (see [scripts/data-training.md](scripts/data-training.md)):

```bash
python scripts/train_and_release.py [--csv my_labelled_data.csv]
```

**To train the false positive detector** in place with a few new examples:

```
training_data = ["Your training data here"]
labels = ["Your labels here"]   # 0 = real copyright, 1 = false positive
agent.train_false_positive_detector_model(training_data, labels)
```
This changes the model the calibration, review band and held-out results were
measured on, so they are dropped: until a gated retrain, `needs_review` flags
only context-dependent statements. Release models through the script above.

**To train the named entity recognition model:**

```
train_path = "path/to/train.spacy"
dev_path = "path/to/dev.spacy"
agent.train_ner_model(train_path, dev_path)
```
The false positive detector is trained on text this model has preprocessed:
after retraining it, retrain the detector too. A detector refuses to load
beside an entity model other than the one it was trained with.

### Saving Trained Models
```
save_path = "path/to/save"
agent.save(save_path)
```

## Dependencies
* scikit-learn
* spaCy
* joblib
* pandas

## Development

```bash
poetry install
poetry run pytest -q          # rules, the FOSSology integration contract, regression cases
poetry run flake8 --max-line-length=120 .
```

## License

This project is licensed under the [GNU LESSER GENERAL PUBLIC LICENSE, Version 2.1, February 1999](LICENSE).

## Contact Information

- **Name**: Abdelrahman Jamal
- **Email**: [abdelrahmanjamal5565@gmail.com](mailto:abdelrahmanjamal5565@gmail.com)
- **LinkedIn**: [linkedin.com/in/abdelrahmanjamal](https://linkedin.com/in/abdelrahmanjamal)
