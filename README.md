<!--
SPDX-FileCopyrightText: © 2023 abdelrahmanjamal5565@gmail.com

SPDX-License-Identifier: LGPL-2.1-only
-->
# Safaa

[![Python Versions](https://img.shields.io/pypi/pyversions/safaa?logo=python)](https://pypi.org/project/safaa/)
[![PyPI Version](https://img.shields.io/pypi/v/safaa?logo=pypi)](https://pypi.org/project/safaa/)
[![License: LGPL v2.1](https://img.shields.io/badge/License-LGPL%20v2.1-blue.svg)](https://www.gnu.org/licenses/old-licenses/lgpl-2.1.html)
[![Code Quality](https://github.com/fossology/safaa/actions/workflows/code-quality.yml/badge.svg)](https://github.com/fossology/safaa/actions/workflows/code-quality.yml)
[![Build Tests](https://github.com/fossology/safaa/actions/workflows/build-test.yml/badge.svg)](https://github.com/fossology/safaa/actions/workflows/build-test.yml)
[![Slack Channel](https://img.shields.io/badge/slack-fossology-blue.svg?longCache=true&logo=slack)](https://join.slack.com/t/fossology/shared_invite/enQtNzI0OTEzMTk0MjYzLTYyZWQxNDc0N2JiZGU2YmI3YmI1NjE4NDVjOGYxMTVjNGY3Y2MzZmM1OGZmMWI5NTRjMzJlNjExZGU2N2I5NGY)

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

### Preprocessing Data
```
data = ["Your raw data here"]
preprocessed_data = agent.preprocess_data(data)
```

### Predicting False Positives
```
predictions = agent.predict(data)
```

### Decluttering Copyright Notices
```
decluttered_data = agent.declutter(data, predictions)
```

### Training Models
**To train the false positive detector:**

```
training_data = ["Your training data here"]
labels = ["Your labels here"]
agent.train_false_positive_detector_model(training_data, labels)
```

**To train the named entity recognition model:**

```
train_path = "path/to/train.spacy"
dev_path = "path/to/dev.spacy"
agent.train_ner_model(train_path, dev_path)
```

### Saving Trained Models
```
save_path = "path/to/save"
agent.save(save_path)
```

## Dependencies
* scikit-learn
* spaCy
* joblib
* regex
* os
* shutil

## License

This project is licensed under the [GNU LESSER GENERAL PUBLIC LICENSE, Version 2.1, February 1999](LICENSE).

## Contact Information

- **Name**: Abdelrahman Jamal
- **Email**: [abdelrahmanjamal5565@gmail.com](mailto:abdelrahmanjamal5565@gmail.com)
- **LinkedIn**: [linkedin.com/in/abdelrahmanjamal](https://linkedin.com/in/abdelrahmanjamal)
