# Safaa

Safaa is a Python package designed for handling false positive detection in copyright notices. Additionally, it can declutter copytright notices, removing unnecessary extra text.

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
