## Safaa Model Retraining Pipeline

This folder provides an automated pipeline for retraining the Safaa false positive detection model. It includes data fetching from the fossology server instance (currently on localhost), preprocessing, model training, evaluation, and automated pull request creation with the updated model.

### Overview

The Safaa Model Retraining Pipeline is designed to:
- Fetch copyright data from a fossology server instance.
- Preprocess and clean copyright data.
- Train a false positive detection model using Safaa.
- Evaluate model performance on test data.
- Automatically create pull requests with retrained models and performance metrics.

### Features

- **Fossology Server Integration**: Fetch copyright data directly from fossology localhost instance.
- **Automated Workflow**: GitHub Actions workflow for model retraining when triggered.
- **Data Pipeline**: Complete data preprocessing including decluttering and splitting
- **Model Training**: Train false positive detection models using the SafaaAgent training script
- **Performance Metrics**: Automatic calculation of accuracy, precision, recall, and F1 score
- **Version Control**: Automatic PR creation with model performance in the title

### Project Structure

```
├── .github/
│   └── workflows/
│       └── pipeline.yml              # GitHub Actions workflow
├── utility/
│   └── retraining/
│       ├── script_for_copyrights.py  # Fossology server copyright fetch script
│       ├── utility_scripts.py        # Main retraining scripts
│       ├── data/                     # Data directory 
│       │   └── copyrights_*.csv      # Fetched copyright data (uses copyrights_timestamp format)
│       └── model/                    # Trained model output
```

### How to fetch copyrights from fossology local instance

- From fetching copyrights from fossology local instance. Create a `.env` file in your project root with the following database credentials:

    ```env
    DB_NAME=your_database_name
    DB_USER=your_database_user
    DB_PASSWORD=your_database_password
    DB_HOST=your_database_host
    DB_PORT=your_database_port
    ```
- Start the fossology instance locally.
- Run the `script_for_copyrights.py` script.

### Pipeline Usage (GitHub Actions) 

The workflow can be manually triggered from the GitHub Actions tab in your repository. It will:
1. Perform Pre-processing till training model steps.
2. Perform evaluation.
3. Create a PR with the updated model.

### Model Output

The trained model produces two files:
- `false_positive_detection_vectorizer.pkl`: Text vectorizer for feature extraction
- `false_positive_detection_model_sgd.pkl`: Trained SGD classifier

### Performance Metrics

The pipeline automatically calculates and reports:
- Accuracy
- Precision
- Recall
- F1 Score

## Contact Information

- **Name**: Abdulsobur Oyewale
- **Email**: [Abdulsoburoyewale@gmail.com.com](mailto:Abdulsoburoyewale@gmail.com)