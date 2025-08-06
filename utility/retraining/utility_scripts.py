# SPDX-License-Identifier: CC-BY-SA-4.0
# SPDX-FileCopyrightText: 2025 Abdulsobur Oyewale

import pandas as pd
from joblib import load, dump
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score
from safaa.Safaa import SafaaAgent
import os
import glob
import argparse


def load_data(filepath: str) -> pd.DataFrame:
    return pd.read_csv(filepath)


def declutter_data(agent: SafaaAgent, data: pd.Series) -> pd.DataFrame:
    predictions = ["f"] * len(data)
    decluttered = agent.declutter(data, predictions)
    return pd.DataFrame({'original_content': data, 'decluttered_content': decluttered})


def preprocess_data(agent: SafaaAgent, data: pd.Series) -> pd.DataFrame:
    preprocessed = agent.preprocess_data(data)
    return pd.DataFrame({'original_content': data, 'preprocessed_content': preprocessed})


def split_data(df: pd.DataFrame, test_size=0.2, random_state=42):
    return train_test_split(df, test_size=test_size, random_state=random_state)


def save_to_csv(df: pd.DataFrame, filepath: str):
    df.to_csv(filepath, index=False)


def find_latest_copyright_file(data_dir: str):
    files = glob.glob(os.path.join(data_dir, "copyrights_*.csv"))
    if not files:
        raise FileNotFoundError("No copyright data files found in 'data/'")
    latest = max(files, key=os.path.getmtime)
    return latest



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--preprocess', action='store_true')
    parser.add_argument('--declutter', action='store_true')
    parser.add_argument('--split', action='store_true')
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--test', action='store_true')
    args = parser.parse_args()

    base_path = os.path.dirname(__file__)
    data_dir = os.path.join(base_path, "data")
    agent = SafaaAgent()

    if args.preprocess:
        latest_file = find_latest_copyright_file(data_dir)
        raw_df = load_data(latest_file)
        raw_data = raw_df['original_content']
        preprocessed_df = preprocess_data(agent, raw_data)
        save_to_csv(preprocessed_df, os.path.join(data_dir, "preprocessed_copyrights.csv"))
        print("✅ Preprocessing completed")

    if args.declutter:
        df = load_data(os.path.join(data_dir, "preprocessed_copyrights.csv"))
        data = df['original_content']
        decluttered_df = declutter_data(agent, data)
        save_to_csv(decluttered_df, os.path.join(data_dir, "decluttered_copyrights.csv"))
        print("✅ Decluttering completed")

    if args.split:
        # For the pipeline preprocessed data
        # df = load_data(os.path.join(data_dir, "preprocessed_copyrights.csv"))

        # For the available data in fossology
        train_data_path = os.path.join(base_path, '..', '..', 'datasets', 'false_positive_detection_dataset.csv')
        dataset_path = os.path.abspath(train_data_path)
        df = pd.read_csv(dataset_path)


        train_df, test_df = split_data(df)
        save_to_csv(train_df, os.path.join(data_dir, "train_data.csv"))
        save_to_csv(test_df, os.path.join(data_dir, "test_data.csv"))
        print("✅ Split complete: train_data.csv and test_data.csv created.")

    if args.train:
        data = load_data(os.path.join(data_dir, "train_data.csv"))
        agent.train_false_positive_detector_model(data["copyright"], data["falsePositive"])
        model_dir = os.path.join(base_path, 'model')
        # agent.save("/home/fossy/Safaa")
        agent.save(model_dir)
        print("✅ Training completed and model saved.")


    if args.test:
        test_data = load_data(os.path.join(data_dir, "test_data.csv"))
        X_test = test_data["copyright"]

        y_true = test_data["falsePositive"].map(lambda x: "f" if x in (1, True) else "t")

        agent = SafaaAgent(use_local_model=False, model_dir=os.path.join(base_path, 'model'))

        y_pred = agent.predict(X_test)

        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, pos_label="f")
        recall = recall_score(y_true, y_pred, pos_label="f")
        f1 = f1_score(y_true, y_pred, pos_label="f")

        print("✅ Evaluation on test set:")
        print(f"  Accuracy : {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall   : {recall:.4f}")
        print(f"  F1 Score : {f1:.4f}")
