# SPDX-License-Identifier: CC-BY-SA-4.0
# SPDX-FileCopyrightText: 2025 Abdulsobur Oyewale

import pandas as pd
from sklearn.model_selection import train_test_split
from safaa.Safaa import SafaaAgent
import os
import glob


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


def main():
    base_path = os.path.dirname(__file__)
    data_dir = os.path.join(base_path, "data")

    agent = SafaaAgent()

    # Preprocessing
    latest_file = find_latest_copyright_file(data_dir)
    raw_df = load_data(latest_file)
    raw_data = raw_df['original_content']
    preprocessed_df = preprocess_data(agent, raw_data)
    prep_data_path = os.path.join(data_dir, "preprocessed_copyrights.csv")
    save_to_csv(preprocessed_df, prep_data_path)

    print("✅ Preprocessing completed")

    # Decluttering
    df = load_data(os.path.join(data_dir, "preprocessed_copyrights.csv"))
    data = df['original_content']
    decluttered_df = declutter_data(agent, data)
    decl_data_path = os.path.join(data_dir, "decluttered_copyrights.csv")
    save_to_csv(decluttered_df, decl_data_path)

    print("✅ Decluttering completed")

    # Data Split
    train_df, test_df = split_data(df)
    train_data_path = os.path.join(data_dir, "train_data.csv")
    test_data_path = os.path.join(data_dir, "test_data.csv")
    save_to_csv(train_df, train_data_path)
    save_to_csv(test_df, test_data_path)

    print("✅ Split complete: train_data.csv and test_data.csv created.")


if __name__ == '__main__':
    main()
