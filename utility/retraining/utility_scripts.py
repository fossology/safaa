# SPDX-License-Identifier: CC-BY-SA-4.0
# SPDX-FileCopyrightText: 2025 Abdulsobur Oyewale

import pandas as pd
from sklearn.model_selection import train_test_split
from Safaa.src.safaa.Safaa import SafaaAgent
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


def find_latest_copyright_file():
    files = glob.glob("data/copyrights_*.csv")
    if not files:
        raise FileNotFoundError("No copyright data files found in 'data/'")
    latest = max(files, key=os.path.getmtime)
    return latest



def main():

    agent = SafaaAgent()

    #Preprocessing
    latest_file = find_latest_copyright_file()
    raw_df = load_data(latest_file)
    raw_data = raw_df['original_content']
    preprocessed_df = preprocess_data(agent, raw_data)
    save_to_csv(preprocessed_df, 'data/preprocessed_copyrights.csv')

    print("✅ Preprocessing completed")

    #Decluttering
    df = load_data('data/preprocessed_copyrights.csv')
    data = df['original_content']
    decluttered_df = declutter_data(agent, data)
    save_to_csv(decluttered_df, 'data/decluttered_copyrights.csv')

    print("✅ Decluttering completed")

    #Data Split
    train_df, test_df = split_data(df)
    save_to_csv(train_df, 'data/train_data.csv')
    save_to_csv(test_df, 'data/test_data.csv')

    print("✅ Split complete: train_data.csv and test_data.csv created.")


if __name__ == '__main__':
    main()
