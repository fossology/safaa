# SPDX-FileCopyrightText: © 2023 abdelrahmanjamal5565@gmail.com
#
# SPDX-License-Identifier: LGPL-2.1-only

from argparse import ArgumentParser
import os
import pandas as pd
from sklearn.metrics import classification_report

from safaa.Safaa import *


def main():
    parser = ArgumentParser(description="Read CSV file and check model file")
    parser.add_argument("--csv-file",
                        help="Path to the CSV file - it should two columns:"
                             " text and label")

    args = parser.parse_args()

    # Read the CSV file using Pandas
    try:
        data = pd.read_csv(args.csv_file)
        print(f"Successfully read the CSV file from: {args.csv_file}")
    except FileNotFoundError:
        print(f"CSV file not found at: {args.csv_file}")
        return

    # Check if the Safaa package is installed in the fossy user pythondeps
    # Simply check if a directory containing Safaa is inside
    # /home/fossy/pythondeps
    try:
        dirs = os.listdir('/home/fossy/pythondeps')
        dirs = [d for d in dirs if 'Safaa' in d]
        if len(dirs) == 0:
            print("Warning: Safaa not found in /home/fossy/pythondeps(This is common for local dev )")
    except FileNotFoundError:
        print("Skipping check for /home/fossy/pythondeps(folder does not exist locally)")
    

    # Create an instance of the class
    agent = SafaaAgent()

    predictions = agent.predict(data['text'].to_list())

    # Print the predictions
    print(classification_report(data['label'].astype(str).to_list(), predictions))


if __name__ == "__main__":
    main()
