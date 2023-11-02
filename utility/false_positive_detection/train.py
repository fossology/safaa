# SPDX-FileCopyrightText: © 2023 abdelrahmanjamal5565@gmail.com
#
# SPDX-License-Identifier: LGPL-2.1-only

from argparse import ArgumentParser

import pandas as pd

from safaa.Safaa import *


def main():
    parser = ArgumentParser(description="Read CSV file and check model file")
    parser.add_argument("--csv-file",
                        help="Path to the CSV file - it should two columns:"
                             " text and label")

    args = parser.parse_args()

    # Read the CSV file using Pandas
    try:
        data = pd.read_csv(args.csv_file_path)
        print(f"Successfully read the CSV file from: {args.csv_file_path}")
    except FileNotFoundError:
        print(f"CSV file not found at: {args.csv_file_path}")
        return

    # Check if the Safaa package is installed in the fossy user pythondeps
    # Simply check if a directory containing Safaa is inside
    # /home/fossy/pythondeps
    dirs = os.listdir('/home/fossy/pythondeps')
    dirs = [d for d in dirs if 'Safaa' in d]

    if len(dirs) == 0:
        print("""
        The Safaa package is not installed in the fossy user pythondeps. 
        Please install by running the post-install script with the --python-experimental flag""")
        return

    agent = SafaaAgent()

    agent.train_false_positive_detector_model(data['text'].to_list(),
                                              data['label'].to_list())

    agent.save()


if __name__ == "__main__":
    main()
