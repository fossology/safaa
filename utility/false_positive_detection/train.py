# SPDX-FileCopyrightText: © 2023 abdelrahmanjamal5565@gmail.com
#
# SPDX-License-Identifier: LGPL-2.1-only

from argparse import ArgumentParser

import pandas as pd
import os
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

    agent = SafaaAgent()
    save_path_initial = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "Safaa/src/safaa/models/")
    agent.train_false_positive_detector_model(data['copyright'].to_list(),
                                              data["falsePositive"].to_list())
    agent.save(path=save_path_initial)


if __name__ == "__main__":
    main()
