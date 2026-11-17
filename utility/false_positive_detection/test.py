# SPDX-FileCopyrightText: © 2023 abdelrahmanjamal5565@gmail.com
#
# SPDX-License-Identifier: LGPL-2.1-only

"""Report the installed Safaa false positive detector's accuracy on a labelled CSV."""

from argparse import ArgumentParser

import pandas as pd
from sklearn.metrics import classification_report

from safaa.Safaa import SafaaAgent


def main():
    parser = ArgumentParser(description="Evaluate the false positive detector on a labelled CSV")
    parser.add_argument("--csv-file", required=True,
                        help="CSV with a text column and a 0/1 label column (1 = false positive)")
    parser.add_argument("--text-col", default=None, help="text column (default: 'text' or 'copyright')")
    parser.add_argument("--label-col", default=None, help="label column (default: 'label' or 'falsePositive')")
    args = parser.parse_args()

    try:
        data = pd.read_csv(args.csv_file, keep_default_na=False)
    except FileNotFoundError:
        print(f"CSV file not found at: {args.csv_file}")
        return
    text_col = args.text_col or next(c for c in ("text", "copyright") if c in data.columns)
    label_col = args.label_col or next(c for c in ("label", "falsePositive") if c in data.columns)

    agent = SafaaAgent()
    # "t"/"f" predictions vs 0 (real) / 1 (false positive) labels
    predictions = [1 if p == "f" else 0 for p in agent.predict(data[text_col].astype(str).to_list())]
    print(classification_report(data[label_col].astype(int).to_list(), predictions,
                                target_names=["real copyright (0)", "false positive (1)"], digits=4))


if __name__ == "__main__":
    main()
