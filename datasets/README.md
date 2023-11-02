<!--
SPDX-FileCopyrightText: © 2023 abdelrahmanjamal5565@gmail.com

SPDX-License-Identifier: LGPL-2.1-only
-->
# Dataset Information

## False Positive Detection Dataset
Constructed from Multiple GitHub Repositories
- Fossology 
- Tensorflow
- Kubernetes
- Additional dataset provided by Siemens

## Declutter
Constructed from the false positive detection dataset text after removing non-english examples and false positives.
**Currently only the first 4000 rows are labeled and that is what is included in the dataset.**

## Entity Recognition
Similar to declutter, it's also constructed from the false positive detection dataset. But only using 1500 examples.
