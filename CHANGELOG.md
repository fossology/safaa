<!-- SPDX-FileCopyrightText: © Fossology contributors

     SPDX-License-Identifier: LGPL-2.1-only
-->
# Changelog for Safaa

All notable changes to this project will be documented in this file.  
This project adheres to [Semantic Versioning](https://semver.org/).
---

## Safaa 0.0.5  (2026-11-16)

This release adds support for Python 3.14, so that Safaa can be installed on
Ubuntu 26.04, and drops support for Python 3.9. On Python 3.14, spaCy 3.8.18
and scikit-learn 1.7.2 are used; older Python versions keep spaCy 3.8.7 and
scikit-learn 1.6.1. pandas is upgraded from 2.3.1 to 2.3.3. The bundled models
are unchanged. The release also introduces a data pipeline to iteratively
train the false positive detector model from a labeled CSV, locally or through
GitHub Actions.

### Authors:
```
> Kaushlendra Pratap <kaushlendra-pratap.singh@siemens.com>
> Sushant Kumar <sushant.kumar@siemens-healthineers.com>
```
### Changes:
* `161c60a` feat(deps): support Python 3.14
* `3b20166` feat(datapipeline): Data Pipeline for Model Training
* `e11a762` chore(deps): bump idna from 3.13 to 3.15

---

## Safaa 0.0.4  (2026-05-12)

This release focuses on dependency updates, build improvements, and CI/CD
enhancements. Key updates include upgrading joblib and urllib3 to their
latest versions, implementing Poetry caching in CI workflows, and upgrading
the Poetry configuration to version 2 specification. Additional improvements
include extended Poetry support for better build compatibility and resolution
of user warnings. The release also updates CI action versions
(fsfe/reuse-action 5 -> 6, actions/checkout 5 -> 6) to maintain
compatibility with the latest toolchain.

### Authors:
```
> Gaurav Mishra <mishra.gaurav@siemens.com>
> Kaushlendra Pratap <kaushlendra-pratap.singh@siemens.com>
```
### Changes:
* `8bc951b` chore(safaa): dependency upgrade for joblib
* `1e8c21c` chore(deps): bump urllib3 from 2.6.3 to 2.7.0
* `a64ebde` feat(ci): use caching for poetry
* `525914f` chore(poetry): upgrade version 2 spec
* `2b8dd45` fix(build): extend poetry support and fix user warnings
* `553dd77` chore(deps): bump fsfe/reuse-action from 5 to 6
* `cdac0ae` chore(deps): bump actions/checkout from 5 to 6

---

## Safaa 0.0.3  (2025-11-06)
This release updates pinned dependencies and CI action versions to keep the project compatible with the latest toolchain.
Key changes include bumps to CI actions (actions/setup-python 5 → 6, actions/checkout 4 → 5), dependency upgrades (joblib 1.5.1 → 1.5.2, pandas 2.3.0 → 2.3.1, spacy 3.8.4 → 3.8.7), and other dependency updates listed below. The release also adds Dependabot configuration and reviewer/assignee automation via `dependabot.yml` to enable automated dependency updates.
### Authors:
```
> Gaurav Mishra <mishra.gaurav@siemens.com>
```
### Changes:
* `8a470ef` chore(deps): bump actions/setup-python from 5 to 6
* `9e0da4f` chore(deps): bump joblib from 1.5.1 to 1.5.2 in /Safaa
* `017d18d` chore(deps): bump actions/checkout from 4 to 5
* `60c2c39` chore(deps): bump pandas from 2.3.0 to 2.3.1 in /Safaa
* `0048da8` chore(deps): bump pandas from 2.2.3 to 2.3.0 in /Safaa
* `bb3e356` chore(deps): bump spacy from 3.8.4 to 3.8.7 in /Safaa
* `c163db3` chore(deps): bump joblib from 1.2.0 to 1.5.1 in /Safaa
* `a2e930b` chore(dependabot): add reviewer and assignee for pip deps
* `fce3766` chore(deps): bump actions/checkout from 3 to 4
* `d1edf34` chore(deps): bump actions/setup-python from 4 to 5
* `5f2044c` feat(deps): add dependabot.yml
---

## Safaa 0.0.2  (2025-01-23)
This release introduces a new model for incremental learning using scikit-learn's SGD Classifier. The model is trained on the provided dataset and can be used for incremental learning tasks. Dependencies have been pinned to avoid conflicts and a pkl file has been added to the repository.
### Authors:
```
> Kaushlendra Pratap <kaushlendra-pratap.singh@siemens.com>
> Gaurav Mishra <mishra.gaurav@siemens.com>
```
### Changes:
* `831e05f` feat(workflows): Add Required Workflows for Build, Code Quality and Compatibility
* `b6d6999` fix(safaa): Pin conflicting dependencies and pkl file
* `28df119` Update: Introduced scikit-learn based SGD Classifier model for incremental learning
* `95b4623` feat(cd): use oidc instead of token
---



## Safaa 0.0.1  (2023-11-2)

This release introduces the Safaa package, which includes a false positive detector and a named entity recognition model. The package can be used to preprocess data, predict false positives, declutter copyright notices, train models, and save trained models.
### Authors:
```
> Gaurav Mishra <mishra.gaurav@siemens.com>
> Hero2323 <abdelrahmanjamal5565@gmail.com>
```

### Changes:
* `e9188b8` fix(package): fix pypi warnings
* `3c56981` fix(cd): add the correct directory path
* `d36aa6e` feat(ci): lint and publish
* `97e9af6` chore(repo): make reuse compliant
* `8d9008d` Update Safaa/setup.py
* `325a503` Update Safaa/MANIFEST.in
* `87cf656` Update README.md
* `d2eb6c5` updated package to safaa instead of Safaa, updated the license
* `1a56c90` Updated the data conversion file documentation
* `5f8a609` Added the package code after renaming it to Safaa, added training scripts for false positive detection, the datasets, as well as utility conversion scripts
---
