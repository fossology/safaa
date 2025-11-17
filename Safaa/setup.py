# SPDX-FileCopyrightText: © 2023 abdelrahmanjamal5565@gmail.com
#
# SPDX-License-Identifier: LGPL-2.1-only
from os import path

from setuptools import setup, find_packages

here = path.dirname(path.abspath(path.dirname(__file__)))
# fetch the long description from the README.md
with open(path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="safaa",
    version="0.0.3",
    url="https://github.com/fossology/safaa",
    author="Abdelrahman Jamal",
    author_email="abdelrahmanjamal5565@gmail.com",
    maintainer=["Kaushlendra Pratap", "Gaurav Mishra"],
    maintainer_email=[
        "kaushlendra-pratap.singh@siemens.com",
        "mishra.gaurav@siemens.com",
        "fossology-devel@fossology.org",
    ],
    description="""Created as a part of the 2023 Google Summer of Code project:
     Reducing Fossology\'s False Positive Copyrights, the purpose is to be able
     to predict whether a given copyright output from the Fossology software
     is a false positive or not. It is also able to remove extra
     text from a copyright notice.""",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(
        where="src",
    ),
    package_dir={"": "src"},
    install_requires=[
        "spacy==3.8.9",
        "joblib==1.5.2",
        "pandas==2.3.1",
        "scikit-learn==1.6.1",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU Lesser General Public License v2 (LGPLv2)",
    ],
    include_package_data=True,
    include_dirs=[],
    package_data={
        "": [
            "src/safaa/models/*.pkl",
            "src/safaa/models/*.",
            "src/safaa/configs/*",
        ]
    },
    python_requires=">=3.6",
)
