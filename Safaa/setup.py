from setuptools import setup, find_packages

setup(
    name='safaa',
    version='0.0.1',
    url='https://github.com/fossology/copyrightfpd',
    author='Abdelrahman Jamal',
    author_email='abdelrahmanjamal5565@gmail.com',
    description="""Created as a part of the 2023 Google Summer of Code project:
     Reducing Fossology\'s False Positive Copyrights, the purpose is to be able to
     predict whether a given copyright output from the Fossology software
     is a false positive or not. It is also able to remove extra
     text from a copyright notice.""",
    packages=find_packages(where='src',),
    package_dir = {"": "src"},
    install_requires=[
        'spacy>=3.0.0',
        'joblib>=1.0.0',
        'pandas>=1.1.0',
        'scikit-learn>=1.3.0',
    ],
    classifiers=[
    'Programming Language :: Python :: 3',
    'License :: OSI Approved :: GNU Lesser General Public License v2.1 (LGPLv2.1)',
    ],
    include_package_data=True,
    include_dirs=[],
    package_data={'': ['src/safaa/models/*.pkl', 'src/safaa/models/*.',
                       'src/safaa/configs/*']},
    python_requires='>=3.6',
)
