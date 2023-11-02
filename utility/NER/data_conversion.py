# SPDX-FileCopyrightText: © 2023 abdelrahmanjamal5565@gmail.com
#
# SPDX-License-Identifier: LGPL-2.1-only

from tqdm import tqdm
import json
import re
import spacy
from sklearn.model_selection import train_test_split


def text_to_json(sentences):
    """
    Convert list of sentences to a specific JSON format.
    """
    new_json = list()
    for sentence in tqdm(sentences):
        labels = list()
        # Appending the sentence and its (empty) labels to the resulting JSON
        new_json.append({'text': sentence, "labels": labels})
    return new_json


def text_to_json_model_assisted(sentences, model):
    """
    Convert list of sentences to a JSON format using a model for predictions.
    """
    sentences = model(sentences)
    new_json = list()
    for sentence in tqdm(sentences):
        labels = list()
        for e in sentence.ents:  # Iterating through detected entities in the sentence
            # Appending start, end character positions and label of the entity
            labels.append([e.start_char, e.end_char, e.label_])
        # Appending the sentence and its detected labels to the resulting JSON
        new_json.append({'text': sentence.text, "labels": labels})
    return new_json


def text_to_json_labels_separate(sentences, labels, entity_name):
    """
    Convert list of sentences and labels to a JSON format with specific entity
    name.
    """
    new_json = list()
    for sentence, label in tqdm(zip(sentences, labels)):
        if label is None or label == '':
            continue  # If label is empty or None, skip the iteration

        # Search for the exact match of the label in the sentence
        pattern = r"\b" + re.escape(label) + r"\b"
        match = re.search(pattern, sentence)
        if not match:
            continue  # If label is not found in the sentence, skip

        # Appending the sentence and the position of its label to the resulting
        # JSON
        new_json.append({'text': sentence,
                         "labels": [match.start(), match.end(), entity_name]})
    return new_json


def write_json_to_disk(new_json, path):
    """
    Write a list of dictionaries in the JSONL format to disk.
    """
    with open(path, 'w') as f:
        for item in new_json:
            # Writing each dictionary in the list as a separate line in the
            # JSONL file
            f.write(json.dumps(item) + '\n')


def convert_jsonl_to_spacy(jsonl_path, spacy_path):
    """
    Load data from a JSONL file and convert it to spaCy's training format.
    """
    data = []
    with open(jsonl_path, 'r') as f:
        for line in f:
            item = json.loads(line)
            text = item['text']
            # Extracting entities from the JSONL line
            entities = [(e[0], e[1], e[2]) for e in item['labels']]
            data.append((text, {'entities': entities}))

    nlp = spacy.blank('en')  # Creating a blank English NLP object
    doc_bin = spacy.tokens.DocBin()  # Initializing a DocBin for efficient storage of `Doc` objects
    for text, annotations in data:
        doc = nlp.make_doc(text)  # Creating a `Doc` object from text
        example = spacy.training.Example.from_dict(doc,
                                                   annotations)  # Creating an example from the `Doc` and its annotations
        doc_bin.add(example.reference)  # Adding the `Doc` to the DocBin
    doc_bin.to_disk(spacy_path)  # Saving the DocBin to disk


def spacy_train_test_split(file_path, split=0.2, random_state=42, shuffle=True):
    """
    Split spaCy formatted data into training and testing sets.
    """
    doc_bin = spacy.tokens.DocBin().from_disk(
        file_path)  # Loading the DocBin from disk
    nlp = spacy.load("en_core_web_sm")  # Loading the English small core model
    docs = list(doc_bin.get_docs(
        nlp.vocab))  # Retrieving `Doc` objects from the DocBin using the model's vocabulary
    # Splitting the `Doc` objects into training and testing sets
    train_docs, test_docs = train_test_split(docs, test_size=split,
                                             random_state=random_state,
                                             shuffle=shuffle)
    train_doc_bin = spacy.tokens.DocBin(
        docs=train_docs)  # Creating a DocBin for training docs
    test_doc_bin = spacy.tokens.DocBin(
        docs=test_docs)  # Creating a DocBin for testing docs

    # Deriving the paths for saving the training and testing DocBins
    train_path = file_path.split('.spacy')[0] + '-train.spacy'
    test_path = file_path.split('.spacy')[0] + '-test.spacy'

    train_doc_bin.to_disk(train_path)  # Saving the training DocBin to disk
    test_doc_bin.to_disk(test_path)  # Saving the testing DocBin to disk
