from tqdm import tqdm
import json
import re
import spacy
from sklearn.model_selection import train_test_split

def text_to_json(sentences):
    """
    text (str): source text
    Returns (list (dict)): deccano format json
    """
    ## Converts list of string to json file to start labeling them.
    new_json = list()
    for sentence in tqdm(sentences):
        labels = list()
        new_json.append({'text': sentence, "labels": labels})
    return new_json

def text_to_json_model_assisted(sentences, model):
    # Uses the NER model to predict the labels before hand. Helps as a way of 
    # partially labeling the datasets before manually clearning it.
    sentences = model(sentences)
    new_json = list()
    for sentence in tqdm(sentences):
        labels = list()
        for e in sentence.ents:
            labels.append([e.start_char, e.end_char, e.label_])
        new_json.append({'text': sentence.text, "labels": labels})
    return new_json

def text_to_json_labels_separate(sentences, labels, entity_name):
    # Assumes that the labels are provided separately, and are either part of the 
    # entire sentence. if label is '' or None, removes that row. If label can't be matched
    # inside sentence, removes that row

    # entity_name is the name of the NER entity used for labeling

    new_json = list()
    for sentence, label in tqdm(zip(sentences, labels)):
        if label is None or label == '':
            continue
        
        pattern = r"\b" + re.escape(label) + r"\b"
        match = re.search(pattern, sentence)
        if not match:
            continue

        new_json.append({'text': sentence.text, "labels": [match.start, 
                                                       match.end, entity_name]})
   
    return new_json

def write_json_to_disk(new_json, path):
    # make sure the file is jsonL if using doccano
    with open(path, 'w') as f:
        for item in new_json:
            f.write(json.dumps(item) + '\n')

# Convert JSONL to spacy compatible format
def convert_jsonl_to_spacy(jsonl_path, spacy_path):
    """
    Load JSONL data from a file and convert it to spaCy's training format.
    :param file_path: The path of the JSONL file.
    :return: A list of training examples.
    """
    data = []
    with open(jsonl_path, 'r') as f:
        for line in f:
            item = json.loads(line)
            text = item['text']
            entities = [(e[0], e[1], e[2]) for e in item['label']]
            data.append((text, {'entities': entities}))
    
    nlp = spacy.blank('en')
    doc_bin = spacy.tokens.DocBin()
    for text, annotations in data:
        doc = nlp.make_doc(text)
        example = spacy.training.Example.from_dict(doc, annotations)
        doc_bin.add(example.reference)
    doc_bin.to_disk(spacy_path)

def spacy_train_test_split(file_path, split=0.2, random_state=42, shuffle=True):

    doc_bin = spacy.tokens.DocBin().from_disk(file_path)
    nlp = spacy.load("en_core_web_sm")
    docs = list(doc_bin.get_docs(nlp.vocab))
    train_docs, test_docs = train_test_split(docs, test_size=split, random_state=random_state, shuffle=shuffle)
    train_doc_bin = spacy.tokens.DocBin(docs=train_docs)
    test_doc_bin = spacy.tokens.DocBin(docs=test_docs)
    
    train_path = file_path.split('.spacy')[0] + '-train.spacy'
    test_path = file_path.split('.spacy')[0] + '-test.spacy'

    train_doc_bin.to_disk(train_path)
    test_doc_bin.to_disk(test_path)

    