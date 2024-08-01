# SPDX-FileCopyrightText: © 2023 abdelrahmanjamal5565@gmail.com
# SPDX-FileContributor: Kaushlendra Pratap Singh <kaushlendra-pratap.singh@siemens.com>
# SPDX-License-Identifier: LGPL-2.1-only

"""
SafaaAgent: A module for handling false positive detection in copyright notices.
"""

import os
import re
import spacy
from joblib import load, dump
import pkg_resources
import shutil

# Constants
DEFAULT_MODEL_DIR = pkg_resources.resource_filename(__name__, "models")
LOCAL_MODEL_DIR = "/home/fossy/Safaa"
CONFIGS_DIR = pkg_resources.resource_filename(__name__, "configs")


class SafaaAgent:
    def __init__(self, use_local_model=True, model_dir=None):
        """
        Initializes the SafaaAgent with the necessary models.

        Parameters:
        use_local_model (bool): Flag to use local models if available.
        model_dir (str): Custom directory path for models. Defaults to None.
        """

        # Determine the model directory based on the provided arguments
        model_dir = (
            model_dir
            if model_dir
            else (
                LOCAL_MODEL_DIR
                if use_local_model and os.path.exists(LOCAL_MODEL_DIR)
                else DEFAULT_MODEL_DIR
            )
        )

        # Construct the file paths for each model
        self.false_positive_detector_path = os.path.join(
            model_dir, "false_positive_detection_model_sgd.pkl"
        )
        self.vectorizer_path = os.path.join(
            model_dir, "false_positive_detection_vectorizer.pkl"
        )
        self.entity_recognizer_path = os.path.join(model_dir, "entity_recognizer")
        self.declutter_model_path = os.path.join(model_dir, "declutter_model")

        # Load the models from the constructed file paths
        self._load_models()

    def _load_models(self):
        """
        Loads models from file paths.
        """

        # Load the False Positive Detector model
        self.false_positive_detector = load(self.false_positive_detector_path)

        # Load the Vectorizer model
        self.vectorizer = load(self.vectorizer_path)

        # Load the Entity Recognizer model using spaCy
        self.entity_recognizer = spacy.load(self.entity_recognizer_path)

        # Load the Declutter model using spaCy
        self.declutter_model = spacy.load(self.declutter_model_path)

    def preprocess_data(self, data):
        """
        Preprocesses the given data by performing various text cleaning and
        transformation tasks.

        Parameters:
        data (iterable): The data to preprocess.

        Returns:
        data (list): List of preprocessed strings.
        """

        # Ensure the data is a list of strings
        data = self._ensure_list_of_strings(data)

        # Replace copyright holder entities in the data
        data = self._replace_entities(data)

        # Perform text substitutions for dates, numbers, symbols, emails, etc.
        data = self._perform_text_substitutions(data)

        return data

    def _ensure_list_of_strings(self, data):
        """
        Ensures the data is a list of strings.

        If the input data is not a list, attempts to convert it to a list.
        Then, ensures each element of the list is a string.

        Parameters:
        data (iterable): The data to be converted to a list of strings.

        Returns:
        list: A list of strings.
        """

        # If data is not a list, try converting it to a list
        if not isinstance(data, list):
            data = data.to_list()
        # Ensure each item in the list is a string
        return [str(item) for item in data]

    def _replace_entities(self, data):
        """
        Replaces detected copyright holder entities with ' ENTITY '.

        Uses the entity_recognizer model to identify copyright holder entities,
        which are often name or organization entities, and replaces them with
        the string ' ENTITY '.

        Parameters:
        data (list): A list of strings.

        Returns:
        list: A list of strings with copyright holder entities replaced.
        """

        new_data = []
        for sentence in data:
            # Process the sentence using the entity recognizer
            doc = self.entity_recognizer(sentence)
            new_sentence = doc.text
            for entity in doc.ents:
                # If the entity is a copyright holder entity, replace it with
                # ' ENTITY '
                if entity.label_ == "ENT":
                    new_sentence = re.sub(
                        re.escape(entity.text), " ENTITY ", new_sentence
                    )
            new_data.append(new_sentence)
        return new_data

    def _perform_text_substitutions(self, data):
        """
        Performs a series of text substitutions to clean and standardize the
        data.

        This includes:
        - Replacing four-digit numbers (assumed to be years) with ' DATE '.
        - Removing all other numbers.
        - Replacing copyright symbols with ' COPYRIGHTSYMBOL '.
        - Replacing emails with ' EMAIL '.
        - Removing any special characters not already replaced or removed.
        - Converting text to lowercase.
        - Stripping extra whitespace from the text.

        Parameters:
        data (list): A list of strings.

        Returns:
        list: A list of cleaned and standardized strings.
        """

        # Define the substitution patterns and their replacements
        subs = [
            (r"\d{4}", " DATE "),
            (r"\d+", " "),
            (r"©", " COPYRIGHTSYMBOL "),
            (r"\(c\)", " COPYRIGHTSYMBOL "),
            (r"\(C\)", " COPYRIGHTSYMBOL "),
            (
                r"""(?:[a-z0-9!#$%&'*+/=?^_`{|}~-]+(?:\.[a-z0-9!#$%&'*+/=?^_`{|}~-]+)*|"(?:[\x01-\x08\x0b\x0c\x0e-\x1f\x21\x23-\x5b\x5d-\x7f]|\\[\x01-\x09\x0b\x0c\x0e-\x7f])*")@(?:(?:[a-z0-9](?:[a-z0-9-]*[a-z0-9])?\.)+[a-z0-9](?:[a-z0-9-]*[a-z0-9])?|\[(?:(?:(2(5[0-5]|[0-4][0-9])|1[0-9][0-9]|[1-9]?[0-9]))\.){3}(?:(2(5[0-5]|[0-4][0-9])|1[0-9][0-9]|[1-9]?[0-9])|[a-z0-9-]*[a-z0-9]:(?:[\x01-\x08\x0b\x0c\x0e-\x1f\x21-\x5a\x53-\x7f]|\\[\x01-\x09\x0b\x0c\x0e-\x7f])+)\])""",
                " EMAIL ",
            ),
            (r"[^a-zA-Z0-9]", " "),
        ]
        # Perform the substitutions for each pattern in the list
        for pattern, replacement in subs:
            data = [re.sub(pattern, replacement, sentence) for sentence in data]
        # Convert text to lowercase and strip extra whitespace
        return [sentence.lower().strip() for sentence in data]

    def predict(self, data, threshold=0.5):
        """
        Predicts false positives in the given data.

        Parameters:
        data (iterable): The data to predict.
        threshold (float): The probability threshold for classification.
                           Defaults to 0.5.

        Returns:
        list: The predictions.
        """

        # Preprocess the data before making predictions
        data = self.preprocess_data(data)

        # Vectorize the preprocessed data using the pre-trained vectorizer
        data = self.vectorizer.transform(data)

        # Check if the model supports probability prediction
        if hasattr(self.false_positive_detector, "predict_proba"):
            # Get probability predictions from the model
            predictions = self.false_positive_detector.predict_proba(data)
            # Classify based on the given threshold. If the threshhold is not
            # met, automatically sets the prediction to true
            return [
                "f" if prediction[1] >= threshold else "t" for prediction in predictions
            ]

        # Get binary predictions from the model if probability prediction is not
        # supported
        return [
            "f" if prediction == 1 else "t"
            for prediction in self.false_positive_detector.predict(data)
        ]

    def declutter(self, data, predictions):
        """
        Cleans up a copyright notice by removing extra text based on the
        predictions.

        Parameters:
        data (iterable): The data to declutter.
        predictions (list): The predictions indicating false positives.

        Returns:
        list: The decluttered data.
        """

        # Iterate over each sentence and its corresponding prediction
        # Remove text from sentences marked as false positives, and keep the
        # entities (copyrights) in other sentences
        return [
            (
                ""
                if prediction == "f"
                else " ".join([ent.text for ent in self.declutter_model(sentence).ents])
            )
            for sentence, prediction in zip(data, predictions)
        ]

    def train_false_positive_detector_model(self, data, labels):
        """
        Trains the false positive detector model iteratively.

        Parameters:
        data (iterable): The data to train the model on.
        labels (iterable): The labels for the training data.
        """

        # Preprocess the data before training
        preprocessed_data = self.preprocess_data(data)
        # Fit the vectorizer to the preprocessed data
        vectorized_data = self.vectorizer.transform(preprocessed_data)
        # Train the false positive detector model
        self.false_positive_detector.partial_fit(vectorized_data, labels)

    def train_ner_model(
        self, train_path, dev_path, declutter_model=False, config_path=None
    ):
        """
        Trains the named entity recognition model using the provided training and
        development datasets.

        Parameters:
        train_path (str): The path to the training data file. Should be a .spacy file
        dev_path (str): The path to the development data file. Should be a .spacy file
        declutter_model (bool): Whether to train the declutter model or the entity recognizer model. Defaults to False.
        config_path (str): The path to the configuration file. Defaults to None.
        """

        # Determine the configuration file path
        cfg_path = config_path or CONFIGS_DIR
        config_file_path = os.path.join(cfg_path, "train.cfg")

        # Read the configuration file
        with open(config_file_path, "r", encoding="utf-8") as file:
            cfg_contents = file.read()

        # Update the training and development data paths in the configuration
        # contents
        updated_cfg_contents = re.sub(
            r'train\s*=\s*".*"',
            f'train = "{train_path}"',
            re.sub(r'dev\s*=\s*".*"', f'dev = "{dev_path}"', cfg_contents),
        )

        # Write the updated configuration to a temporary file
        tmp_cfg_path = os.path.join(cfg_path, "tmp.cfg")
        with open(tmp_cfg_path, "w", encoding="utf-8") as file:
            file.write(updated_cfg_contents)

        # Determine the model directory paths
        tmp_model_path = os.path.join(LOCAL_MODEL_DIR, "tmp")
        new_model_dir = "declutter_model" if declutter_model else "entity_recognizer"
        new_model_path = os.path.join(LOCAL_MODEL_DIR, new_model_dir)

        # Create the new model directory if it doesn't exist
        os.makedirs(new_model_path, exist_ok=True)

        # Construct the training command and execute it
        train_command = (
            f"python -m spacy train '{tmp_cfg_path}' " f"--output '{tmp_model_path}'"
        )
        os.system(train_command)

        # Move the trained model files to the new model directory
        self._move_files(os.path.join(tmp_model_path, "model-best"), new_model_path)

        # Clean up the temporary files and directories
        os.remove(tmp_cfg_path)
        shutil.rmtree(tmp_model_path)

    def _move_files(self, src_dir, dst_dir):
        """
        Internal helper method to move files from one directory to another.

        Parameters:
        src_dir (str): The source directory from where the files are to be moved.
        dst_dir (str): The destination directory where the files are to be moved.
        """

        # List all items in the source directory
        for item in os.listdir(src_dir):
            # Construct full paths for source and destination
            src_item_path = os.path.join(src_dir, item)
            dst_item_path = os.path.join(dst_dir, item)
            # Move each item from the source directory to the destination
            # directory
            shutil.move(src_item_path, dst_item_path)

    def save(self, path=None):
        """
        Saves the trained models and vectorizer to the specified path.

        Make sure the directory you have permissions to the directory that you
        are saving to.

        Parameters:
        path (str): The path to save the models and vectorizer to. Defaults to
                    None.
        """

        # Determine the directory path to save the models and vectorizer
        save_path = path or "/home/fossy/Safaa"

        # Create the directory if it does not exist
        os.makedirs(save_path, exist_ok=True)

        # Check directory permissions
        if not os.access(path, os.W_OK):
            print(
                "Write permissions are not granted for the directory: " f"{save_path}"
            )
            return

        # Construct the full paths for the model and vectorizer files
        false_positive_detector_path = os.path.join(
            save_path, "false_positive_detection_model_sgd.pkl"
        )
        vectorizer_path = os.path.join(
            save_path, "false_positive_detection_vectorizer.pkl"
        )

        # Save the false positive detector model and vectorizer to the specified
        # paths
        dump(self.false_positive_detector, false_positive_detector_path)
        dump(self.vectorizer, vectorizer_path)
