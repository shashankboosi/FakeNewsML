"""
COMP9417
Assignment: Fake news Challenge
Authors: Shashank Reddy Boosi(z5222766), Connor McLeod (z5058240), Leonard Lee(z5173917), Darren Zhang (z5113901)
main.py: Main file for program execution
"""

from src.data_import import FakeNewsData
from src.train_validation_split import DataSplit
from src.preprocess import Preprocess
from src.feature_extraction import Features
from src.models import Models
from src.score import LABELS
from src.utils import input_file, output_file
import scipy.sparse as sp

import os
import time

# Global Variables
trainStancePath = "data/train_stances.csv"
testStancePath = "data/competition_test_stances.csv"
trainBodyPath = "data/train_bodies.csv"
testBodyPath = "data/competition_test_bodies.csv"

# header attributes
primary_id = "Body ID"
stance = "Stance"
body = "articleBody"
headline = "Headline"
base_preprocess_path = "preprocessed_data"
base_feature_path = "final_features"
output = "output"


def target_labels(stances):
    labels = []
    for i in range(len(stances)):
        labels.append(LABELS.index(stances[i][stance]))

    return labels


def headlines_bodies(temp_headline, temp_body):
    headlines = []
    bodies = []
    for i in range(len(temp_headline)):
        bodies.append(temp_body[int(temp_headline[i][primary_id])])
        headlines.append(temp_headline[i][headline])

    return headlines, bodies


'''
This files combine all the data mining part starting from Data importing, spliting,
pre processing, feature transformation, modelling and visualization.

Check README for clear understanding of what is happening.

'''
if __name__ == "__main__":

    t0 = time.time()
    # Importing the data
    train = FakeNewsData(trainStancePath, trainBodyPath)
    test = FakeNewsData(testStancePath, testBodyPath)

    # Extracting IDs for data splitting
    ids = list(train.articleBody.keys())

    # The DataSplit generates the train and validation splits according to our split size
    print("Data Splitting")
    train_validation_split = DataSplit(ids=ids, headline=train.headlineInstances, split_size=0.8)
    train_stances, validation_stances = train_validation_split.split()

    # Preprocess the train
    print("Start of pre-processing for train")
    if not (os.path.exists(base_preprocess_path + "/" + "training_headlines.p") and os.path.exists(
            base_preprocess_path + "/" + "training_bodies.p")):
        preprocessed_train_data = Preprocess(headline=train_stances, body=train.articleBody,
                                             preprocess_type="lemma")
        train_preprocessed_headlines, train_preprocessed_bodies = preprocessed_train_data.get_clean_headlines_and_bodies()
        output_file(train_preprocessed_headlines, base_preprocess_path + "/" + "training_headlines.p")
        output_file(train_preprocessed_bodies, base_preprocess_path + "/" + "training_bodies.p")
    else:
        train_preprocessed_headlines = input_file(base_preprocess_path + "/" + "training_headlines.p")
        train_preprocessed_bodies = input_file(base_preprocess_path + "/" + "training_bodies.p")

    # Preprocess the validation
    print("Start of pre-processing for validation")
    if not (os.path.exists(base_preprocess_path + "/" + "validation_headlines.p") and os.path.exists(
            base_preprocess_path + "/" + "validation_bodies.p")):
        preprocessed_validation_data = Preprocess(headline=validation_stances, body=train.articleBody,
                                                  preprocess_type="lemma")
        validation_preprocessed_headlines, validation_preprocessed_bodies = preprocessed_validation_data.get_clean_headlines_and_bodies()
        output_file(validation_preprocessed_headlines, base_preprocess_path + "/" + "validation_headlines.p")
        output_file(validation_preprocessed_bodies, base_preprocess_path + "/" + "validation_bodies.p")
    else:
        validation_preprocessed_headlines = input_file(base_preprocess_path + "/" + "validation_headlines.p")
        validation_preprocessed_bodies = input_file(base_preprocess_path + "/" + "validation_bodies.p")

    # Preprocess the test
    print("Start of pre-processing for test")
    if not (os.path.exists(base_preprocess_path + "/" + "test_headlines.p") and os.path.exists(
            base_preprocess_path + "/" + "test_bodies.p")):
        preprocessed_test_data = Preprocess(headline=test.headlineInstances, body=test.articleBody,
                                            preprocess_type="lemma")
        test_preprocessed_headlines, test_preprocessed_bodies = preprocessed_test_data.get_clean_headlines_and_bodies()
        output_file(test_preprocessed_headlines, base_preprocess_path + "/" + "test_headlines.p")
        output_file(test_preprocessed_bodies, base_preprocess_path + "/" + "test_bodies.p")
    else:
        test_preprocessed_headlines = input_file(base_preprocess_path + "/" + "test_headlines.p")
        test_preprocessed_bodies = input_file(base_preprocess_path + "/" + "test_bodies.p")

    # Split headlines and bodies for train, validation and test
    train_headlines, train_bodies = headlines_bodies(train_stances, train.articleBody)
    validation_headlines, validation_bodies = headlines_bodies(validation_stances, train.articleBody)
    test_headlines, test_bodies = headlines_bodies(test.headlineInstances, test.articleBody)

    if not (os.path.exists(base_feature_path + "/" + "train_features.p") and os.path.exists(
            base_feature_path + "/" + "validation_features.p") and os.path.exists(
        base_feature_path + "/" + "test_features.p")):

        # Feature extraction and combining them for the models
        print("Feature extraction for train")
        train_features = Features(train_preprocessed_headlines, train_preprocessed_bodies,
                                  train_headlines,
                                  train_bodies)

        # TF-IDF weight extraction
        train_tfidf_weights, validation_tfidf_weights, test_tfidf_weights = train_features.tfidf_extraction(
            validation_headlines, validation_bodies, test_headlines, test_bodies)

        # Sentence weighting for train
        train_sentence_weights = train_features.sentence_weighting()

        print("Feature extraction for validation")
        validation_features = Features(validation_preprocessed_headlines, validation_preprocessed_bodies,
                                       validation_headlines, validation_bodies)
        # Sentence weighting for validation
        validation_sentence_weights = validation_features.sentence_weighting()

        print("Feature extraction for test")
        test_features = Features(test_preprocessed_headlines, test_preprocessed_bodies,
                                 test_headlines, test_bodies)
        # Sentence weighting for test
        test_sentence_weights = test_features.sentence_weighting()

        # Combine the features to prepare them as an inout for the models
        final_train_features = sp.bmat([[train_tfidf_weights, train_sentence_weights.T]]).A
        output_file(final_train_features, base_feature_path + "/" + "train_features.p")
        final_validation_features = sp.bmat([[validation_tfidf_weights, validation_sentence_weights.T]]).A
        output_file(final_validation_features, base_feature_path + "/" + "validation_features.p")
        final_test_features = sp.bmat([[test_tfidf_weights, test_sentence_weights.T]]).A
        output_file(final_test_features, base_feature_path + "/" + "test_features.p")
    else:
        print("Feature Extraction")
        final_train_features = input_file(base_feature_path + "/" + "train_features.p")
        final_validation_features = input_file(base_feature_path + "/" + "validation_features.p")
        final_test_features = input_file(base_feature_path + "/" + "test_features.p")

    t1 = time.time()

    print("Time for feature extraction is:", t1 - t0)

    # Target variables
    train_target_labels = target_labels(train_stances)
    validation_target_labels = target_labels(validation_stances)
    test_target_labels = target_labels(test.headlineInstances)

    # Modelling the features
    print("Start of Modelling")
    models = Models(final_train_features, final_validation_features, final_test_features, train_target_labels,
                    validation_target_labels, test_target_labels)

    # Calling the 4 models
    models.get_lr()
    models.get_dt()
    models.get_nb()
    models.get_rf()

    '''
    Used read_from_csv in utils to know the actual labels and the predicted labels
    to produce the correctness visualizations graphs for the report.
    '''

    t2 = time.time()
    print("Time for the total is:", t2 - t0)

    print("\nEnd of tests\n")
