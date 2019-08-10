"""
COMP9417
Assignment
Authors: Connor McLeod (z5058240), name (zid), name (zid), name (zid)
main.py: Main file for program execution
"""

from src.connors_model import *
from src.data_import import FakeNewsData
from src.train_validation_split import DataSplit
from src.preprocess import Preprocess
from src.feature_extraction import Features
from src.utils import input_file
from src.utils import output_file
import os

# Global Variables
trainStancePath = "data/train_stances.csv"
testStancePath = "data/competition_test_stances.csv"
trainBodyPath = "data/train_bodies.csv"
testBodyPath = "data/competition_test_bodies.csv"

# header attributes
primary_id = "Body ID"
stance = "stance"
body = "articleBody"
headline = "Headline"
base_path = "preprocessed_data"


def headlines_bodies(temp_headline, temp_body):
    headlines = []
    bodies = []
    for i in range(len(temp_headline)):
        bodies.append(temp_body[int(temp_headline[i]['Body ID'])])
        headlines.append(temp_headline[i]['Headline'])

    return headlines, bodies


'''
The main function should call the classes that we declare and
the operations should happen in the main function and the logic should 
be written in their respective python files
'''
if __name__ == "__main__":

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
    if not (os.path.exists(base_path + "/" + "training_headlines.p") and os.path.exists(
            base_path + "/" + "training_bodies.p")):
        preprocessed_train_data = Preprocess(headline=train_stances[:100], body=train.articleBody,
                                             preprocess_type="lemma")
        train_preprocessed_headlines, train_preprocessed_bodies = preprocessed_train_data.get_clean_headlines_and_bodies()
        output_file(train_preprocessed_headlines, base_path + "/" + "training_headlines.p")
        output_file(train_preprocessed_bodies, base_path + "/" + "training_bodies.p")
    else:
        train_preprocessed_headlines = input_file(base_path + "/" + "training_headlines.p")
        train_preprocessed_bodies = input_file(base_path + "/" + "training_bodies.p")

    # Preprocess the validation
    print("Start of pre-processing for validation")
    if not (os.path.exists(base_path + "/" + "validation_headlines.p") and os.path.exists(
            base_path + "/" + "validation_bodies.p")):
        preprocessed_validation_data = Preprocess(headline=validation_stances[:100], body=train.articleBody,
                                                  preprocess_type="lemma")
        validation_preprocessed_headlines, validation_preprocessed_bodies = preprocessed_validation_data.get_clean_headlines_and_bodies()
        output_file(validation_preprocessed_headlines, base_path + "/" + "validation_headlines.p")
        output_file(validation_preprocessed_bodies, base_path + "/" + "validation_bodies.p")
    else:
        validation_preprocessed_headlines = input_file(base_path + "/" + "validation_headlines.p")
        validation_preprocessed_bodies = input_file(base_path + "/" + "validation_bodies.p")

    # Preprocess the test
    print("Start of pre-processing for test")
    if not (os.path.exists(base_path + "/" + "test_headlines.p") and os.path.exists(
            base_path + "/" + "test_bodies.p")):
        preprocessed_test_data = Preprocess(headline=test.headlineInstances[:100], body=test.articleBody,
                                            preprocess_type="lemma")
        test_preprocessed_headlines, test_preprocessed_bodies = preprocessed_test_data.get_clean_headlines_and_bodies()
        output_file(test_preprocessed_headlines, base_path + "/" + "test_headlines.p")
        output_file(test_preprocessed_bodies, base_path + "/" + "test_bodies.p")
    else:
        test_preprocessed_headlines = input_file(base_path + "/" + "test_headlines.p")
        test_preprocessed_bodies = input_file(base_path + "/" + "test_bodies.p")


    # connors_model()
    # your_model_goes_here

    print("\nEnd of tests\n")
