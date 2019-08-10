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
import random

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

'''
The main function should call the classes that we declare and
the operations should happen in the main function and the logic should 
be written in their respective python files
'''
if __name__ == "__main__":
    train = FakeNewsData(trainStancePath, trainBodyPath)
    test = FakeNewsData(testStancePath, testBodyPath)

    # For train
    print('The first stance for train', train.headlineInstances[0])
    print('The length of train stances is', len(train.headlineInstances))
    print('The length of train body is', len(train.articleBody))

    # For test
    print('The first stance for test', test.headlineInstances[0])
    print('The length of test stances is', len(test.headlineInstances))
    print('The length of test body is', len(test.articleBody))

    # Extracting IDs for data splitting
    ids = list(train.articleBody.keys())

    # The DataSplit generates the train and validation splits according to our split size
    print("Data Splitting")
    train_validation_split = DataSplit(ids=ids, headline=train.headlineInstances, split_size=0.8)
    train_stances, validation_stances = train_validation_split.split()
    print(train_stances[0])

    # Preprocess the train, validation and test
    print("Start of pre-processing for train")
    preprocessed_train_data = Preprocess(headline=train_stances[:100], body=train.articleBody, preprocess_type="lemma")
    train_headlines, train_bodies = preprocessed_train_data.get_clean_headlines_and_bodies()
    print(train_headlines[0])
    print(train_bodies[0])

    # Choosing the type of trimming for the words

    # connors_model()
# your_model_goes_here

print("\nEnd of tests\n")
