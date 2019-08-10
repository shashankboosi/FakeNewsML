"""
COMP9417
Assignment
Authors: Connor McLeod (z5058240), name (zid), name (zid), name (zid)
main.py: Main file for program execution
"""

from src.connors_model import *
from src.data_import import FakeNewsData
from src.train_validation_split import DataSplit
from src.feature_transformation import Features
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
    train_validation_split = DataSplit(ids=ids, headline=train.headlineInstances, body=train.articleBody,
                                       split_size=0.8)
    train_stances, validation_stances = train_validation_split.split()
    print(train_stances[0])

    # connors_model()
# your_model_goes_here

print("\nEnd of tests\n")

""" 
To do:
- Add your own model implementation to this mainfile by calling it from another file
- Document your research into your selected model for text classification
- ?
"""
