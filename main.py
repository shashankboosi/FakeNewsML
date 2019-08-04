"""
COMP9417
Assignment
Authors: Connor McLeod (z5058240), name (zid), name (zid), name (zid)
main.py: Main file for program execution
"""

from src.connors_model import *
from src.data_import import FakeNewsData

# Global Variables
trainStancePath = "data/train_stances.csv"
testStancePath = "data/test_stances_unlabeled.csv"
trainBodyPath = "data/train_bodies.csv"
testBodyPath = "data/test_bodies.csv"

'''
The main function should call the clsses that we declare and
the operations should happen in the main function and the logic should 
be written in their respective python files
'''
if __name__ == "__main__":
    train = FakeNewsData(trainStancePath, trainBodyPath)
    test = FakeNewsData(testStancePath, testBodyPath)

    # For train
    print('The first stance for train', train.stances[0])
    print('The length of train stances is', len(train.stances))
    print('The length of train body is', len(train.articleBody))

    # For test
    print('The first stance for test', test.stances[0])
    print('The length of test stances is', len(test.stances))
    print('The length of test body is', len(test.articleBody))

    connors_model()

    # your_model_goes_here

    print("\nEnd of tests\n")

    """ 
    To do:
    - Add your own model implementation to this mainfile by calling it from another file
    - Document your research into your selected model for text classification
    - ?
    """
