import numpy as np

'''
Metrics used for evaluation the models

Input: Confusion matrix produced from the official score.py
from the authors of the challenge

Output: Precision, recall and the f1-score for agree, disagree, discuss and
unrelated.

'''


def performance_metrics(cm):
    TP = np.diag(cm)
    FP = np.sum(cm, axis=0) - TP
    FN = np.sum(cm, axis=1) - TP
    classes = 4
    TN = []
    for i in range(classes):
        temp = np.delete(cm, i, 0)
        temp = np.delete(temp, i, 1)
        TN.append(sum(sum(temp)))

    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1_score = 2 * TP / (2 * TP + FP + FN)

    return precision, recall, f1_score
