from sklearn.linear_model import LogisticRegression
from src.score import LABELS, score_submission, print_confusion_matrix

'''

Input:

Output:

'''


class Models:
    def __init__(self, X_train, X_validate, X_test, Y_train, Y_validate, Y_test):
        self.X_train = X_train
        self.X_validate = X_validate
        self.X_test = X_test
        self.Y_train = Y_train
        self.Y_validate = Y_validate
        self.Y_test = Y_test

    def get_lr(self):
        lr = LogisticRegression(random_state=66, multi_class="auto", class_weight='balanced', solver="lbfgs",
                                max_iter=200)

        lr.fit(self.X_train, self.Y_train)
        Y_pred = lr.predict(self.X_validate)
        predicted_labels = [LABELS[int(pred)] for pred in Y_pred]
        actual_labels = [LABELS[int(pred)] for pred in self.Y_validate]

        print("The validation score is:")
        score, confusion_matrix = score_submission(actual_labels, predicted_labels)
        print_confusion_matrix(confusion_matrix)

