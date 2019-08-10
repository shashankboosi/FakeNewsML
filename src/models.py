from sklearn.linear_model import LogisticRegression
from src.score import LABELS, score_submission, print_confusion_matrix, score_defaults
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier

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
                                max_iter=150)

        lr.fit(self.X_train, self.Y_train)

        # Validation results
        Y_val_pred = lr.predict(self.X_validate)
        predicted_validation_labels = [LABELS[int(pred)] for pred in Y_val_pred]
        actual_validation_labels = [LABELS[int(pred)] for pred in self.Y_validate]
        print(actual_validation_labels)

        validation_score, validation_confusion_matrix = score_submission(actual_validation_labels,
                                                                         predicted_validation_labels)
        print_confusion_matrix(validation_confusion_matrix)

        null_score, max_score = score_defaults(actual_validation_labels)
        print("Percentage of validation score for Logistic Regression is:", validation_score / float(max_score))

        # Test results
        Y_test_pred = lr.predict(self.X_test)
        predicted_test_labels = [LABELS[int(pred)] for pred in Y_test_pred]
        actual_test_labels = [LABELS[int(pred)] for pred in self.Y_test]

        test_score, test_confusion_matrix = score_submission(actual_test_labels, predicted_test_labels)
        print_confusion_matrix(test_confusion_matrix)
        null_score, max_score = score_defaults(actual_test_labels)
        print("Percentage of test score for Logistic Regression is:", test_score / float(max_score))

    def get_dt(self):
        pass

    def get_svm(self):
        pass

    def get_nb(self):
        pass

    def get_rf(self):
        pass
