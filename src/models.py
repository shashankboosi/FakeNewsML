from sklearn.linear_model import LogisticRegression
from src.score import LABELS, score_submission, print_confusion_matrix, score_defaults
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
import src.metrics as metrics
from src.utils import count_stances
from src.utils import write_to_csv

# Global attributes
output = "output"

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
                                max_iter=340)

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
        count_stances(actual_test_labels)
        print(actual_test_labels)

        # CSV output
        write_to_csv(output + "/" + "lr_actual_labels.csv", actual_test_labels)
        write_to_csv(output + "/" + "lr_predicted_labels.csv", predicted_test_labels)

        test_score, test_confusion_matrix = score_submission(actual_test_labels, predicted_test_labels)
        print_confusion_matrix(test_confusion_matrix)
        null_score, max_score = score_defaults(actual_test_labels)
        print("Percentage of test score for Logistic Regression is:", test_score / float(max_score))

        precision, recall, f1 = metrics.performance_metrics(validation_confusion_matrix)
        print("Precision for LR: ", precision)
        print("Recall for LR:", recall)
        print("F1 Score for LR:", f1)

    def get_dt(self):
        dt = DecisionTreeClassifier(random_state=66, max_depth=10)

        dt.fit(self.X_train, self.Y_train)

        # Validation results
        Y_val_pred = dt.predict(self.X_validate)
        predicted_validation_labels = [LABELS[int(pred)] for pred in Y_val_pred]
        actual_validation_labels = [LABELS[int(pred)] for pred in self.Y_validate]
        print(actual_validation_labels)

        validation_score, validation_confusion_matrix = score_submission(actual_validation_labels,
                                                                         predicted_validation_labels)
        print_confusion_matrix(validation_confusion_matrix)

        null_score, max_score = score_defaults(actual_validation_labels)
        print("Percentage of validation score for Decision Tree is:", validation_score / float(max_score))

        # Test results
        Y_test_pred = dt.predict(self.X_test)
        predicted_test_labels = [LABELS[int(pred)] for pred in Y_test_pred]
        actual_test_labels = [LABELS[int(pred)] for pred in self.Y_test]

        write_to_csv(output + "/" + "dt_actual_labels.csv", actual_test_labels)
        write_to_csv(output + "/" + "dt_predicted_labels.csv", predicted_test_labels)

        test_score, test_confusion_matrix = score_submission(actual_test_labels, predicted_test_labels)
        print_confusion_matrix(test_confusion_matrix)
        null_score, max_score = score_defaults(actual_test_labels)
        print("Percentage of test score for Decision Tree is:", test_score / float(max_score))

        precision, recall, f1 = metrics.performance_metrics(validation_confusion_matrix)

        print("Precision for DT: ", precision)
        print("Recall for DT:", recall)
        print("F1 Score for DT:", f1)

    def get_svm(self):
        svm = SVC(kernel='linear', verbose=True, gamma="scale")
        svm.fit(self.X_train, self.Y_train)

        # Validation results
        Y_val_pred = svm.predict(self.X_validate)
        predicted_validation_labels = [LABELS[int(pred)] for pred in Y_val_pred]
        actual_validation_labels = [LABELS[int(pred)] for pred in self.Y_validate]

        validation_score, validation_confusion_matrix = score_submission(actual_validation_labels,
                                                                         predicted_validation_labels)
        print_confusion_matrix(validation_confusion_matrix)

        null_score, max_score = score_defaults(actual_validation_labels)
        print("Percentage of validation score for Support Vector Machine is:", validation_score / float(max_score))

        # Test results
        Y_test_pred = svm.predict(self.X_test)
        predicted_test_labels = [LABELS[int(pred)] for pred in Y_test_pred]
        actual_test_labels = [LABELS[int(pred)] for pred in self.Y_test]

        write_to_csv(output + "/" + "svm_actual_labels.csv", actual_test_labels)
        write_to_csv(output + "/" + "svm_predicted_labels.csv", predicted_test_labels)

        test_score, test_confusion_matrix = score_submission(actual_test_labels, predicted_test_labels)
        print_confusion_matrix(test_confusion_matrix)
        null_score, max_score = score_defaults(actual_test_labels)
        print("Percentage of test score for Support Vector Machine is:", test_score / float(max_score))

        precision, recall, f1 = metrics.performance_metrics(validation_confusion_matrix)

        print("Precision for SVM: ", precision)
        print("Recall for SVM:", recall)
        print("F1 Score for SVM:", f1)

    def get_nb(self):
        nb = MultinomialNB(alpha=0.01, class_prior=None, fit_prior=True)

        nb.fit(self.X_train, self.Y_train)

        # Validation results
        Y_val_pred = nb.predict(self.X_validate)
        predicted_validation_labels = [LABELS[int(pred)] for pred in Y_val_pred]
        actual_validation_labels = [LABELS[int(pred)] for pred in self.Y_validate]
        print(actual_validation_labels)

        validation_score, validation_confusion_matrix = score_submission(actual_validation_labels,
                                                                         predicted_validation_labels)
        print_confusion_matrix(validation_confusion_matrix)

        null_score, max_score = score_defaults(actual_validation_labels)
        print("Percentage of validation score for Naive Bayes is:", validation_score / float(max_score))

        # Test results
        Y_test_pred = nb.predict(self.X_test)
        predicted_test_labels = [LABELS[int(pred)] for pred in Y_test_pred]
        actual_test_labels = [LABELS[int(pred)] for pred in self.Y_test]

        write_to_csv(output + "/" + "nb_actual_labels.csv", actual_test_labels)
        write_to_csv(output + "/" + "nb_predicted_labels.csv", predicted_test_labels)

        test_score, test_confusion_matrix = score_submission(actual_test_labels, predicted_test_labels)
        print_confusion_matrix(test_confusion_matrix)
        null_score, max_score = score_defaults(actual_test_labels)
        print("Percentage of test score for Naive Bayes is:", test_score / float(max_score))

        precision, recall, f1 = metrics.performance_metrics(validation_confusion_matrix)

        print("Precision for NB: ", precision)
        print("Recall for NB:", recall)
        print("F1 Score for NB:", f1)

    def get_rf(self):
        rf = RandomForestClassifier(n_estimators=50, random_state=66, verbose=True)

        rf.fit(self.X_train, self.Y_train)

        # Validation results
        Y_val_pred = rf.predict(self.X_validate)
        predicted_validation_labels = [LABELS[int(pred)] for pred in Y_val_pred]
        actual_validation_labels = [LABELS[int(pred)] for pred in self.Y_validate]
        print(actual_validation_labels)

        validation_score, validation_confusion_matrix = score_submission(actual_validation_labels,
                                                                         predicted_validation_labels)
        print_confusion_matrix(validation_confusion_matrix)

        null_score, max_score = score_defaults(actual_validation_labels)
        print("Percentage of validation score for Random Forest Classifier is:", validation_score / float(max_score))

        # Test results
        Y_test_pred = rf.predict(self.X_test)
        predicted_test_labels = [LABELS[int(pred)] for pred in Y_test_pred]
        actual_test_labels = [LABELS[int(pred)] for pred in self.Y_test]

        write_to_csv(output + "/" + "rf_actual_labels.csv", actual_test_labels)
        write_to_csv(output + "/" + "rf_predicted_labels.csv", predicted_test_labels)

        test_score, test_confusion_matrix = score_submission(actual_test_labels, predicted_test_labels)
        print_confusion_matrix(test_confusion_matrix)
        null_score, max_score = score_defaults(actual_test_labels)
        print("Percentage of test score for Random Forest Classifier is:", test_score / float(max_score))

        precision, recall, f1 = metrics.performance_metrics(validation_confusion_matrix)

        print("Precision for RF: ", precision)
        print("Recall for RF:", recall)
        print("F1 Score for RF:", f1)
