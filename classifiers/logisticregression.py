import os
from sklearn import metrics
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
import time
from joblib import dump, load

from baseClassifier import ClassifierBase

# Abstract class feature
class LogisticregressionClassifier (ClassifierBase):

    def __init__(self):
        super(LogisticregressionClassifier, self).__init__()
        self.clf = LogisticRegression(C=1e5)

    def train(self, X_train, y_train):
        self.clf.fit(X_train, y_train)

    def predict(self, X_test, y_test = None):
        y_pred_proba = self.clf.predict_proba(X_test)[:, 1]
        if not(y_test is None):
            print("Score: ", self.clf.score(X_test, y_test))
            y_pred = self.clf.predict(X_test)
            print("F1 score is: {}".format(f1_score(y_test, y_pred)))
            print("AUC Score is: {}".format(roc_auc_score(y_test, y_pred_proba)))
        return y_pred_proba

    def save(self, path):
        dump(self.clf, os.path.join(path,'clf.joblib'))

    def load(self, path):
        self.clf = load(os.path.join(path,'clf.joblib')) 

