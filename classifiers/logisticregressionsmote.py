import os
from sklearn import metrics
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from imblearn.pipeline import make_pipeline as make_pipeline_imb
from imblearn.over_sampling import SMOTE
from imblearn.metrics import classification_report_imbalanced
import xgboost as xgb
from xgboost import XGBClassifier
import time
from joblib import dump, load
from baseClassifier import ClassifierBase

# Abstract class feature
class LogisticregressionClassifierSmote (ClassifierBase):

    def __init__(self):
        super(LogisticregressionClassifierSmote, self).__init__()
        self.clf = LogisticRegression(C=1e5)

    def train(self, X_train, y_train):
        self.clf.fit(X_train, y_train)
        # build model with SMOTE imblearn
        self.smote_pipeline = make_pipeline_imb(SMOTE(random_state=4), self.clf)
        self.smote_model = self.smote_pipeline.fit(X_train, y_train)

    def predict(self, X_test, y_test = None):
        y_pred_proba = self.clf.predict_proba(X_test)[:, 1]

        smote_prediction = self.smote_model.predict(X_test)
        smote_prediction_proba = self.smote_model.predict_proba(X_test)[:, 1]

        if not(y_test is None):
            print(classification_report_imbalanced(y_test, smote_prediction))
            print('SMOTE Pipeline Score {}'.format(self.smote_pipeline.score(X_test, y_test)))
            print("SMOTE AUC score: ", roc_auc_score(y_test, smote_prediction_proba))
            print("SMOTE F1 Score: ", f1_score(y_test, smote_prediction))
        return smote_prediction_proba

    def save(self, path):
        dump(self.clf, os.path.join(path,'clf.joblib'))
        dump(self.smote_model, os.path.join(path,'smote.joblib'))
        dump(self.smote_pipeline, os.path.join(path,'smote_pipeline.joblib'))

    def load(self, path):
        self.clf = load(os.path.join(path,'clf.joblib')) 
        self.smote_model = load(os.path.join(path,'smote.joblib'))
        self.smote_pipeline = load(os.path.join(path,'smote_pipeline.joblib'))


