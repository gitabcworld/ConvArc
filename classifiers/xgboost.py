import os
from sklearn import metrics
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import GridSearchCV
import xgboost as xgb
from xgboost import XGBClassifier
import time
from joblib import dump, load
from classifiers.baseClassifier import ClassifierBase

# Abstract class feature
class XGBoostClassifier (ClassifierBase):

    def __init__(self, useTrainCV=True, cv_folds=5, early_stopping_rounds=50):
        super(XGBoostClassifier, self).__init__()
        self.useTrainCV = useTrainCV
        self.cv_folds = cv_folds
        self.early_stopping_rounds = early_stopping_rounds
        self.clf = XGBClassifier(learning_rate=0.1, n_estimators=140, max_depth=5,
                min_child_weight=3, gamma=0.2, subsample=0.6, colsample_bytree=1.0,
                objective='binary:logistic', n_jobs=6, scale_pos_weight=1, seed=27)

    def train(self, X_train, y_train):
        if self.useTrainCV:
            print("Start Feeding Data for Cross Validation")
            xgb_param = self.clf.get_xgb_params()
            xgtrain = xgb.DMatrix(X_train, label=y_train)
            cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=self.clf.get_params()['n_estimators'], nfold=self.cv_folds,
                            early_stopping_rounds=self.early_stopping_rounds)
            self.clf.set_params(**cvresult)
            # param_test1 = {}
            # gsearch1 = GridSearchCV(estimator=XGBClassifier(learning_rate=0.1, n_estimators=140, max_depth=5,
            #                                                 min_child_weight=3, gamma=0.2, subsample=0.8,
            #                                                 colsample_bytree=1.0,
            #                                                 objective='binary:logistic', nthread=4, scale_pos_weight=1,
            #                                                 seed=27),
            #                         param_grid=param_test1,
            #                         scoring='f1',
            #                         n_jobs=4, iid=False, cv=5)
            # gsearch1.fit(X_train, y_train)
            # print(gsearch1.cv_results_, gsearch1.best_params_, gsearch1.best_score_)

        self.clf.fit(X_train, y_train, eval_metric='auc')

    def predict(self, X_test, y_test = None):

        y_pred_proba = self.clf.predict_proba(X_test)[:, 1]
        if not(y_test is None):
            print("Score: ", self.clf.score(X_test, y_test))
            y_pred = self.clf.predict(X_test)
            print("Acc : %.4g" % metrics.accuracy_score(y_test, y_pred))
            print("F1 score is: {}".format(f1_score(y_test, y_pred)))
            print("AUC Score is: {}".format(roc_auc_score(y_test, y_pred_proba)))
        return y_pred_proba

    def printFeatureImportance(self,X_train):
        feat_imp = self.clf.feature_importances_
        feat = X_train.columns.tolist()
        #res_df = pd.DataFrame({'Features': feat, 'Importance': feat_imp}).sort_values(by='Importance', ascending=False)
        #res_df.plot('Features', 'Importance', kind='bar', title='Feature Importances')
        #plt.ylabel('Feature Importance Score')
        #plt.show()
        #print(res_df)
        #print(res_df["Features"].tolist())
        print('Importance feats:',feat)

    def save(self, path):
        dump(self.clf, os.path.join(path,'clf.joblib'))

    def load(self, path):
        self.clf = load(os.path.join(path,'clf.joblib')) 

