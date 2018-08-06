# encoding: utf-8
#!/usr/bin/env python
'''
@Author: Mia
@Contact: fengmiao@meituan.com
@Software: PyCharm
@Site    : 
@Time    : 2018/8/6 下午2:45
@File    : CModel.py
@Theme   :
'''

import numpy as np
from sklearn.model_selection import KFold
import xgboost as xgb
from async.card_fraud.extractData.CardFeatureList import MODEL_FEATURE_COLUMNS
from sklearn import cross_validation, metrics
from sklearn.grid_search import GridSearchCV
from util.plot_util import plotImg
import xgboost as xgb
from xgboost.sklearn import XGBClassifier

import sklearn
from sklearn.datasets import load_iris

class CModel(object):

    def __init__(self,
                 clf,
                 X_test,
                 y_test,
                 X_train,
                 y_train):
        self.clf = clf
        self._X_test = X_test
        self._y_test = y_test
        self._X_train = X_train
        self._y_train = y_train

    def train(self, x_train, y_train, x_val, y_val):
        pass

    def predict(self, model, x_test):
        pass

    def save_model(self, file_name):
        pass

    def load_model(self, file_name, n_folds=5):
        pass

    def evalFeaImportance(self, feat_imp):
        plotImg(feat_imp, ylabel='Feature Importance Score', title='Feature Importance', kind='bar')



    def modelFit(self, clf, trainSet, testSet,
                      useTrainCV=True,
                      cv_folds=5,
                      early_stopping_rounds=50):
        '''
        This function will do the following:

        1. fit the model
        2. determine training accuracy
        3. determine training AUC
        4. determine testing AUC
        5. update n_estimators with cv function of xgboost package
        6. plot Feature Importance
        :param clf:
        :param trainSet:
        :param testSet:
        :param useTrainCV:
        :param cv_folds:
        :param early_stopping_rounds:
        :return:
        '''

        if useTrainCV:
            xgb_param = clf.get_xgb_params()
            xgtrain = xgb.DMatrix(self._X_train.values, label=self._y_train.values)
            xgtest = xgb.DMatrix(self._X_test.values)
            cvresult = xgb.cv(xgb_param,
                              xgtrain,
                              num_boost_round=clf.get_params()['n_estimators'],
                              nfold=cv_folds,
                              metrics='auc',
                              early_stopping_rounds=early_stopping_rounds,
                              show_progress=False)
            clf.set_params(n_estimators=cvresult.shape[0])

        # Fit the classifier on the data
        clf.fit(self._X_train, self._y_train, eval_metric='auc')

        # Predict training set:
        trainSet_predictions = clf.predict(self._X_train)
        trainSet_predprob = clf.predict_proba(self._X_train)[:, 1]

        # Print model report:
        print "\nModel Report"
        print "Accuracy : %.4g" % metrics.accuracy_score(self._y_train.values, trainSet_predictions)
        print "AUC Score (Train): %f" % metrics.roc_auc_score(self._y_train, trainSet_predprob)

        #     Predict on testing data:
        testSet['predprob'] = clf.predict_proba(self._X_train)[:, 1]
        # results = test_results.merge(testSet[['ID', 'predprob']], on=self._mergeKeys)
        # print 'AUC Score (Test): %f' % metrics.roc_auc_score(results[targetKey], results['predprob'])

        feat_imp = pd.Series(clf.booster().get_fscore()).sort_values(ascending=False)

        self.evalFeaImportance(feat_imp)


    def hypTuningGSearch(self,clf,paramDict):
        '''
        Fine tuning the hyperparameters of clf, such as XGBClassifier
        :param paramDict:
        :return:
        '''
        gsearch = GridSearchCV(
            estimator=clf(
                learning_rate=0.1,
                n_estimators=140,
                max_depth=5,
                min_child_weight=1, gamma=0, subsample=0.8,
                colsample_bytree=0.8,
                objective='binary:logistic',
                nthread=4,
                scale_pos_weight=1,
                seed=27),
            param_grid=paramDict,
            scoring='roc_auc',
            n_jobs=4,
            iid=False,
            cv=5)
        gsearch.fit(self._X_train[self._feaKeys], self._X_train[self._target])

        print 'Fine tuning hyper-parameters\n'+paramDict+'\n'+80*'-'

        print gsearch.grid_scores_, gsearch.best_params_, gsearch.best_score_

    def hypFineTuning(self,df):
        param_test1 = {
            'max_depth': range(3, 10, 2),
            'min_child_weight': range(1, 6, 2)
        }

        param_test2a = {
            'max_depth': [4, 5, 6],
            'min_child_weight': [4, 5, 6]
        }

        param_test2b = {
            'min_child_weight': [6, 8, 10, 12]
        }

        param_test3 = {
            'gamma': [i / 10.0 for i in range(0, 5)]
        }

        param_test4 = {
            'subsample': [i / 10.0 for i in range(6, 10)],
            'colsample_bytree': [i / 10.0 for i in range(6, 10)]
        }

        param_test5 = {
            'subsample': [i / 100.0 for i in range(75, 90, 5)],
            'colsample_bytree': [i / 100.0 for i in range(75, 90, 5)]
        }

        param_test6 = {
            'reg_alpha': [1e-5, 1e-2, 0.1, 1, 100]
        }

        param_test7 = {
            'reg_alpha': [0, 0.001, 0.005, 0.01, 0.05]
        }



        pass

if __name__=='__main__':
    X,y = load_iris().data
    print X.shape