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


class CModel(object):

    def __init__(self, X_test, y_test, X_train, y_train, feaKeys, target=['label'], feaCols=MODEL_FEATURE_COLUMNS):
        self._X_test = X_test
        self._y_test = y_test
        self._X_train = X_train
        self._y_train = y_train
        self._target = target
        self._feaCols = feaCols
        self._feaKeys = feaKeys

    def train(self, x_train, y_train, x_val, y_val):
        pass

    def predict(self, model, x_test):
        pass

    def get_oof(self, x_train, y_train, x_test, n_folds=5):
        num_train, num_test = x_train.shape[0], x_test.shape[0]
        oof_train = np.zeros((num_train,))
        oof_test = np.zeros((num_test,))

        full_oof_test = np.zeros((num_test, n_folds))
        kf = KFold(n_splits=n_folds)
        aucs = []
        self.models = []
        for i, (train_index, val_index) in enumerate(kf.split(x_train)):
            print('{0} fold, train {1}, val {2}'.format(i, len(train_index), len(val_index)))

            x_partial_train, y_partial_train = x_train[train_index], y_train[train_index]
            x_val_train, y_val_train = x_train[val_index], y_train[val_index]
            model, auc = self.train(x_partial_train, y_partial_train, x_val_train, y_val_train)

            aucs.append(auc)
            self.models.append(model)
            oof_train[val_index] = self.predict(model, x_val_train)
            full_oof_test[:, i] = self.predict(model, x_test)

        oof_test = np.mean(full_oof_test, axis=1)
        print('all aucs {0}, average {1}'.format(aucs, np.mean(aucs)))
        return oof_train, oof_test

    def get_test_oof(self, model_path, x_test, n_folds=5):
        self.load_model(model_path, n_folds)
        full_oof_test = np.zeros((len(x_test), n_folds))
        for idx, model in enumerate(self.models):
            full_oof_test[:, idx] = self.predict(model, x_test)
        oof_test = np.mean(full_oof_test, axis=1)
        return oof_test

    def save_model(self, file_name):
        pass

    def load_model(self, file_name, n_folds=5):
        pass

    def modelFit(self, alg, trainSet, testSet, predictors, useTrainCV=True, cv_folds=5, early_stopping_rounds=50):

        if useTrainCV:
            xgb_param = alg.get_xgb_params()
            xgtrain = xgb.DMatrix(trainSet[predictors].values, label=trainSet[self._target].values)
            xgtest = xgb.DMatrix(testSet[predictors].values)
            cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,
                              metrics='auc', early_stopping_rounds=early_stopping_rounds, show_progress=False)
            alg.set_params(n_estimators=cvresult.shape[0])

        # Fit the algorithm on the data
        alg.fit(trainSet[predictors], trainSet['Disbursed'], eval_metric='auc')

        # Predict training set:
        trainSet_predictions = alg.predict(trainSet[predictors])
        trainSet_predprob = alg.predict_proba(trainSet[predictors])[:, 1]

        # Print model report:
        print "\nModel Report"
        print "Accuracy : %.4g" % metrics.accuracy_score(trainSet['Disbursed'].values, trainSet_predictions)
        print "AUC Score (Train): %f" % metrics.roc_auc_score(trainSet['Disbursed'], trainSet_predprob)

        #     Predict on testing data:
        testSet['predprob'] = alg.predict_proba(testSet[predictors])[:, 1]
        results = test_results.merge(testSet[['ID', 'predprob']], on=self._feaKeys)
        print 'AUC Score (Test): %f' % metrics.roc_auc_score(results['Disbursed'], results['predprob'])

        feat_imp = pd.Series(alg.booster().get_fscore()).sort_values(ascending=False)
        feat_imp.plot(kind='bar', title='Feature Importances')
        plt.ylabel('Feature Importance Score')
