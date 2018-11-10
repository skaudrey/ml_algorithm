#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/8/9 18:30
# @Author  : MiaFeng
# @Contact : skaudrey@163.com
# @Site    :
# @File    : pipeLR.py
# @Software: PyCharm
__author__ = 'MiaFeng'

from CModel import CModel
import xgboost as xgb
import os
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.datasets import load_iris


class CXGBCCard(CModel):
    def __init__(self, num_rounds=100, early_stopping_rounds=15, **params):
        self._num_rounds = num_rounds
        self._early_stopping_rounds = early_stopping_rounds

        self._params = {
            'objective': 'binary:logistic',
            'eta': 0.1,
            'max_depth': 3,
            'eval_metric': 'auc',
            'seed': 0,
            'silent': 0
        }

        for key, value in params.items():
            self._params[key] = value


    def fit(self, model, X_train, y_train):
        model.fit(X_train, y_train)
        return model,model.feature_importances_

    def predict(self, model, X_test,y_test):
        print('test with xgbc model')
        y_pred = model.predict(X_test)
        score = model.score(X_test,y_test)
        return y_pred,score

    def save_model(self, model, fileName=None):
        if not os.path.exists(fileName):
            os.makedirs(fileName)
        # for i, model in enumerate(self.models):
        #     model.save_model(os.path.join(file_name, str(i) + '.model'))
        model.save_model(fileName)

    def set_params(self, n_estimators):
        self._num_rounds = n_estimators

    def load_model(self, file_name=None, n_thread=5):
        self.models = []
        # for i in range(n_folds):
        #     model = xgb.Booster(model_file=os.path.join(file_name, str(i) + '.model'))
        #     self.models.append(model)

        bst = xgb.Booster({'nthread': n_thread})  # init model
        bst.load_model(model_file=os.path.join(file_name + '.model'))  # load data

    def fit(self, model, X_train, y_train):

        model.fit(X_train, y_train)

        return model,model.feature_importances_

    def plotFeaImportance(self, feaImp, feaName):
        plt.figure()
        plt.barh(xrange(len(feaName)), feaImp, align='center', alpha=0.4)
        # fea_imp.plot(kind='bar')
        plt.yticks(xrange(len(feaName)), feaName)
        plt.show()




if __name__ == '__main__':
    X,y = load_iris().data

    feaNameVals = ['d1','e2','e2']

    # X_train = df_train[MODEL_FEATURE_COLUMNS]
    # y_train = df_train['label']
    #
    # X_test = df_test[MODEL_FEATURE_COLUMNS]
    # y_test = df_test['label']
    # y_test_pos_idx = y_test[y_test == 1].index

    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.33,random_state=2018)

    model = XGBClassifier()
    clf = CXGBCCard()
    model,feaImp = clf.fit(model,X_train, y_train)
    y_pred,score = clf.predict(model,X_test, y_test)
    print 'The accuracy of xgbc on testing set is %.2f' % score

    feaName = []
    imps = []
    for idx, imp in enumerate(feaImp):
        if imp > 0:
            imps.append(imp)
            feaName.append(feaNameVals[idx])
    # fea_imp.columns = MODEL_FEATURE_COLUMNS
    print imps

    clf.plotFeaImportance(imps,feaName)

    clf.eval(y_test,y_pred,'confusion_matrix')

    clf.save_model(model,'model_groupDate')


