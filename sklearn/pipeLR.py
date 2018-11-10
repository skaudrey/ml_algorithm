#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/8/6 11:56
# @Author  : MiaFeng
# @Contact : skaudrey@163.com
# @Site    :
# @File    : pipeLR.py
# @Software: PyCharm
__author__ = 'MiaFeng'

import  os

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from CModel import CModel

class CPipeLR(CModel):
    def __init__(self,penalty='l2',cv=5, **params):

        self.pipe_lr = Pipeline([
            ('sc', StandardScaler()),
            ('pca', PCA(n_components=2)),
            ('clf', LogisticRegressionCV(
                random_state=1,
                penalty=penalty,
                cv=cv))
        ])

        for key, value in params.items():
            self.params[key] = value

    def train(self, X_train, y_train):
        print("train with lr pipelined with standardScaler and PCA model")
        self.pipe_lr.fit(X_train, y_train)

        return self.pipe_lr


    def predict(self, X_test):
        print('predict with pipeline logistic model')

        return self.pipe_lr.predict(X_test)

    def save_model(self, file_name=None):
        if not os.path.exists(file_name):
            os.makedirs(file_name)
        for i, model in enumerate(self.models):
            model.save_model(os.path.join(file_name, str(i) + '.model'))

    def load_model(self, file_name=None, n_folds=5):
        self.models = []
        for i in range(n_folds):
            model = self.pipe_lr(model_file=os.path.join(file_name, str(i) + '.model'))
            self.models.append(model)

    def predict_prob(self,X_test):
        return self.pipe_lr.predict_proba(X_test)




if __name__=='__main__':
    pass