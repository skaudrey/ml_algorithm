#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/8/7 14:57
# @Author  : MiaFeng
# @Contact : skaudrey@163.com
# @Site    :
# @File    : __init__.py.py
# @Software: PyCharm
__author__ = 'MiaFeng'

from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import StratifiedKFold
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.datasets.samples_generator import make_blobs


'''make dataset'''
data,target = make_blobs(n_samples=50000,centers=2,random_state=0,cluster_std=0.6)
'''Define sub models for ensemble learning'''
clfs = [
    RandomForestClassifier(n_estimators=5,n_jobs=-1,criterion='gini'),
    RandomForestClassifier(n_estimators=5,n_jobs=-1,criterion='entropy'),
    ExtraTreesClassifier(n_estimators=5,n_jobs=-1,criterion='gini'),
    ExtraTreesClassifier(n_estimators=5,n_jobs=-1,criterion='entropy'),
    GradientBoostingClassifier(learning_rate=0.05,subsample=0.5,max_depth=6,n_estimators=5)
]

'''split some data as test set'''
X,X_predict,y,y_predict = train_test_split(data,target,test_size=0.33,random_state=2017)

# '''five fold stacking'''
# n_folds = 5
# skf = list(StratifiedKFold(y,n_folds))

'''Split dataset to two part'''
X_d1,X_d2,y_d1,y_d2 = train_test_split(X,y,test_size=0.5,random_state=2017)
dataset_d1 = np.zeros((X_d2.shape[0],len(clfs)))
dataset_d2 = np.zeros((X_predict.shape[0],len(clfs)))

for j,clf in enumerate(clfs):

    clf.fit(X_d1,y_d1)

    y_submission = clf.predict_proba(X_d2)[:,1]
    dataset_d1[:,j] = y_submission
    dataset_d2[:,j] = clf.predict_proba(X_predict)[:,1]

    print("val auc score: %f")%roc_auc_score(y_predict,dataset_d2[:,j])

'''merge model'''
clf = GradientBoostingClassifier(learning_rate=0.02,subsample=0.5,max_depth=6,n_estimators=30)
clf.fit(dataset_d1,y_d2)
y_submission = clf.predict_proba(dataset_d2)[:,1]


print("Linear stretch of predictions to [0,1]")
y_submission = (y_submission - y_submission.min()) / (y_submission.max() - y_submission.min())
print("blend result")
print("val auc Score: %f" % (roc_auc_score(y_predict, y_submission)))
