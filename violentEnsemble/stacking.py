#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/8/7 11:32
# @Author  : MiaFeng
# @Contact : skaudrey@163.com
# @Site    :
# @File    : stacking.py
# @Software: PyCharm
__author__ = 'MiaFeng'


from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import StratifiedKFold
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.datasets.samples_generator import make_blobs
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier

if __name__=='__main__':

    ''' Create training set'''
    data, target = make_blobs(n_samples=50000, centers=2, random_state=0,cluster_std=0.6)

    '''Define the single model for ensemble learning'''
    clfs = [
        RandomForestClassifier(n_estimators=5, n_jobs=-1, criterion='gini'),
        RandomForestClassifier(n_estimators=5,n_jobs=-1,criterion='entropy'),
        ExtraTreesClassifier(n_estimators=5,n_jobs=-1,criterion='gini'),
        ExtraTreesClassifier(n_estimators=5,n_jobs=-1,criterion='entropy'),
        GradientBoostingClassifier(learning_rate=0.05,subsample=0.5,max_depth=6,n_estimators=10)
    ]

    '''split dataset as training set and test set'''
    X, X_predict,y,y_predict = train_test_split(data,target,test_size=0.33,random_state=2017)

    dataset_blend_train = np.zeros((X.shape[0],len(clfs)))
    dataset_blend_test = np.zeros((X_predict.shape[0],len(clfs)))

    '''five fold stacking'''
    n_folds = 5
    skf = list(StratifiedKFold(y,n_folds))
    for j,clf in enumerate(clfs):
        '''Train each model'''
        dataset_blend_test_j = np.zeros((X_predict.shape[0],len(skf)))

        for i, (train,test) in enumerate(skf):
            # i -- instance index
            # j -- sub model index
            '''Predict the ith fold, and train by the other folds'''

            X_train, y_train, X_test, y_test = X[train],y[train],X[test],y[test]
            clf.fit(X_train,y_train)
            y_submission = clf.predict_proba(X_test)[:,1]
            dataset_blend_train[test,j] = y_submission
            dataset_blend_test_j[:,i] = clf.predict(X_predict)
        '''Test set will take the average of all predictions as new feature'''
        dataset_blend_test[:,j] = dataset_blend_test_j.mean(1)
        print "sub model %d's val auc Score: %f" % (j,roc_auc_score(y_predict,dataset_blend_test[:,j]))


    clf = GradientBoostingClassifier(learning_rate=0.02,subsample=0.5,max_depth=6,n_estimators=30)
    clf.fit(dataset_blend_train,y)
    y_submission = clf.predict_proba(dataset_blend_test)[:,1]

    print 'Linear stretch of predictions to [0,1]'
    y_submission = (y_submission - y_submission.min())/(y_submission.max() - y_submission.min())
    print "stacking result\nval auc score: %f" % (roc_auc_score(y_predict,y_submission))