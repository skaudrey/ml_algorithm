#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/8/13 14:30
# @Author  : MiaFeng
# @Contact : skaudrey@163.com
# @Site    :
# @File    : selectFeaSubset.py
# @Software: PyCharm
__author__ = 'MiaFeng'

'''
The package mlxtend for selecting features in wrapper, can be searched in
https://rasbt.github.io/mlxtend/user_guide/feature_selection/SequentialFeatureSelector/#sequential-feature-selector
'''


from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split

class CSelectFeaSubset(object):
    def __init__(self):
        pass

    def getFea(self,clfs,K,metrics,X,y):
        sbfs = SFS(clfs,
                   k_features=K,
                   forward=False,
                   floating=True,
                   scoring=metrics,
                   cv=5,
                   n_jobs=-1)
        sbfs = sbfs.fit(X, y)

        print('\nSequential Backward Floating Selection (k=%d):' % K)
        print(sbfs.k_feature_idx_)
        print('CV Score:')
        print(sbfs.k_score_)

        return sbfs.k_feature_idx_,sbfs.k_score_

if __name__=="__main__":
    X, y = load_diabetes()

    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.33,random_state=2018)

    clf = RandomForestClassifier()

    tools = CSelectFeaSubset()

    idxs = []
    union_idx = set()
    inter_idx = set()

    for i in xrange(10):
        idx,score = tools.getFea(clf,K=11,metrics='recall',X=X_train,y=y_train)
        if score>0.5:
            idxs.append(idx)
            if i==0:
                inter_idx.add(idx)
            union_idx.union(idx)
            inter_idx.intersection(idx)


    print "Feature intersection:"
    print inter_idx
    print "Feature union:"
    print union_idx

