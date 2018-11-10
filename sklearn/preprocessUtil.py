#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/7/31 16:45
# @Author  : MiaFeng
# @Contact : skaudrey@163.com
# @Site    :
# @File    : preprocessUtil.py
# @Software: PyCharm
__author__ = 'MiaFeng'

from sklearn.preprocessing.imputation import Imputer
from sklearn.feature_selection import f_regression
import pandas as pd
from sklearn.cluster import KMeans
# import numpy as np
from mpl_toolkits.mplot3d import Axes3D



FILL_DICT = {
    'miss1':0,
    'miss2':1
}

def fillMissing(df):
    df = df.fillna(FILL_DICT)

    # df = fill(df,'bankcard_fraud_charge_hist_features.androidappcnt',strategy='most_frequent')
    # filling androidappcnt with mode
    embarked_mode = df['androidappcnt'].mode()[0]
    df['androidappcnt'] = df['androidappcnt'].fillna(embarked_mode)

    return df

def fill(data,fill_loc,strategy='mean',missing_values='NaN',aixs=0):
    '''
    Filling missing data of DataFrame
    :param data: Raw data
    :param col_loc: The col name list of DataFrame you wanna to fill.
    :param strategy: Filling strategy. 'mean','median','most_frequent' can be chosen.
    :param missing_values: The missing value in your data. 'NaN' etc. can be chosen.
    :param aixs: 'axis=0' denotes filling the columns, 'axis=1' denotes filling the rows.
    :return: The dataframe after filling.
    '''
    imputer = Imputer(missing_values=missing_values, strategy=strategy, axis=0)
    imputer.fit(data.loc[:, fill_loc])
    x_new = imputer.transform(data.loc[:, fill_loc])
    return x_new

def autoUnivaraiateSelectFeature(X, y, score,preserve_strategy,preserve):
   '''
   Scoring the features by a specific strategy
    https://www.jianshu.com/p/b3056d10a20f

    There are four strategies in the module sklearn.feature_selection:

    (1) Removing features with low variance
        VarianceThreshold
        >>> X = [[0, 0, 1], [0, 1, 0], [1, 0, 0], [0, 1, 1], [0, 1, 0], [0, 1, 1]]
        >>> sel = VarianceThreshold(threshold=(.8 * (1 - .8)))
        Remove the feature that is 0 or 1 with 80%.
        >>> sel.fit_transform(X)
    (2) Univariate feature selection
        One can preserve the kth best or n% percent features estimated by the scores. The scores can be set as:
            Classification:  f_regression, mutual_info_regression
            Regression:      chi2, f_classif (F score in ANOVA), mutual_info_classif

        >>> from sklearn.feature_selection import SelectKBest
            # SelectPercentile is also available
        >>> from sklearn.feature_selection import chi2
            # Estimating data by chi2.
        >>> from sklearn.datasets import load_iris
        >>> iris = load_iris()
        >>> X, y = iris.data, iris.target
        >>> X.shape
        >>> X_new = SelectKBest(chi2, k=2).fit_transform(X, y)
            # Estimated by chi2, and preserve the 2 best.
        >>> X_new.shape

    (3) Recursive feature elimination
        Cross validation and reducing features until to the number of features you want.

    (4) Feature selection using SelectFromModel
        It is used with other models: L1 regularization, decision tree, or random sparse model

        4.1 L1 regularization: select the features with non-zero coefficients
            >>> from sklearn.svm import LinearSVC
            >>> from sklearn.datasets import load_iris
            >>> from sklearn.feature_selection import SelectFromModel
            >>> iris = load_iris()
            >>> X, y = iris.data, iris.target
            >>> X.shape
            >>> lsvc = LinearSVC(C=0.01, penalty="l1", dual=False).fit(X, y)
            >>> model = SelectFromModel(lsvc, prefit=True)
            >>> X_new = model.transform(X)
            >>> X_new.shape
        4.2 Random sparse model: select features by estimating features' contribution with randomly sampled data.
           Such as using stability selection in sklearn.linear_mode.

        4.3 Decision tree
            >>> from sklearn.ensemble import ExtraTreesClassifier
            >>> from sklearn.datasets import load_iris
            >>> from sklearn.feature_selection import SelectFromModel
            >>> iris = load_iris()
            >>> X, y = iris.data, iris.target
            >>> X.shape
            >>> clf = ExtraTreesClassifier()
            >>> model = SelectFromModel(clf, prefit=True)
            >>> X_new = model.transform(X)
            >>> X_new.shape
   :param X:
   :param y:
   :param score: See (2) above.
   :param preserve_strategy: 'kbest' for selecting the k top best features, 'percent' for preserving n% features.
   :param preserve: The preserve parameters, integer for kbest, float for percentile
   :return: Transformed features
   '''
   F = f_regression(X, y)

   print(len(F))

   print(F)

   from sklearn.feature_selection import SelectKBest
   return SelectKBest(score, k=preserve).fit_transform(X, y)

def countNull(df):
    nullCount = df.apply(lambda x: sum(x.isnull()))

    nullRate = nullCount/len(df)

    # saveDF2CSV(dataFrame=nullRate,filename='card_fraud_fea_null_rate')

    return nullCount

def countCategories(df):
    '''
    Return the name list which just has one specific columns or null columns
    :param df:
    :return:
    '''
    dropNameList = []
    var = df.columns.values
    for v in var:
        print '\nFrequency count for variable %s' % v
        count = df[v].value_counts()
        print count
        if(len(count)<2 or len(count) == None):
            dropNameList.append(v)

    return dropNameList



def getDropColNames(df,rate):
    '''
    Return the col name list that the null rate in this column is bigger than rate, and the columns that has only one value
    :param rate:
    :return: Column name string list. Mind there is single quotes between each string
    '''
    nullRate = df.apply(lambda x: sum(x.isnull())) / len(df) # A series has two features: values, index

    nullCol = nullRate[nullRate > rate].index

    uniqueFea = countCategories(df) # drop the features that just has one value

    combine = list(set(uniqueFea).union(set(nullCol)))

    # return map(str,",".join(combine))    # combine to string list
    return combine

def dropNullRowByCols(df, colNameStrList):
    '''
    Drop the instances that has null missing in some specific columns
    :param df:
    :param colNameStrList:
    :return:
    '''
    new_df = df.dropna(subset=colNameStrList) # Attention!! Have to return the new df
    print new_df.shape
    return new_df

def dropFeatures(df,colNameStrList):
    new_df = df.drop(colNameStrList,axis=1)  # Attention!! Have to return the new df
    print new_df.shape
    return new_df

def timeSeriesRoll(df,interval = '5T'):
    '''
    :param df:
    :param interval: 'T' for minutes, 'M' for months, 'D' for days
    :return:
    '''

    df_time = df[['userid', 'currenttime', 'outmoney', 'orderid']]

    result = pd.DataFrame()

    for name, group in df_time.groupby(['userid']):
        # print name

        cout_group = group[['currenttime', 'outmoney']]

        # # Caution: The resample function only works in the data indexed with datetimedelta or sth. like it.

        cout_group.set_index('currenttime', inplace=True)

        tmpResult = cout_group.rolling(interval).agg(['count', 'sum', 'max', 'mean'])


        concatTmp = pd.DataFrame({
            'factor_user_outmoney_cnt_%s'%interval: tmpResult['outmoney']['count'].values,
            'factor_user_outmoney_sum_%s'%interval: tmpResult['outmoney']['sum'].values,
            'factor_user_outmoney_max_%s'%interval: tmpResult['outmoney']['max'].values,
            'factor_user_outmoney_min_%s'%interval: tmpResult['outmoney']['mean'].values
        })

        result = pd.concat([result, concatTmp], ignore_index=True)

        # for idx in xrange(len(tmpMonth)):
        #     print "userid [%s] -- month [%d] -- total charge money -- %.2f" % (
        #         name, tmpMonth[idx], result['outmoney']['sum'][idx])

    df = pd.concat([df,result],axis=1)

    return df

def cocurrenceMatrix(df,onehotMap,target):
    '''
    Counting the cocurrence matrix of DataFrame
    :param df:  Raw DataFrame
    :param onehotMap:The mapping matrix for discrete variable
    :param target: The interesting feature names, which is the columns of DataFrame df
    :return:
    '''
    df_cnt = df[target]

    matrixItm = {}

    for iA, iB in zip(df_cnt[target[0]], df_cnt[target[1]]):
        # print iA, iB
        keys = {}
        key1 = '%s_%s'%(iA,iB)
        key2 = '%s_%s' % (iB, iA)

        keys[key1] = matrixItm.get(key1)==None
        keys[key2] = matrixItm.get(key2) == None
        tmp = 0
        if(keys[key1] or keys[key2]):
            if(keys[key1]):
                tmp = keys[key1] + 1
                matrixItm[key1] = tmp
            else:
                tmp = keys[key2] + 1
                matrixItm[key2] = tmp
        else:
            matrixItm[key1] = 1

    cnt = pd.DataFrame(data=matrixItm.values(), index=matrixItm.keys(), columns=['cnt'])

    return cnt


def clusterDf(df,K,target,extra=None):
    df_clust = (df[[target,extra,'label']].values)
    print df_clust.shape
    kmeans = KMeans(n_clusters=K, random_state=0).fit(df_clust)

    df_result = []

    for k in xrange(K):
        df_result.append([[],[],[]])

    for idx,i in enumerate(kmeans.labels_):
        df_result[i][0].append(df_clust[idx][0])
        df_result[i][1].append(df_clust[idx][1])
        if df_clust[idx][2]!=None:
            df_result[i][2].append(df_clust[idx][2])

    # show result and save
    fig = plt.figure(figsize=(8, 8))
    Color = 'rbgyckm'

    if df_clust.shape[0]>2:
        ax = fig.add_subplot(111, projection='3d')
        # plot clusters' centroids
        ax.scatter(kmeans.cluster_centers_[:, 0],
                   kmeans.cluster_centers_[:, 1],
                   kmeans.cluster_centers_[:, 2],
                   color=Color[0])
        # plot clusters, z denotes the positive and negative labels
        for i in range(K):
            mydata = df_result[i]
            print len(mydata)
            ax.scatter(mydata[0], mydata[1],mydata[2], color=Color[i + 1])

        ax.set_xlabel(target)
        ax.set_ylabel(extra)
        ax.set_zlabel('pos-neg label')

    else:
        ax = fig.add_subplot(111)
        # plot clusters' centroids
        ax.scatter(kmeans.cluster_centers_[:, 0],
                   kmeans.cluster_centers_[:, 1],
                   color=Color[0])
        # plot clusters, y denotes the positive and negative labels
        for i in range(K):
            mydata = df_result[i]
            print len(mydata)
            ax.scatter(mydata[0], mydata[1], color=Color[i + 1])
        ax.xlabel(target)
        ax.ylabel('pos-neg label')

    plt.title('%s-%s' % (target, extra))
    plt.savefig('%s%s-%s.png', dpi=500)
    plt.show()

    return kmeans.labels_,kmeans