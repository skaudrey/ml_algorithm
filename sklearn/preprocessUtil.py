# encoding: utf-8
#!/usr/bin/env python
'''
@Author: Mia
@Contact: fengmiao@163.com
@Software: PyCharm
@Site    : 
@Time    : 2018/7/31 下午4:45
@File    : preprocessUtil.py
@Theme   :
'''

from sklearn.preprocessing.imputation import Imputer
from sklearn.feature_selection import f_regression


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