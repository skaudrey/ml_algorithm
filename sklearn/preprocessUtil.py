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

def fillMissing(data,fill_loc,strategy='mean',missing_values='NaN',aixs=0):
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


