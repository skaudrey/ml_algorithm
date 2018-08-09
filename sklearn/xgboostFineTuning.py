#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/7/31 21:27
# @Author  : MiaFeng
# @Contact : skaudrey@163.com
# @Site    :
# @File    : xgboostFineTuning.py
# @Software: PyCharm
__author__ = 'MiaFeng'

from sklearn.datasets import load_iris
from sklearn.cross_validation import train_test_split
import xgboost as xgb
from sklearn.metrics import accuracy_score

def loadData():
    X,y = load_iris().data, load_iris().target

    y = y.reshape((-1,1))

    y[y==2]=1 # change to binary classification

    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.33,random_state=2018)

    return X_train,X_test,y_train,y_test

if __name__=='__main__':
    X_train, X_test, y_train, y_test = loadData()

    # XGBoost自定义了一个数据矩阵类DMatrix，优化了存储和运算速度
    dtrain = xgb.DMatrix(data=X_train, label=y_train)
    dtest = xgb.DMatrix(data=X_test, label=y_test)
    # print dtrain.num_col(),dtrain.num_row(),dtest.num_row()

    # specify parameters via map. The base function softmax requires setting num_class
    param = {'max_depth': 2, 'eta': 1, 'silent': 0, 'objective': 'multi:softmax', 'num_class': 3}
    # 设置boosting迭代计算次数
    num_round = 2
    # Train with preloaded training set dtrain.
    bst = xgb.train(param, dtrain, num_round)

    # !!!!!!
    # XGBoost预测的输出是概率。这里分类是一个二类分类问题，输出值是样本为第一类的概率。（一般按从小到大的顺序）
    # 即样本label为0的概率
    # 我们需要将概率值转换为0或1。
    # !!!!!!

    y_test_preds = bst.predict(dtest)
    train_predictions = [round(value) for value in train_preds] # 四舍五入，0.5为界
    y_train = dtrain.get_label()  # 值为输入数据的第一行
    train_accuracy = accuracy_score(y_train, train_predictions)
    print "Train Accuary: %.2f%%" % (train_accuracy * 100.0)
