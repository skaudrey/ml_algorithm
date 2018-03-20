#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/3/5 17:41
# @Author  : MiaFeng
# @Site    : 
# @File    : perceptron.py
# @Software: PyCharm
__author__ = 'MiaFeng'

# Reference: Sebastian Raschka. Python Machine Learning,2017

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from util.path_util import currentPath
from util.plot_util import plot_decision_regions,plot_colors,plot_markers
import pandas as pd

class Perceptron:
    def __init__(self,X,y,eta = 0.01,n_iter = 10):
        self.X = X
        self.y = y
        self.eta = eta  # learning rate
        self.n_iter = n_iter    # passes over the training dataset
        self.case_size,self.dim = X.shape
        self.w__ = np.zeros(1+self.dim) # weights(including b,which is defined at w__[0]) after fitting
        self.errors__ = []  # number of misclassifications in every epoch


    def net_input(self,x):
        '''
        Calculate net input of one sample
        :return: net input
        '''
        return np.dot(x,self.w__[1:])+self.w__[0]

    def predict(self,x):
        '''
        Predict the label for a sample x
        :param x:
        :return: 1 for positive, -1 for negative
        '''
        return np.where(self.net_input(x) >= 0.0,1,-1)

    def fit(self):
        for _ in range(self.n_iter):    # the index here won't be used, thus can be ignored by _
            errors = 0
            for xi, target in zip(self.X,self.y):
                update = self.eta * (target - self.predict(xi)) # update != 0.0 only when prediction is wrong
                # another way for estimating whether this sample is classified correctly: target - self.predict(xi)
                self.w__[1:] += update * xi
                self.w__[0] += update
                errors += int(update != 0.0)
            self.errors__.append(errors)

    def plot_errors(self):
        '''
        Plot the #(errors) per iteration. #errors mean the number of the samples which are predicted incorrectly
        :return:
        '''

        plt.figure()
        plt.plot(range(1,len(self.errors__)+1),self.errors__,marker = 'o')
        plt.savefig(basepath+'errors.png')
        plt.show()

def loadData():
    # df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)
    # df.to_csv('flowers.csv')
    df = pd.read_csv('flowers.csv')
    df.tail()
    y = df.iloc[0:100,5].values
    y_df = pd.DataFrame(y)

    y = np.where(y =='Iris-setosa', -1 ,1) #-1代表山鸢尾，1代表变色鸢尾，将Iris-setosa类标变为-1，其余变为1
    X = df.iloc[0:100,[1,3]].values # features: petal length, sepal length萼片长度，花瓣长度
    plt.figure(1)
    plt.scatter(X[:50, 0], X[:50,1],color=color_plot[0],marker='o',label='setosa')  #山鸢尾
    plt.scatter(X[50:, 0], X[50:, 1], color=color_plot[1], marker='o', label='versicolor')  # 变色鸢尾
    plt.xlabel('petal length')
    plt.ylabel('sepal length')
    plt.legend(loc='upper left')
    plt.savefig(basepath+'raw.png')
    plt.show()
    return X,y


if __name__ == '__main__':
    sns.set()

    # set path and color style
    basepath = currentPath()+'/file/percepTex/fig/'
    color_plot = plot_colors()

    # load data
    X,y = loadData()

    ppn = Perceptron(X,y)
    ppn.fit()
    ppn.plot_errors()
    plot_decision_regions(X,y,classifier=ppn,basepath=basepath,fileName='decision.png')