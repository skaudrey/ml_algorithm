#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/4/21 22:31
# @Author  : MiaFeng
# @Site    : 
# @File    : knn.py
# @Software: PyCharm
__author__ = 'MiaFeng'

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from util.path_util import currentPath
from util.plot_util import plot_decision_regions,plot_colors,plot_markers

class KNN:

    def __init__(self,X,y,k=3,DistanceType='Euclidean'):
        '''
        Initialization
        :param X: data matrix (N,M), where N is #test_cases, M is #dimension
        :param y: label matrix (N,1), where N is #test_cases
        :param k: the top k minimum distance points
        :param DistanceType: String, define the type for calculating distance, which equals to 'Euclidean' or 'Manhattan'
        '''
        self.X = X
        self.y = y
        self.case_size, self.dim = self.X.shape
        self.k = k
        self.distanceType = DistanceType

    def euclideanDistance(self,x1,x2):
        '''
        :param x1: Instance one with shape (#case_size,#features)
        :param x2: Instance one with shape (#case_size,#features)
        :return:
        '''
        return np.sqrt(np.sum((x1-x2)**2))

    def manhattanDistance(self,x1,x2):
        return np.sum(np.abs(x1-x2))

    def getNeibors(self):
        distances = []
        length = self.case_size - 1
        for x in range(self.case_size):
            dist = []
            if self.distanceType == 'Euclidean':
                dist = self.euclideanDistance(self.X,self.y)
            elif self.distanceType == 'Manhattan':
                dist = self.manhattanDistance(self.X,self.y)


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

if __name__=='__main__':
    sns.set()

    # set path and color style
    basepath = currentPath() + '/file/knnTex/fig/'
    color_plot = plot_colors()