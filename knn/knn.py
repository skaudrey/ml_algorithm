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
import operator
from sklearn.cross_validation import train_test_split

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

    def euclideanDistance(self,x_test):
        '''
        :param x_test: Instance one with shape (#case_size,#features)
        :return:
        '''
        # x_test = np.cox_test * self.case_size
        return np.sqrt(np.sum((self.X - x_test)**2,axis=1))

    def manhattanDistance(self,x_test):
        # x_test = x_test * self.case_size
        return np.sum(np.abs(self.X - x_test),axis=1)


    def getNeighbors(self,x_test):
        distances = []
        length = self.case_size - 1
        for idx in range(self.case_size):
            dist = []
            if self.distanceType == 'Euclidean':
                dist = self.euclideanDistance(x_test)
            elif self.distanceType == 'Manhattan':
                dist = self.manhattanDistance(x_test)
            distances.append((self.X[idx],dist))
        distances.sort(key=operator.itemgetter(1))  # sort by the distance , which is the 2nd dimension in data
        neighbors = []
        for x in range(self.k):
            neighbors.append(distances[x][0])

        return neighbors

    def predict(self,neighbors):
        classVotes = {}
        for x in range(len(neighbors)):
            response = self.y[x]
            if response in classVotes:
                classVotes[response] += 1
            else:
                classVotes[response] = 1
        sortedVotes = sorted(classVotes.items(),key=operator.itemgetter(1))
        return sortedVotes[0][0]

    def getAccuracy(self,y_test,predictions):
        correct = 0
        for idy in range(len(y_test)):
            if y_test[idy] == predictions[idy]:
                correct += 1
        return (correct/float(len(y_test))) * 100.0


def loadData():
    # df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)
    # df.to_csv('flowers.csv')
    df = pd.read_csv('flowers.csv')
    df.tail()
    y = df.iloc[0:100,5].values
    y_df = pd.DataFrame(y)

    # y = np.where(y =='Iris-setosa', -1 ,1) #-1代表山鸢尾，1代表变色鸢尾，将Iris-setosa类标变为-1，其余变为1
    y = y.reshape((y.shape[0],1))
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

    X,y = loadData()

    # shuffle, manually
    training_data = np.hstack((X,y))
    np.random.shuffle(training_data)

    train_size = X.shape[0]/10*7
    test_size = X.shape[0]-train_size

    X_train,y_train = training_data[0:train_size,:-1],training_data[0:train_size,-1]
    y_train = y_train.reshape((y_train.shape[0],1))
    X_test,y_test = training_data[train_size:,:-1],training_data[train_size:,-1]
    y_test = y_test.reshape((y_test.shape[0], 1))


    print('Training set: ' + repr(len(X_train)))    # make the array printed as string
    print('Test set: '+repr(len(X_test)))           # make the array printed as string

    k = 3
    predictions = []

    clf = KNN(X=X_train,y=y_train,k=k,DistanceType='Euclidean')

    for idx in range(len(X_test)):
        neighbors = clf.getNeighbors(X_test[idx,:])
        result = clf.predict(neighbors)
        predictions.append(result)
        print('&gt; predicted = '+ repr(result)+', actual = '+ repr(y_test[idx]))

    accuracy = clf.getAccuracy(y_test,predictions)

