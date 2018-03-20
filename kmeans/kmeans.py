#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/3/4 14:30
# @Author  : MiaFeng
# @Site    : 
# @File    : perceptron.py
# @Software: PyCharm
__author__ = 'MiaFeng'

import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns
import random as rd
import pandas as pd
import os


marker_data = ['or', 'ob', 'og', 'ok', '^r', '+r', 'sr', 'dr', '<r', 'pr']

marker_centroids = ['Dr', 'Db', 'Dg', 'Dk', '^b', '+b', 'sb', 'db', '<b', 'pb']

basepath = os.getcwd()+'/file/percepTex/fig/'

class KMeans:

    def __init__(self,X,k):
        '''
        Initialization
        :param X: data matrix (N,M), where N is #test_cases, M is #dimension
        :param k: the #cluster centroids
        '''
        self.X = X
        self.k = k
        self.case_size,self.dim = self.X.shape
        self.centroids = np.zeros((self.k,self.dim))
        self.labelDecision = np.zeros((self.case_size,2))
        # the first column stores the label of testcase
        # the second column stores the distance between this point and its centroids


    def euclDistance(self,x1,x2):
        return np.sqrt(np.sum((x1-x2)**2))

    def randJ(self,choosedList):
        '''
        get the index of centroids randomly
        :param i:  the index of last one
        :return: the index j of alpha2
        '''
        j = rd.sample(range(self.case_size),1)[0]
        flag = True
        while(flag):
            if j not in choosedList:
                return j
            else:
                j = rd.sample(range(self.case_size), 1)[0]

    def initCentroids(self):
        chooseList = []
        for i in range(self.k):
            index = self.randJ(chooseList)
            chooseList.append(index)
            self.centroids[i,:] = self.X[index,:]

    def train(self):
        '''
        training
        steps:
            1) initiate centroids randomly
            2) for each sample
                2.1) find the index of centroid which is closest to the sample
                2.2) update the sample's centroids
            3) for each centroids
                update centroids
        :return:
        '''

        # 1) initiate centroids randomly
        self.initCentroids()
        clusterChanged = True

        iters = 0

        while clusterChanged:
            clusterChanged = False
            iters += 1
            # 2) for each sample, update centroid
            for i in np.arange(self.case_size):
                minDist = 100000.0
                minIndex = 0

                # 2.1)for each centroid,get the index of the centroid
                # which defines the minimum distance
                # between this sample and centroid
                for j in range(self.k):
                    distance = self.euclDistance(self.centroids[j,:],self.X[i,:])
                    if distance<minDist:
                        minDist = distance
                        minIndex = j

                # 2.2) update the sample's centroids
                if(self.labelDecision[i,0] != minIndex):
                    clusterChanged = True
                    self.labelDecision[i,:] = minIndex,minDist
            # 3) for each centroids,update centroids

            for j in range(self.k):
                pointsInCluster = self.X[np.nonzero((self.labelDecision[:,0]==j))[0]]
                self.centroids[j,:] = np.mean(pointsInCluster,axis = 0) # get mean of each dimension

            print('the %d-th iter is running'% iters)
            self.showCluster(iters)

        print('clustering done')

    def showCluster(self,iter):
        plt.figure()
        if self.dim!=2:
            print("sorry, I cannot draw because the dimension of features is not 2")
            return 1
        # marker_data,marker_centroids = plot_colors()

        if self.k > len(marker_data):
            print("Sorry, the k you defined is too large to show")
            return 1

        # draw all samples
        for i in np.arange(self.case_size):
            mark_index = int(self.labelDecision[i,0])
            plt.plot(self.X[i,0],self.X[i,1],marker_data[mark_index])

        # draw all centroids
        for j in range(self.k):
            plt.plot(self.centroids[j,0],self.centroids[j,1],marker_centroids[j])

        plt.savefig("%siter-%02d.png" % (basepath, iter))   # have to savefig before show it,cause show function new another blank figure
        plt.show()


def loadData(filename):
    df = pd.read_csv(filename)
    df.tail()
    # y = df.iloc[:, 5].values
    # y_df = pd.DataFrame(y)

    # y = np.where(y == 'Iris-setosa', -1, 1)  # -1代表山鸢尾，1代表变色鸢尾，将Iris-setosa类标变为-1，其余变为1
    X = df.iloc[:,:].values  # features: petal length, sepal length萼片长度，花瓣长度
    plt.figure()
    plt.scatter(X[:, 0], X[:, 1], marker=marker_data[0][0],color=marker_data[0][1],label='data')
    plt.xlabel('feature1')
    plt.ylabel('feature2')
    plt.legend(loc='upper left')
    plt.savefig(basepath + 'raw.png')    # have to savefig before show it,cause show function new another blank figure
    plt.show()

    return X

if __name__== '__main__':
    sns.set(style="white", color_codes=True)

    X = loadData('data.csv')

    kmeans = KMeans(X=X,k=4)

    kmeans.train()



