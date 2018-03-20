#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/3/1 19:20
# @Author  : MiaFeng
# @Site    : 
# @File    : svm.py
# @Software: PyCharm
__author__ = 'MiaFeng'
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# from __future__ import division #using the dibision in python3
import scipy as sp
import random as rd
# from numpy.random import random


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
    plt.show()
    return X,y

class Kernel:
    @classmethod
    def Gauss_kernel(cls, x, z, sigma=2):
        return np.exp(-np.sum((x - z) ** 2 / (2 * sigma ** 2)))

    @classmethod
    def Linear_kernel(cls, x, z):
        return np.sum(x*z)  # 实际上就是原始空间中的内积

class SVM:


    def __init__(self,X,y,c=10,tol=0.01,kernel=Kernel.Linear_kernel):
        '''
        :param X: N*M matrix, where N is #features, M is the #train_case
        :param y: 1*M array, where M is the #train_case
        :param c: slack variable
        :param tol: termination condition for iteration,which is a tolerance value used for determining equality of floating point numbers
        :param kernel: kernel function
        '''
        self.X = np.array(X)
        self.y = np.array(y)
        self.tol = tol
        self.N,self.M = self.X.shape
        self.C = c  # slack variables
        self.kernel = kernel
        self.alpha = np.zeros((1,self.M)).flatten(1)    # change the #train_case as the first dimension, #alpha = #train_case
        self.supportVec = []    # support vectors
        self.b = 0
        self.E = np.zeros((1,self.M)).flatten(1)

    def fitKKT(self,i):
        '''
        check and pick up the alpha who violates the KKT condition
        I. satisfy KKT condition
            1) yi*f(i) >= 1 and alpha == 0 (outside the boundary)
            2) yi*f(i) == 1 and 0<alpha< C (on the boundary)
            3) yi*f(i) <= 1 and alpha == C (between the boundary)
        II. violate KKT condition
        because y[i]*E_i = y[i]*f(i) - y[i]^2 = y[i]*f(i) - 1, so
            1) if y[i]*E_i < 0, so yi*f(i) < 1, if alpha < C, violate!(alpha = C will be correct)
            2) if y[i]*E_i > 0, so yi*f(i) > 1, if alpha > 0, violate!(alpha = 0 will be correct)
            3) if y[i]*E_i = 0, so yi*f(i) = 1, it is on the boundary, needless optimized

        :param i: the index of test_case
        :return: Bool, True for fitting KKT condition, Flase for not fitting KKT condition
        '''

        if(((self.y[i]*self.E[i]<-self.tol and self.alpha[i]<self.C)) or \
                ((self.y[i] * self.E[i])> self.tol and (self.alpha[i] > 0)) ):
            return False
        return True

    def updateE(self,i):
        '''
        update E of ith test_case
        refrence the equation (7.105) in page 127 of LiHang's book
        E_i = \sum\limits_{j=1}^{m}\alpha_j*y_j*K\big(x_i,x_j\big) + b - y_i
        :param i: the index of test_case
        :return:
        '''
        self.E[i] = 0
        for j in range(self.M):
            self.E[i] += self.alpha[j] * self.y[j] * self.kernel(self.X[:,i],self.X[:,j])

        self.E[i] += self.b - self.y[i]

    def randJ(self,i):
        '''
        get the index of alpha2,
        when one cannot get an inxde j of alpha,
        of which the corresponding abs(E[i],E[j]) is large enough
        :param i:  the index of alpha1
        :return: the index j of alpha2
        '''
        j = rd.sample(range(self.M),1)
        while j==i:
            j = rd.sample(range(self.M),1)
        return j[0] # what rd.sample returns is a list

    def findJ(self,i,ls):
        '''
        Find the index j for alpha2.
        Two methods:
        1) take the index j where abs(E[i]-E[j]) is the largest,
        which means alpha2 will make more changes for the new alpha2
        2) if for all the j in list, there's not a proper j, get a j randomly
        :param i: the index of alpha1
        :param ls:  candidate index list
        :return:
        '''
        ansj = -1
        maxx = -1
        self.updateE(i) # set E[i] firstly

        for j in ls:
            if i==j:continue
            self.updateE(j) # set E[j] for the candidate j
            # take the index j where abs(E[i]-E[j]) is the largest, which means alpha2 will make more changes for the new alpha2
            deltaE = np.abs(self.E[i]-self.E[j])
            if deltaE> maxx:
                maxx = deltaE
                ansj = j

        # if for all the j in list, there's not a proper j, get a j randomly
        if ansj == -1:
            return self.randJ(i)
        return ansj

    def select(self,i):
        '''
        take the inital state of alpha,which is a zero-array into consideration,
        and minimize the searching range of index by limit the range from
        :param i:
        :return:
        '''
        pp = np.nonzero(self.alpha>0)[0]
        if(pp.size>0):  # limit the searching range of index in the supportVector
            j = self.findJ(i,pp)
        else:   # the initial alpha is a zero-array
            j = self.findJ(i,range(self.M))
        return j

    def clipAlpha(self,aj,H,L):
        if aj > H:
            aj = H
        if aj < L:
            aj = L
        return aj


    def InnerLoop(self,i,threshold):
        '''
       The InnerLoop for choosing alpha2 and update alpha,b,E.

       Main reference: the books of LiHang

       Algorithm:
        1) update alpha[j]
            equation (7.106) in page 127
        2) clip alpha[j]
            equation in the bottom in page 126
        3) update alpha[i],b,and E
            update alpha[i]: equation (7.114) in page 129
            update b: equation (7.114-7.116) and the other in the top of page 130
            update E: equation (7.105) in page 127

        :param i:  the index of alpha1
        :param threshold:
        :return:
        True for fail
            1) the changes of alpha[j] is smaller than threshold, have to go back and chooose another j
            2) divide by zero while updating alpha[j]
         false for done
        '''
        j = self.select(i)

        self.updateE(i)
        self.updateE(j)

        # backup of the alpha which are gonna be updated
        a2_old = self.alpha[j]
        a1_old = self.alpha[i]


        # ===========  1) update alpha[j]=============
        k11 = self.kernel(self.X[:,i],self.X[:,i])
        k22 = self.kernel(self.X[:,j],self.X[:,j])
        k12 = self.kernel(self.X[:,i],self.X[:,j])
        eta = k11 + k22 - 2 * k12

        # cannot divide by zero
        if eta==0:
            return True

        self.alpha[j] = a2_old + y[j]* (self.E[i] - self.E[j])/eta

        # ===========  2) clip alpha[j] =============
        if(self.y[i] == self.y[j] ):
            L = max(0, a2_old + a1_old - self.C)
            H = min(self.C, a1_old + a2_old)
        else:
            L = max(0, a2_old - a1_old)
            H = min(self.C, self.C + a2_old - a1_old)

        self.alpha[j] = self.clipAlpha(self.alpha[j],H,L)

        # if alpha[j] doesn't change to much, have to pick another j
        if np.abs( self.alpha[j] - a2_old) < threshold:
            return True

        # ===========  3) update alpha[i],b,and E =============
        # update alpha[i]
        self.alpha[i] = a1_old + self.y[i] * self.y[j] * (a2_old - self.alpha[j])

        # update b, need trade-off
        b1_new = self.b - self.E[i] - self.y[i]* k11 * (self.alpha[i] - a1_old)-\
            self.y[j] * k12 * (self.alpha[j] - a2_old)

        b2_new = self.b - self.E[j] - self.y[i]* k12 * (self.alpha[i] - a1_old)-\
            self.y[j] * k22 * (self.alpha[j] - a2_old)

        if (self.alpha[i]>0 and self.alpha[i]<self.C and self.alpha[j]>0 and self.alpha[j]<self.C):
            self.b = b1_new # or b2_new, they are the same
        else:
            self.b = ( b1_new + b2_new ) / 2

        # update E
        self.updateE(j)
        self.updateE(i)

        return False


    def train(self,maxiter = 100, threshold = 0.000001):
        '''
        If the new alpha doesn't change much enough,
        aka abs(alpha[j]-a2_old)< threshold, we have to choose another i,
        which means we have to jump to the outer loop and
        traverse the support vectors corresponding to index i.
        After traversing the support vecots corresponding to index i,
        if the condition is still not satisfied,
        we have to drop the index i and choose another i
        :param maxiter: termination iteration number
        :param threshold:
        :return:
        '''
        iters = 0
        flag = False
        # initialization
        for i in range(self.M):
            self.updateE(i)

        while(iters<maxiter and (not flag)):
            flag = True
            temp_supportVec = np.nonzero((self.alpha>0))[0]
            iters += 1
            # if the alpha[j] isn't changed too much, tranverse the support vector
            for i in temp_supportVec:
                self.updateE(i)
                # choose alpha[i] and alpha[j], updates
                if(not self.fitKKT(i)):
                    flag = flag and self.InnerLoop(i,threshold)
            # if all the alpha[j] is not proper, drop alpha[i] and choose another one
            if(flag):
                for i in range(self.M):
                    self.updateE(i)
                    if(not self.fitKKT(i)):
                        flag = flag and self.InnerLoop(i,threshold)
            print('the %d-th iter is running'% iters)
        self.supportVec = np.nonzero((self.alpha>0))[0]


    def predict(self,x):
        '''
        w is updated by support vectors and corresponding alpha
        fx is the hyperplane
        :param x:
        :return:
        '''
        fx = 0
        for t in self.supportVec:
            fx += self.alpha[t] * self.y[t]*self.kernel(self.X[:,t],x).flatten(1)
        fx += self.b
        return np.sign(fx)

    def pred(self,X):
        test_X = np.array(X)
        y = []
        for i in range(test_X.shape[1]):
            y.append(self.predict(test_X[:,i]))
        return y

    def error(self,X,y):
        py = np.array(self.pred(np.array(X))).flatten(1)

        print("the #error_case is  ", np.sum(py!=np.array(y)))


    def plot_test_linear(self):
        w = 0
        for t in self.supportVec:
            w+=self.alpha[t]*self.y[t]*self.X[:,t].flatten(1)
        w = w.reshape(1,w.size)

        print(np.sum(np.sign(np.dot(w,self.X)+self.b).flatten(1)!=self.y),'errors')

        x1 = 0
        y1 = -self.b/w[0][1]
        y2 = 0
        x2 = -self.b/w[0][0]

        plt.figure(2)

        # plt.plot([x1+x1-x2,x2])
        # plt.plot(the array of x-axis, the array of y axis)
        plt.plot([x1, x2], [y1 , y2])
        x1_max = np.ceil(max(self.X[0,:]))
        x2_max = np.ceil(max(self.X[1,:]))
        x1_min = np.floor(min(self.X[0,:]))
        x2_min = np.floor(min(self.X[1,:]))
        x1_width = x1_max - x1_min
        x2_width = x2_max - x2_min
        x1_min -= x1_width*0.1
        x1_max += x1_width*0.1
        x2_min -= x2_width*0.1
        x2_max += x2_width*0.1

        plt.axis([x1_min,x1_max,x2_min,x2_max])



        for i in range(self.M):
            if self.y[i]==-1:
                plt.plot(self.X[0,i],self.X[1,i],'or')
            elif self.y[i] == 1:
                plt.plot(self.X[0,i],self.X[1,i],'ob')
        for i in self.supportVec:
            plt.plot(self.X[0,i],self.X[1,i],'oy')
        plt.show()




if __name__== '__main__':
    sns.set(style="white", color_codes=True)
    color_plot = {0: 'r', 1: 'b'}  # render the positive class to blue color and negative class to red color
    X,y = loadData()
    X = np.transpose(X)
    # X = X.flatten(2)

    print(X.shape)
    svms = SVM(X,y,kernel=Kernel.Linear_kernel)

    svms.train()
    print(len(svms.supportVec),'SupportVectors:\n')

    for i in range(len(svms.supportVec)):
        t = svms.supportVec[i]
        print(svms.X[:, t])
    svms.error(X, y)
    svms.plot_test_linear()
