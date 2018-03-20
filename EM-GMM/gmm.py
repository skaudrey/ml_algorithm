#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/3/4 20:30
# @Author  : MiaFeng
# @Site    : 
# @File    : gmm.py
# @Software: PyCharm
__author__ = 'MiaFeng'

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os

from mpl_toolkits.mplot3d import Axes3D # plot 3D figure

# Warning!!! put color in front of marker style
marker_data = ['ro', 'bo', 'go', 'ko', 'r^', 'r+', 'rs', 'rd', 'r<', 'rp']

marker_centroids = ['rD', 'bD', 'gD', 'kD', 'b^', 'b+', 'bs', 'bd', 'b<', 'bp']

line_color = ['r-','b^','g-','k-.']

basepath = os.getcwd()+'/file/gmmTex/fig/'

def generateData(Sigma,Mu,alpha,k,case_size):
    '''
    Generate training data with shape (case_size,2) according to Sigma,Mu and case_size
    :param Sigma: 2*2 array,covariance of multivariate gaussian distribution(#(features) = 2)
    :param Mu: k*2 array, mean of each multivariate Gaussian Distribution
    :param k: scalar, # (multivariate Guassian Distribution component)
    :param case_size: the size of training data
    :param alpha: k*1, the coefficient of components
    :return: X :training data with shape (case_size,2);
              y: The label of training data X with shape (case_size,1), domain[0,k-1]
    '''
    X = np.zeros((case_size,2),dtype=np.float32)
    y = np.zeros((case_size,1),dtype=np.int)

    for i in np.arange(case_size):
        rand = np.random.random()   # generate a random number in [0,1]
        sum = 0
        index = 0
        while(index<k):
            sum += alpha[index]
            if(rand < sum):
                X[i,:] = np.random.multivariate_normal(Mu[index,:].flatten(),Sigma,1)
                y[i] = index
                break
            else:
                index += 1
    return X,y

def plot_raw(X):
    plt.figure()
    plt.scatter(X[:,0],X[:,1],marker='o',color = 'r',label='original data')
    plt.savefig(basepath+'raw.png')
    plt.show()



def plot_raw_label(X,y,fileName="raw-true.png"):
    plt.figure()
    case_size,dim = X.shape
    labels = ['Component 1','Component 2','Component 3','Component 4']
    groups_count = len(np.unique(y))

    pts_group = [[],[],[],[]]

    for i in np.arange(case_size):
        pts_group[y[i,0]].append(X[i,:])
        # plt.plot(X[i,0],X[i,1],marker_data[y[i,0]])

    for j in np.arange(groups_count):
        x1 = [x[0] for x in pts_group[j][:]]    # WARNING: for list have to get one column in iteration
        x2 = [x[1] for x in pts_group[j][:]]
        plt.scatter(x1,x2,s=10*(j+1),c= marker_data[j][0],label=labels[j])
        # WARNING --- list indices must be intergers or slices, not tuple

    plt.xlabel('feature 1')
    plt.ylabel('feature 2')
    plt.legend()

    plt.savefig(basepath+fileName)
    plt.show()


class GMM:
    def __init__(self,X,sigma,k,tolerance=0.0001,maxIterations=None):
        self.X = X
        self.k = k
        self.case_size, self.dim = self.X.shape
        self.mu = np.zeros((self.k,self.dim))
        # self.sigma = np.zeros((self.k,self.dim))
        self.expectations = np.zeros((self.case_size,self.k))# probability density of latent variable
        self.tolerance = tolerance  # threshold for terminating
        self.alpha = np.zeros((self.k,1))
        # self.sigma = np.zeros((self.dim,self.dim))
        self.sigma = np.zeros((self.k,self.dim,self.dim))
        self.maxIterations = maxIterations  # termination condition
        # self.mu_trace = np.zeros((self.maxIterations,self.k,self.dim))
        # store the old mu of each component in each iteration
        self.mu_trace = [[],[],[],[]]
        self.label = np.zeros((self.case_size,1),dtype=np.int)
        self.prbDensity = np.zeros((self.case_size,1),dtype=np.float32)
        # mixture gaussian probability density of each samples

    def initParam(self):
        '''
        Initialize mu to a random vector , alpha follows normal distribution
        :return:
        '''
        self.mu = np.random.random((self.k,self.dim))*10

        mu1, mu2, mu3, mu4 = [5, 35], [30, 40], [20, 20], [45, 15]
        Mu = np.array([mu1, mu2, mu3, mu4])
        self.mu = Mu

        self.alpha = np.array([0.25,0.25,0.25,0.25])
        self.sigma = np.array([[[10,0],[0,10]],[[10,0],[0,10]],[[10,0],[0,10]],[[10,0],[0,10]]]) # cannot be too small incase of overflowing

    def PDF(self,x,mu,sigma):
        '''
        The probability density function of 2-d gaussian distribution
        :return: scalar: the probability density
        '''
        sigma_sqrt = np.sqrt(np.linalg.det(sigma))
        sigma_inv = np.linalg.inv(sigma)
        minus_mu = x - mu
        minus_mu_transpose = np.transpose(minus_mu)

        # np.dot() gives a scalar
        pdf = np.exp(-0.5*(np.dot(np.dot(minus_mu_transpose,sigma_inv),minus_mu)))/(2.0 * np.pi * sigma_sqrt)

        return pdf

    def e_step(self):
        '''
        Get responsivity of each samples,
        and update the expectations of probability density of each samples
        Reference : equation in page 195 of LiHang's book
        :return:
        '''
        for i in range(self.case_size):
            denom = 0
            for j in range(self.k):
                denom += self.alpha[j]* self.PDF(X[i,:],self.mu[j,:],self.sigma[j,:])
                numer = self.PDF(self.X[i,:],self.mu[j,:],self.sigma[j,:])
                self.expectations[i,j] = self.alpha[j]*numer/denom

    def m_step(self):
        '''
        Update parameters(mu,sigma,alpha) by maximizing Q function
        Reference: equation (9.30-9.32) in page 165 of LiHang's book

        Tips: In the view of 2-d multivariate Gaussian distribution, mu's shape is (1,2).
        Thus,
        :return:
        '''

        for j in range(self.k):
            denom = 0.0
            numer = 0.0
            numer_sigma = np.zeros((2,2),dtype=np.float32)
            for i in range(self.case_size):
                # according to equation (9.31), the mu here is the old one not the new one
                # but most versions I saw take the new mu into derivations
                # remained open here
                # BUT!!!!!!!!!!!!!!!!!!
                # when I wrote like these, after updating sigma, there are sigular matrix in sigma
                # !---------------- remained open here ---------------------!
                # numer_sigma += self.expectations[i, j] * (self.X[i, :] - self.mu[j, :]) ** 2
                numer += self.expectations[i,j] * self.X[i,:]
                denom += self.expectations[i,j]
            self.mu[j,:] = numer/denom
            self.alpha[j] = denom/self.case_size

            # xshift = self.X[i,:]-self.mu[j,:]
            # xshift.shape = (2,1)    #IMPORTANT: numpy cannot transpose 1-d array,have to set shape
            # xshiftT = np.transpose(xshift)
            #
            # for i in np.arange(self.case_size):
            #     numer_sigma += self.expectations[i, j] * xshift * xshiftT

            # self.sigma[j,:] = numer_sigma/denom

    def train(self):
        '''
        Main function for training
            1) Initialization
            2) Repeat{
                2.1) E-step : get the responsivity for each samples
                2.2) M-step : update parameters: alpha,sigma, mu
            } util delta < tolerance or iters>maxIterations
        :return:
        '''
        self.initParam()
        iters = 0
        if self.maxIterations == None:
            while(True):
                err = 0
                err_alpha = 0
                old_mu = self.mu.copy()
                self.e_step()
                self.m_step()
                delta = self.mu - old_mu
                iters += 1
                for j in range(self.k):
                    self.mu_trace[j].append(self.mu[j,:])
                delta_temp = True
                for ss in np.abs(delta):
                    temp = ss < self.tolerance
                    delta_temp = delta_temp and temp[0] and temp[1]
                # delta_temp = [delta_temp and (ss < self.tolerance) for ss in np.abs(delta)]
                if delta_temp:
                    # make decisions
                    self.decision()
                    return iters
        else:
            for i in range(self.maxIterations):
                # while (True):
                    err = 0
                    err_alpha = 0
                    old_mu = self.mu.copy()
                    self.e_step()
                    self.m_step()
                    delta = self.mu - old_mu
                    iters = i + 1
                    for j in range(self.k):
                        self.mu_trace[j].append(self.mu[j, :])
                    delta_temp = True
                    delta_temp = [delta_temp and (ss < self.tolerance) for ss in np.abs(delta)]
                    if delta_temp:
                        # make decisions
                        self.decision()
                        return iters    # early stop
        return iters


    def decision(self):
        '''
        make a label decision for each samples according to the maximum expectations
        :return:
        '''
        for i in np.arange(self.case_size):
            for j in np.arange(self.k):
                if self.expectations[i,j]==max(self.expectations[i,:]):
                    self.label[i] = j
                self.prbDensity[i] += self.alpha[j] * self.PDF(X[i,:],self.mu[j,:],self.sigma[j,:])


    def plot_trace(self):
        '''
        3D figure for plotting the mean per iteration
        :return:
        '''
        plt.figure()
        for j in range(self.k):
            plt.subplot(2,2,j+1,projection='3d')
            plt.xlabel("eruptions")
            plt.ylabel('waiting')
            iters_size = len(self.mu_trace[0][:])
            mu_1 = [x[0] for x in self.mu_trace[j][:]]
            mu_2 = [x[1] for x in self.mu_trace[j][:]]
            plt.plot(range(iters_size),mu_1,mu_2,line_color[j])
            plt.title('The mean trajectory of Component-%d'%(j+1))

        plt.savefig(basepath+'trace.png')
        plt.show()

    def plot_prbDensity(self):
        '''
        3D figure, plot the probability density of each samples
        :return:
        '''
        plt.figure()
        plt.title("The mixed probability density of each samples")

        axs = plt.subplot(111, projection='3d')

        for i in np.arange(self.case_size):
            y_label = self.label[i][0]
            axs.scatter(self.X[i,0],
                        self.X[i,1],
                        self.prbDensity[i],
                        c=marker_data[y_label][0])


        plt.savefig(basepath+'prdDensity.png')
        plt.show()

    def plot_data(self):
        plot_raw_label(self.X,self.label,fileName='decisionData.png')


if __name__== '__main__':
    sns.set(style='white')
    alpha = np.array([0.1,0.2,0.3,0.4])
    iter_num = 1000 # the maximum number of iterations
    case_size = 500
    k = 4
    probability = np.zeros(case_size)   # mixed gaussian distribution

    #===========  define the mean and covariance of each multivariate gaussian distribution component ===============
    mu1,mu2,mu3,mu4 = [5,35],[30,40],[20,20],[45,15]
    Mu = np.array([mu1,mu2,mu3,mu4])
    Sigma = np.array([[30,0],[0,30]])

    X,y = generateData(Sigma,Mu,alpha,k,case_size)
    plot_raw(X)
    plot_raw_label(X,y)

    # ==================     EM training and estimation    ============
    gmm = GMM(X,Sigma,k,tolerance=0.0001)
    iters = gmm.train()
    print("The total number of iterations: %d" % iters)
    gmm.plot_data()
    gmm.plot_prbDensity()
    gmm.plot_trace()
