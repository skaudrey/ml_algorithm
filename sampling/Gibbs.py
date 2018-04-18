#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/4/4 19:21
# @Author  : MiaFeng
# @Site    : 
# @File    : Gibbs.py
# @Software: PyCharm
__author__ = 'MiaFeng'

'''
Test the two-dimensional Gibbs sampling

The target steady distribution is a two-dimensional Gaussian distribution Norm(\mu,\Sigma)

\mu = (\mu_1,\mu_2)= (5,-1)
\Sigma = [[\sigma_1^2,\rho \sigma_1 \sigma_2],[\rho \sigma_1 \sigma_2, \sigma_2^2]]=[[1,1],[1,4]]

The conditional status transformation distribution are:

P(x_1|x_2) = Norm(\mu_1+\frac{\rho \sigma_1}{sigma_2\big(x_2-\mu_2\big)},(1-\rho^2)\sigma_1^2)
P(x_2|x_1) = Norm(\mu_2+\frac{\rho \sigma_2}{sigma_1\big(x_1-\mu_1\big)},(1-\rho^2)\sigma_2^2)


'''


from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import multivariate_normal
import random
import seaborn as sns
from scipy.stats import multivariate_normal
import math
import matplotlib.pyplot as plt

def p_ygivenx(x,m1,m2,s1,s2):
    return random.normalvariate(m2+rho*s2/s1 * (x-m1),
                                math.sqrt(1-rho**2)*s2) # require standard covariance

def p_xgiveny(y,m1,m2,s1,s2):
    return random.normalvariate(m1+rho*s1/s2*(y-m2),
                                math.sqrt(1-rho**2)*s1)

if __name__=="__main__":
    sns.set()

    sampleSource = multivariate_normal(mean=[5,-1],cov=[[1,0.5],[0.5,2]])

    N = 5000
    K = 20
    x_res = []
    y_res = []
    z_res = []
    m1 = 5
    m2 = -1
    s1 = 1
    s2 = 2

    rho = 0.5
    y = m2

    # Gibbs

    for i in range(N):
        for j in range(K):
            x = p_xgiveny(y,m1,m2,s1,s2)
            y = p_ygivenx(x,m1,m2,s1,s2)    # Attention: x have been updated
            z = sampleSource.pdf([x,y])

            x_res.append(x)
            y_res.append(y)
            z_res.append(z)


    num_bins = 50

    plt.figure()
    plt.hist(x_res, num_bins, normed=1, facecolor='green',alpha=0.5)
    plt.hist(y_res, num_bins, normed=1, facecolor='red', alpha=0.5)
    plt.title("Histogram of Gibbs Sampling")
    plt.show()

    # the two dimensional gaussian distribution of samples
    fig = plt.figure()
    ax = Axes3D(fig, rect=[0,0,1,1],elev=30, azim=20)
    ax.scatter(x_res,y_res,z_res,marker='o')
    plt.show()


