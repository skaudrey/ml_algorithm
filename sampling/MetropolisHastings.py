#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/4/4 19:21
# @Author  : MiaFeng
# @Site    : 
# @File    : MetropolisHastings.py
# @Software: PyCharm
__author__ = 'MiaFeng'

'''
Simulate the Metropolis-Hastings sampling 

The target distribution is a norm distribution with mean three and covariance 2
'''


import random
import math
from scipy.stats import norm
import matplotlib.pyplot as plt
import seaborn as sns

# calculate the probability density of the steady target distribution
def norm_dist_prob(theta):
    y = norm.pdf(theta, loc = 3, scale = 2) #shift the standard gaussian distribution with mean loc and varaince scale
    return y



if __name__=="__main__":
    sns.set()
    T = 5000
    pi = [ 0 for i in range(T)]
    sigma = 1   # start from standard norm distribution
    t = 0

    # the status transformation matrix Q is omitted here cause it is taken as the symmetric matrix
    while t < T-1:
        t = t+1
        pi_star = norm.rvs(loc=pi[t-1], scale = sigma, size = 1 ,random_state=None) # sample one with the gaussian distribution with mean loc and variance scale
        alpha = min(1,(norm_dist_prob(pi_star[0])/norm_dist_prob(pi[t-1])))

        u = random.uniform(0,1)

        # Accept and reject
        if u < alpha:
            pi[t] = pi_star[0]
        else:
            pi[t] = pi[t-1] #不接受转移，则用上一次采样的值作为当前时刻的采样值

    plt.figure()
    plt.scatter(pi,norm.pdf(pi,loc = 3,scale=2))    # 采样点所对应的平稳分布的真实概率密度
    num_bins = 50
    plt.hist(pi, num_bins,normed = 1, facecolor ='red',alpha=0.7)   # normed is set true for probability
    # 采样数据集的频率分布直方图，可以看到还是很接近概率密度值的
    plt.show()

import pickle
