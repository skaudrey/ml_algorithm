#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/3/4 15:19
# @Author  : MiaFeng
# @Site    : 
# @File    : plot_util.py
# @Software: PyCharm
__author__ = 'MiaFeng'

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np

def plot_set(style):
    sns.set(style=style,color_codes=True)

def plot_markers():
    return ('s','x','o','^','v','.','+','*','D','d','<','*','1','2','3','4','s','p')

def plot_colors():
    #b---blue   c---cyan  g---green    k----black    m---magenta     r---red  w---white    y----yellow
    return ('red','blue','lightgreen','gray','cyan','black','yellow','magenta','white')

def plot_decision_regions(X,y,classifier,basepath,fileName,resolution = 0.02):
    markers = plot_markers()
    colors = plot_colors()

    cmap = ListedColormap(colors[:len(np.unique(y))])

    plt.figure()

    # plot the decision surface
    x1_width = X[:,0].max() - X[:, 0].min()
    x2_width = X[:,1].max() - X[:, 1].min()
    x1_min, x1_max = np.floor(X[:,0].min() - 0.3* x1_width), np.ceil(X[:, 0].max() + 0.3* x1_width)
    x2_min, x2_max = np.floor(X[:,1].min() - 0.3* x2_width), np.ceil(X[:, 1].max() + 0.3* x2_width)

    xx1, xx2 = np.meshgrid(np.arange(x1_min,x1_max,resolution),
                           np.arange(x2_min,x2_max,resolution))
    # ss1 = xx1.ravel()
    # ss2 = xx2.ravel()
    Z = classifier.predict(np.array([xx1.ravel(),xx2.ravel()]).T) # flatten
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1,xx2,Z,alpha=0.4,cmap=cmap)
    plt.xlim(xx1.min(),xx1.max())
    plt.ylim(xx2.min(),xx2.max())

    # plot class samples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x = X[y==cl,0],y=X[y==cl,1],alpha=0.8,c=cmap(idx),marker=markers[idx],label = cl)

    plt.savefig(basepath+fileName)
    plt.show()


def plotImg(df,xlabel=None,ylabel=None,title=None,kind='bar'):
    sns.set()

    df.plot(kind=kind, title=title)
    plt.ylabel(ylabel)