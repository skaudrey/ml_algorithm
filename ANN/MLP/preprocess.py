#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/4/10 19:48
# @Author  : MiaFeng
# @Site    : 
# @File    : preprocess.py
# @Software: PyCharm
__author__ = 'MiaFeng'

import os
import struct
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys

def load_mnist(path,kind = 'train'):
    '''
    load MNIST data from path
    :param path:
    :param kind:
    :return:
    '''

    labels_path = os.path.join(path,'%s-labels.idx1-ubyte'%kind)
    # labels_path = 'D:\Code\TestData\mnist\\train-labels-idx1-ubyte'
    images_path = os.path.join(path,'%s-images.idx3-ubyte'%kind)

    with open(labels_path,'rb') as lbpath:
        magic, n = struct.unpack('>II',lbpath.read(8))  # big ending with unsigned integer
        labels = np.fromfile(lbpath,dtype=np.uint8)

    with open(images_path,'rb') as imgpath:
        magic, num, rows,cols = struct.unpack(">IIII",imgpath.read(16))
        images = np.fromfile(imgpath, dtype=np.uint8).reshape(len(labels),784)
    return images,labels

# def plot_image_alltype():
#     fig,ax = plt.subplots(nrows=2,ncols=5,sharex = True, sharey = True) #Attention: subplots not subplot
#     ax = ax.flatten()   # flatten the position for plotting
#
#     for i in range(10):
#         img = X_train[y_train==i][0].reshape(28,28)
#         ax[i].imshow(img, cmap='Greys',interpolation='nearest')
#
#     ax[0].set_xticks([])
#     ax[1].set_yticks([])
#     plt.tight_layout()
#     plt.show()
#
# def plot_image_onetype():
#     fig,ax = plt.subplots(nrows=5,ncols=5,sharex=True,sharey=True)
#
#     ax = ax.flatten()
#
#     for i in range(25):
#         img = X_train[y_train==7][i].reshape(28,28)
#         ax[i].imshow(img,cmap='Greys',interpolation='nearest')
#     ax[0].set_xticks([])
#     ax[1].set_yticks([])
#     plt.tight_layout()
#     plt.show()
#
# def saveData():
#     basepath = os.getcwd()
#     np.savetxt(basepath+'/train_img.csv',X_train,fmt='%i',delimiter=',')
#     np.savetxt(basepath + '/train_labels.csv', y_train, fmt='%i', delimiter=',')
#     np.savetxt(basepath + '/test_img.csv', X_test, fmt='%i', delimiter=',')
#     np.savetxt(basepath + '/test_labels.csv', y_test, fmt='%i', delimiter=',')
#
# def loadData():
#     basepath = os.getcwd()
#     X_train =  np.genfromtxt(basepath+'/train_img.csv',dtype=int,delimiter=',')
#     y_train = np.genfromtxt(basepath + '/train_labels.csv', dtype=int,delimiter=',')
#     X_test = np.genfromtxt(basepath + '/test_img.csv', dtype=int,delimiter=',')
#     y_test = np.genfromtxt(basepath + '/test_labels.csv', dtype=int,delimiter=',')
#
#     return X_train, y_train, X_test, y_test

# sns.set()
#
# X_train, y_train = load_mnist('D:/Code/TestData/mnist',kind='train')
# print('Rows: %d, columns: %d'%(X_train.shape[0],X_train.shape[1]))
#
# X_test, y_test = load_mnist('D:/Code/TestData/mnist',kind='test')
# print('Rows: %d, columns: %d'%(X_test.shape[0],X_test.shape[1]))
#
# plot_image_alltype()
# plot_image_onetype()
# # saveData()
# X1,y1,X2,y2 = loadData()
# print(X1.shape)