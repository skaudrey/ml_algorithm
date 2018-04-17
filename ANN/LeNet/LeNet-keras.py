#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/4/16 23:36
# @Author  : MiaFeng
# @Site    : 
# @File    : LeNet-keras.py
# @Software: PyCharm
__author__ = 'MiaFeng'

from keras.datasets import mnist
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Dense,Activation,Flatten
from keras.models import Sequential
from keras.optimizers import Adam
from keras.utils import np_utils    # change to one hot

class CreateLeNet(object):
    def createLeNet(self,input_shape,nb_class):

        conv_layers = [
            Conv2D(filters=6,kernel_size=5,strides=1,padding='same',activation='relu',input_shape=input_shape),
            MaxPooling2D(pool_size=(2,2),strides=(2,2)),
            Conv2D(filters=16, kernel_size=5, strides=1, padding='same', activation='relu'),
            MaxPooling2D(pool_size=(2, 2), strides=(2,2)),
            Flatten()
        ]

        fc_layers = [
            Dense(120,activation='relu'),
            Dense(nb_class,activation='softmax')
        ]

        model = Sequential(conv_layers+fc_layers)

        return model

# parameters
VERBOSE = 1
IMG_ROW,IMG_COL = 28,28
NB_CLASS = 10
BATCH_SIZE = 128
OPTIMIZER = Adam()
VALIDATION_SPLIT = 0.2
INPUT_SHAPE = [IMG_ROW,IMG_COL,1]
NB_EPOCH = 20


# load mnist dataset
(X_train, Y_train),(X_test, Y_test) = mnist.load_data()
