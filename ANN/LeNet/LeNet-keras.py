#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/4/16 23:36
# @Author  : MiaFeng
# @Site    : 
# @File    : LeNet-keras.py
# @Software: PyCharm
__author__ = 'MiaFeng'

import keras.datasets.mnist
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Dense,Activation,Flatten
from keras.models import Sequential


class CreateLeNet(object):
    def createLeNet(self):

        conv_layers = [
            Conv2D(filters=6,kernel_size=5,strides=2,padding='same',activation='relu'),
            MaxPooling2D(pool_size=(2,2),strides=2),
            Conv2D(filters=20, kernel_size=5, strides=2, padding='same', activation='relu'),
            MaxPooling2D(pool_size=(2, 2), strides=2),
            Flatten()
        ]

        fc_layers = [
            Dense(120,activation='relu'),
            Dense(10,activation='softmax')
        ]

        model = Sequential([conv_layers,fc_layers])

        return model