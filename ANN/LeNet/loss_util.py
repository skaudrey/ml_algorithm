#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/4/16 23:55
# @Author  : MiaFeng
# @Site    : 
# @File    : loss_util.py
# @Software: PyCharm
__author__ = 'MiaFeng'

from keras.callbacks import Callback

class LossOfKeras(Callback):
    def on_epoch_begin(self, epoch, logs={}):
        self.acc = [{'epoch',},{'batch',}]
