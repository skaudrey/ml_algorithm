#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/4/16 23:55
# @Author  : MiaFeng
# @Site    : 
# @File    : loss_util.py
# @Software: PyCharm
__author__ = 'MiaFeng'

from keras.callbacks import Callback
import matplotlib.pyplot as plt
import seaborn as sns

class LossOfKeras(Callback):
    def on_epoch_begin(self, epoch, logs={}):
        self.acc = {'epoch':[],'batch':[]}
        self.loss = {'epoch': [], 'batch': []}
        self.val_loss = {'epoch': [], 'batch': []}
        self.val_acc = {'epoch': [], 'batch': []}

    # get loss after each batch
    def on_batch_end(self, batch, logs={}):
        self.loss['batch'].append(logs.get('loss'))
        self.acc['batch'].append(logs.get('acc'))
        self.val_acc['batch'].append(logs.get('val_acc'))
        self.val_loss['batch'].append(logs.get('val_loss'))

    def on_epoch_end(self, epoch, logs={}):
        self.loss['epoch'].append(logs.get('loss'))
        self.acc['epoch'].append(logs.get('acc'))
        self.val_loss['epoch'].append(logs.get('val_loss'))
        self.val_acc['epoch'].append(logs.get('val_acc'))

    def loss_plot(self,loss_type):
        '''

        :param loss_type: 'epoch' for ploting the loss after each epoch; 'batch' for plotting the loss after each batch
        :return:
        '''

        sns.set()

        iters = range(len(self.loss[loss_type]))

        plt.figure()

        # acc
        plt.plot(iters,self.acc[loss_type],'r',label='train acc')
        # loss
        plt.plot(iters,self.loss[loss_type],'g',label='train loss')

        if loss_type=='epoch':
            # val_loss
            plt.plot(iters,self.val_loss[loss_type],'b',label='val loss')
            # val_acc
            plt.plot(iters,self.val_acc[loss_type],'k',label='val acc')

        plt.xlabel(loss_type)
        plt.ylabel('acc-loss')
        plt.legend(loc='upper right')
        plt.show()