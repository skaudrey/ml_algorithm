#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/8/6 14:45
# @Author  : MiaFeng
# @Contact : skaudrey@163.com
# @Site    :
# @File    : CModel.py
# @Software: PyCharm
__author__ = 'MiaFeng'

import numpy as np
import itertools
from sklearn import metrics
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

class CModel(object):

    def __init__(self,
                 **params):
        for key, value in params.items():
            self.params[key] = value

    def train(self, x_train, y_train, x_val, y_val):
        pass


    def predict(self, x_test):
        pass

    def save_model(self, file_name):
        pass


    def load_model(self, file_name, n_folds=5):
        pass

    def plot_ROC(self,y, y_predict):
        plt.figure()
        fpr, tpr, thresholds = metrics.roc_curve(
            y, y_predict, pos_label=1)
        auc = "%.2f" % metrics.auc(fpr, tpr)
        title = 'ROC Curve, AUC = ' + str(auc)
        with plt.style.context(('ggplot')):
            fig, ax = plt.subplots()
            ax.plot(fpr, tpr, "#000099", label='ROC curve')
            ax.plot([0, 1], [0, 1], 'k--', label='Baseline')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.legend(loc='lower right')
            plt.title(title)
        plt.show()

    def plotConfusionMatrix(self,cm,classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
        '''
        This function prints and plots the confusion matrix.
            Normalization can be applied by setting `normalize=True`.
        :param cm: confusion matrix
        :param y: the real label
        :param y_predict: the predicted label
        :param classes: The class name of each class denoted by integers.
        :param normalize:
        :param title:
        :param cmap: color map
        :return:
        '''
        plt.figure()
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')

        print(cm)

        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)  # 0 for negative, 1 for positive
        plt.yticks(tick_marks, classes)

        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.show()


    def eval(self,y,y_predict,metrics='auc'):
        if metrics=='roc_curve':
            self.plot_ROC(y,y_predict)
        elif metrics == 'confusion_matrix':
            self.plotConfusionMatrix(cm = confusion_matrix(y,y_predict),
                                     classes=['negative','positive'])
        elif metrics == 'auc':
            print 'auc score of model -- %s: %.2f' % (self.modelName(),roc_auc_score(y, y_predict))
        else:
            print "help yourself"

    def modelName(self):
        return self.__class__.__name__

if __name__=='__main__':
    X,y = load_iris().data
    print X.shape