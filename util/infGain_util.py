#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/3/6 11:16
# @Author  : MiaFeng
# @Site    : 
# @File    : infGain_util.py
# @Software: PyCharm
__author__ = 'MiaFeng'

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns




class InfGain:
    '''
    基尼系数可以理解为降低误分类可能性的标准，与熵的结果差不多。与熵类似，当所有类别是等比例分布时，基尼系数的值最大
    熵和基尼系数对各类别样本数量的变动不敏感，但是误分类率对其敏感
    所以实践中常用熵或者基尼系数作为决策树的criterion
    '''

    def __init__(self,P):
        '''
        Get information gain measured by Gini Index, Entropy, Standard Entropy, Error rate
        :param P: array of probabilities
        '''
        self.P = P
        self.entropy__ = [self.entropy(p) if p!=0 else None for p in P]
        self.std_entropy__ = [e*0.5 if e else None for e in self.entropy__] #standard entropy
        self.error__ = [self.error(p) for p in P]
        self.gini__ = self.gini(P)


    # @classmethod
    def gini(self,p):
        return p*(1-p)+(1-p)*(1-(1-p))

    # @classmethod
    def entropy(self,p):
        return -p*np.log2(p)-(1-p)*np.log2((1-p))

    # @classmethod
    def std_entropy(self,p):
        return [e*0.5 if e else None for e in self.entropy(p)]

    # @classmethod
    def error(self,p):
        return 1-np.max([p,1-p])


    def plot_information_gain(self):
        # P = np.arange(0.0,1.0,0.01)
        fig = plt.figure()
        ax = plt.subplot(111)
        for i , lab, ls, c in zip([self.entropy__,self.std_entropy__,self.gini__,self.error__],
                                  ['Entropy','Entropy(scaled)','Gini Impurity','Misclassification Error'],
                                   ['-','-','--','-.'],
                                    ['black','lightgray','red','green','cyan']):
            line = ax.plot(self.P, i , label = lab, linestyle = ls, lw = 2, color = c )

        ax.legend(loc = 'upper center', bbox_to_anchor=(0.5,1.15),ncol = 3, fancybox = True, shadow = False)
        ax.axhline(y=0.5, linewidth = 1, color='k', linestyle='--')
        ax.axhline(y=1.0, linewidth=1, color='k', linestyle='--')
        plt.ylim([0,1.1])
        plt.xlabel('p(i=1)')
        plt.ylabel('Impurity Index')
        plt.show()



# if __name__=='__main__':
#     sns.set()
#     P = np.arange(0.0,1.0,0.01)
#     inf = InfGain(P)
#     inf.plot_information_gain()
#     print('Done')

