#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/4/5 23:12
# @Author  : MiaFeng
# @Site    : 
# @File    : SteadyMarkovTranforMatrix.py
# @Software: PyCharm
__author__ = 'MiaFeng'

import numpy as np

'''
The steady of the markov chain transform matrix can be explained by the eigenvalues

For all the items in this matrix, they are all smaller than one,and bigger than zero.
The summation of each rows and each columns are both one.
As for the eigenvalues of it, one equals one, they other are smaller than one.
Thus, the matrix for n power will converge to a constant.

All of these can be found in the open-class of linear algebra of MIT

'''

if __name__=='__main__':
    trans_matrix = np.matrix([[0.9,0.075,0.025],[0.15,0.8,0.05],[0.25,0.25,0.5]],dtype=float)

    vector1 = np.matrix([[0.3,0.4,0.3]],dtype=float)

    # the result converge while i is bigger than 60
    for i in range(100):
        vector1 = vector1*trans_matrix
        print("epoch:%d"%(i+1))
        print(vector1)

    vector2 = np.matrix([[0.7, 0.1, 0.2]], dtype=float)

    print("================================================")
    # the result converge while i is bigger than 56
    for i in range(100):
        vector2 = vector2 * trans_matrix
        print("epoch:%d" % (i + 1))
        print(vector2)

    #  All of them converge to the same result