#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/7/31 21:27
# @Author  : MiaFeng
# @Contact : skaudrey@163.com
# @Site    :
# @File    : xgboostFineTuning.py
# @Software: PyCharm
__author__ = 'MiaFeng'

from sklearn.datasets import load_diabetes

def loadData():
    X ,y = load_diabetes().data, load_diabetes().target



