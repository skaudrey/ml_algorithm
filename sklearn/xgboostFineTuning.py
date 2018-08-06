# encoding: utf-8
#!/usr/bin/env python
'''
@Author: Mia
@Contact: fengmiao@meituan.com
@Software: PyCharm
@Site    : 
@Time    : 2018/7/31 下午9:27
@File    : xgboostFineTuning.py
@Theme   :
'''

from sklearn import datasets

def loadData():
    diabetes = datasets.load_diabetes()
    X ,y = diabetes.data, diabetes.target

