#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/4/19 14:16
# @Author  : MiaFeng
# @Site    : 
# @File    : naivebayes.py
# @Software: PyCharm
__author__ = 'MiaFeng'

import numpy as np

class NaiveBayes:
    def __init__(self):
        self._creteria = 'NB'

    def _createVocabList(self,datalist):
        '''
        build a vocabulary vector
        :param datalist: origin datalist
        :return:  the list of vocabulary set
        '''
        vocabset = set([])
        for line in datalist:
            print(set(line))
            vocabset = vocabset | set(line)
        return list(vocabset)

    # the model of the vocabulary set of file
    def _setOfWords2Vec(self,vocablist,inputset):
        '''
        Mapping each words in inputset to the vacablist, 1 denotes the word appears in the vocabulary vector, 0 fot not
        :param vocablist: the defined vocabulary vector, built by function _createVocabList()
        :param inputset:  the words of file
        :return:
        '''
        outputvec = [0] * len(vocablist)
        for word in inputset:
            if word in vocablist:
                outputvec[vocablist.index(word)]=1
            else:
                print("The word: %s is not in my vocabulary!" % word)
        return outputvec

    def _bagOfWords2Vec(self,vocablist,inputSet):
        '''
        The weighted word vector, which is weighted by the frequency of one specific word
        :param vocablist: The same as function _setOfWords2Vec
        :param inputSet: The same as function _setOfWords2Vec
        :return:
        '''
        outputVec = [0]*len(vocablist)
        for word in inputSet:
            if word in vocablist:
                outputVec[vocablist.index(word)]+=1

        return outputVec

    def _trainNB(self,trainMatrix, trainLabel):
        '''
        Training and calculating the probability of each label and conditional probability
        :param trainMatrix: numpy matrix
        :param trainLabel:
        :return:
        '''

        numTrainDocs = len(trainMatrix) # #case_size
        numWords = len(trainMatrix[0])  # #features, which is the length of vocabulary dict theoretically
        pNeg = sum(trainLabel)/float(numTrainDocs)  # the probability of negative samples

        p0Num = np.ones(numWords)   # initialize the number of samples as 1 to avoiding the denomitor is zero
        p1Num = np.ones(numWords)   # the same target

        p0InAll = 2.0   # the number of labels,which is used for laplace smoother
        p1InAll = 2.0

        # update the positive and negative samples in one specific file or the whole vocabulary dict
        for i in range(numTrainDocs):
            if trainLabel[i] == 1:
                p1Num += trainMatrix[i]
                p1InAll += sum(trainMatrix[i])
            else:
                p0Num += trainMatrix[i]
                p0InAll += sum(trainMatrix[i])
        print(p1InAll)

        # log for avoiding underflow
        p0Vect = np.log(p0Num/p0InAll)
        p1Vect = np.log(p1Num/p1InAll)
        return p0Vect,p1Vect,pNeg

    def _classifyNB(self,vecSample, p0Vec, p1Vec, pNeg):
        '''
        prediction
        :param vecSample:
        :param p0Vec:
        :param p1Vec:
        :param pNeg:
        :return:
        '''

        prob_y0 = sum(vecSample * p0Vec) + np.log(1-pNeg)


