#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/4/10 18:23
# @Author  : MiaFeng
# @Site    : 
# @File    : mlp.python
# @Software: PyCharm
__author__ = 'MiaFeng'

import numpy as np
from scipy import special
import sys
import os


from ANN.MLP.preprocess import load_mnist


# three layers

class MLP(object):
    def __init__(self,n_output,n_features,n_hidden=30,
                 l1=0.0,l2=0.0,epochs=500,eta=1e-3,
                 alpha=0.0,decrease_const=0.0,shuffle=True,
                 minibatches=1,random_state=None):
        '''

        :param n_output: output shape (#nb_class,1)
        :param n_features: shape of input features
        :param n_hidden:
        :param l1:
        :param l2:
        :param epochs:
        :param eta: learning rate
        :param alpha: parameters for momentum
        :param decrease_const: decrease the adaptive learning rate
        :param shuffle:
        :param minibatches:
        :param random_state:
        '''
        np.random.seed(random_state)    # make the random number that is generated each step is same
        self.n_output = n_output
        self.n_features = n_features
        self.n_hidden = n_hidden
        self.w1,self.w2 = self._initialize_weights()
        self.l1 = l1
        self.l2 = l2
        self.epochs = epochs
        self.eta = eta
        self.alpha = alpha
        self.decrease_const = decrease_const
        self.shuffle = shuffle
        self.minibatches = minibatches

    def _initialize_weights(self):
        w1 = np.random.uniform(-1.0,1.0,
                               size=self.n_hidden*(self.n_features+1))  #bias is also included
        w1 = w1.reshape(self.n_hidden,self.n_features+1)

        w2 = np.random.uniform(-1.0,1.0,
                               size=self.n_output*(self.n_hidden+1))
        w2 = w2.reshape(self.n_output,self.n_hidden+1)

        print(w1[:,0])  # the 1st column denotes the bias
        print(w2[:, 0]) # the 1st column denotes the bias

        return w1,w2

    def _encode_labesl(selfself,y,k):
        onehot = np.zeros((k,y.shape[0]))
        for idx, val in enumerate(y):
            onehot[val,idx] = 1.0
        return onehot

    def _sigmoid(self,z):
        return special.expit(z)
        # return 1.0/(1.0+np.exp(-z))


    # the gradient of sigmoid equals to sg*(1-sg) at the specific point
    def _sigmoid_gradient(self,z):
        sg = self._sigmoid(z)
        return sg*(1-sg)

    def _add_bias_unit(self,X,how='column'):
        if(how=='column'):
            X_new = np.ones((X.shape[0],X.shape[1]+1))
            X_new[:, 1:] = X
        elif how=='row':
            X_new = np.ones(X.shape[0]+1,X.shape[1])
            X_new[1:,:] = X
        else:
            raise AttributeError('`how` must be `column` or `row`')
        return X_new

    def _feedforward(self,X,w1,w2):
        a1 = self._add_bias_unit(X,how='column')
        z2 = w1.dot(a1.T)
        a2 = self._sigmoid(z2)
        a2 = self._add_bias_unit(a2,how='row')
        z3 = w2.dot(a2)
        a3 = self._sigmoid(z3)
        return a1,z2,a2,z3,a3

    def _L2_reg(self,lambda_,w1,w2):
        return (lambda_/2.0)* (np.sum(w1[:,1:]**2)+np.sum(w2[:,1:]**2)) # the l2 norm of coefficient w without taking bias b into consideration

    def _L1_reg(self,lambda_,w1,w2):
        return (lambda_/2.0) *(np.abs(w1[:,1:]).sum()+np.abs(w2[:,1:]).sum())

    # cross_entropy plus regularization error
    # Recall the cost function of logistic regression
    def _get_cost(self,y_enc,output,w1,w2):
        '''
        calculate the cost after uopdating the weights
        cost = cross_entropy + regularization error
        :param y_enc: the real label with the shape of (#case_size,NB_CLASS)
        :param output: the prediction label with the shape of (#case_size,NB_CLASS)
        :param w1:  the weights from the 1st layer to the 2nd layer
        :param w2: the weights from the 2nd layer to the 3rd layer
        :return: cost :
        '''
        term1 = -y_enc * (np.log(output))
        term2 = (1-y_enc) * np.log(1-output)
        cost = np.sum(term1-term2)  #data cost
        L1_term = self._L1_reg(self.l1,w1,w2)
        L2_term = self._L2_reg(self.l2,w1,w2)
        cost = cost + L1_term + L2_term
        return cost

    # back propagation
    #???????????????  Q: so the bias is not updated in the whole network   ???????????
    def _get_gradient(self,a1,a2,a3,z2,y_enc,w1,w2):
        '''
        Back propagation, which is used for calculating the gradient of loss
        If the loss is defined as the summation of the square of the errors, then using chain rules, the 1st term is
        sigma3 in the following codes, that's the gradient of loss related to the variable of the output of the whole network
        :param a1:  The output of the 1st layer, which is activated
        :param a2:  The output of the 2nd layer which is activated
        :param a3:  the output of the 3rd layer
        :param z2:  the net_output of the 2nd layer without taking the bias into consideration
        :param y_enc: the onehot vector of the real labels
        :param w1:  the weight from the 1st layer to the 2nd layer
        :param w2:  the weight from the 2nd layer to the 3rd layer
        :return:
        '''
        sigma3 = a3 - y_enc # sigmoid(z2+bias) = a3
        z2 = self._add_bias_unit(z2,how='row')  # the gradient is about z2, which is sigmoid at the point z2+bias
        sigma2 = w2.T.dot(sigma3) * self._sigmoid_gradient(z2)
        sigma2 = sigma2[1:,:]
        grad1 = sigma2.dot(a1)
        grad2 = sigma3.dot(a2.T)

        # regularize,why????????????????
        # well, it is right, cause the output a3 is related to the regularization term also, which means s3 is a
        # function of (w1,w2,etc)
        # the gradient of the regularization term with respect to the weights w is the w*(lambda) , where lambda is the
        # coefficients of the regularization term
        grad1[:,1:] += w1[:,1:]*(self.l1+self.l2)
        grad2[:,1:] += w2[:,1:]*(self.l1+self.l2)

        return grad1,grad2

    def predict(self,X):
        a1,z2,a2,z3,a3 = self._feedforward(X,self.w1,self.w2)
        y_pred = np.argmax(z3,axis=0)
        return y_pred

    def fit(self,X,y,print_progress=False):
        self.cost_ = []
        X_data, y_data = X.copy(), y.copy()
        y_enc = self._encode_labesl(y,self.n_output)

        delta_w1_prev = np.zeros(self.w1.shape)
        delta_w2_prev = np.zeros(self.w2.shape)

        for i in range(self.epochs):
            # adaptive learning rate
            self.eta /= (1+self.decrease_const*i)

            if print_progress:
                sys.stderr.write('\rEpoch: %d %d'%(i+1,self.epochs))
                sys.stderr.flush()
            if self.shuffle:
                idx = np.random.permutation(y_data.shape[0])
                X_data,y_data = X_data[idx],y_data[idx]

            mini = np.array_split(range(y_data.shape[0]),self.minibatches)

            for idx in mini:
                # feedforward
                a1,z2,a2,z3,a3 = self._feedforward(X[idx],self.w1,self.w2)
                cost = self._get_cost(y_enc=y_enc[:,idx],
                                      output=a3,
                                      w1 = self.w1,
                                      w2 = self.w2)
                self.cost_.append(cost)

                # compute gradient via back propagation
                grad1,grad2 = self._get_gradient(a1 = a1, a2 = a2,
                                                 a3 = a3, z2 = z2,
                                                 y_enc=y_enc[:,idx],
                                                 w1 = self.w1,
                                                 w2 = self.w2)

                # update gradients
                delta_w1 , delta_w2 = self.eta * grad1,self.eta*grad2

                self.w1 -= delta_w1 + (self.alpha * delta_w1_prev)
                self.w2 -= delta_w2 + (self.alpha * delta_w2_prev)

                delta_w1_prev,delta_w2_prev = delta_w1,delta_w2
        return self

    # def plot_bias(self):


if __name__=='__main__':
    # X_train, y_train, X_test, y_test = loadData()
    basepath = os.getcwd()
    X_train, y_train = load_mnist(basepath+'/mnist', kind='train')
    print('Rows: %d, columns: %d' % (X_train.shape[0], X_train.shape[1]))

    X_test, y_test = load_mnist(basepath+'/mnist', kind='test')
    print('Rows: %d, columns: %d' % (X_test.shape[0], X_test.shape[1]))

    NB_CLASS = 10
    IMG_ROW,IMG_COLS = 28,28
    N_FEATURES = IMG_ROW * IMG_COLS
    HIDDEN_NEURONS = 50
    L2 = 0.1
    L1 = 0.0
    EPOCHS = 1000
    ETA = 1e-3
    DECREASE_CONST = 1e-6
    SHUFFLE = True
    MINI_BATCHES = 50
    RANDOM_STATE = 1

    nn = MLP(n_output=NB_CLASS,
             n_features=N_FEATURES,
             n_hidden=HIDDEN_NEURONS,
             l2=L2,
             l1=L1,
             epochs=EPOCHS,
             eta=ETA,
             alpha=0.001,
             decrease_const=DECREASE_CONST,
             shuffle=SHUFFLE,
             minibatches=MINI_BATCHES,
             random_state=RANDOM_STATE)

    nn.fit(X_train,y_train,print_progress=True)