__author__ = 'MiaFeng'
import numpy as np
from sklearn.datasets import make_moons  #test datasets
class Logistic:
    def __init__(self):
        self.dim = 2
        self.w = np.array([1.0,1.0])
        self.b = 0
        self.eta = 0.2  # learning rate

    def sigmoid(self,x):    #not wx, but x
        return 1.0/(1+np.exp(-x))

    def logistic_regression(self,x,y,eta):
        itr = 0
        self.eta = eta
        pts_size,dim = np.shape(x)  # the quantity of the points in datasets and the dimension of features
        xpts = np.linspace(-1.5, 2.5)   # test set
        while itr<= 100:
            fx = np.dot(self.w,x.T)+self.b
            hx = self.sigmoid(fx)
            t = hx-y
            s = [[i[0]*i[1][0], i[0]*i[1][1]] for i in zip(t,x)]
            # two dimension,each dimension requires a calculation of
            gradien_w = np.sum(s,0)/pts_size
            gradien_b = np.sum(t,0)/pts_size
            self.w -= gradien_w * self.eta
            self.b -= gradien_b * self.eta

            ypts = (lr.w[0] * xpts + lr.b) / (-lr.w[1])
            # dont't know why at first,well, need an explanation
            # 0=b*1+w1x1+w2x2 (x0=1) 此处x=x1；y=x2
            # y = x2 = (b*1 + w1x1)/w2
            # that makes sense

            if itr%20==0:
                self.plot_classifation(x,y,color,xpts,ypts,self.eta,itr)
            itr += 1



    def plot_classifation(self,origin_x,origin_y,color,xpts,ypts,eta,itr):
        """plot the classification line.
                Args:
                   origin_x: The whole original dataset of input features origin_x.
                   origin_y: The whole original label of input features origin_x.
                   color: The color for plotting, either on positive class and negative class
                   xpts: The coordinates in x-axis of the decision hyper-plane
                   ypts: The coordinates in y-axis calculated by the decision hyper-plane
                   eta: learning rate of this experiment
                   itr: iteration index
               Returns:

               Raises:
               """
        data_size,dim = np.shape(origin_x)
        plt.figure()
        for i in range(data_size):
            plt.plot(origin_x[i,0],origin_x[i,1],color[y[i]]+'o')
        ylim_min = np.min(xpts)
        ylim_max = np.max(xpts)
        plt.ylim([ylim_min,ylim_max])
        plt.plot(xpts,ypts,'g-',lw = 2)
        plt.title('ets = %s, Iteration = %s \n'%(str(eta), str(itr)))
        plt.savefig('fig/p_N%s_itr%s' % (str(data_size),str(itr)),dpi=200,bbox_inches='tight')

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set()

    data_size = 250
    noise_level = 0.25
    x,y = make_moons(data_size,noise=noise_level)
    color = {0:'r',1:'b'}   # render the positive class to blue color and negative class to red color
    lr = Logistic()
    lr.logistic_regression(x,y,eta=1.2)

    # make gif
    from util.gif_util import GifUtil
    import os,sys
    path = os.getcwd()
    figPath = path+'/fig'
    print(figPath)
    figName = 'decision.gif'
    gifUtil = GifUtil.makeGif(figDir=figPath,figName=figName,method=2)
