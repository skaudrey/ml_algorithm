#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/4/16 23:36
# @Author  : MiaFeng
# @Site    : 
# @File    : LeNet-keras.py
# @Software: PyCharm
__author__ = 'MiaFeng'

from keras.datasets import mnist
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Dense,Activation,Flatten
from keras.models import Sequential
from keras.optimizers import Adam
from keras.utils import np_utils    # change to one hot
from keras import backend as K
from keras.utils.vis_utils import plot_model

from ANN.LeNet.loss_util import LossOfKeras

class CreateLeNet(object):
    def createLeNet(self,input_shape,nb_class):

        conv_layers = [
            Conv2D(filters=6,kernel_size=5,strides=1,padding='same',activation='relu',input_shape=input_shape),
            MaxPooling2D(pool_size=(2,2),strides=(2,2)),
            Conv2D(filters=16, kernel_size=5, strides=1, padding='same', activation='relu'),
            MaxPooling2D(pool_size=(2, 2), strides=(2,2)),
            Flatten()
        ]

        fc_layers = [
            Dense(120,activation='relu'),
            Dense(nb_class,activation='softmax')
        ]

        model = Sequential(conv_layers+fc_layers)

        return model

# parameters
VERBOSE = 1
IMG_ROW,IMG_COL = 28,28
NB_CLASS = 10
BATCH_SIZE = 128
OPTIMIZER = Adam()
VALIDATION_SPLIT = 0.2
INPUT_SHAPE = [IMG_ROW,IMG_COL,1]
NB_EPOCH = 20


# load mnist dataset
(X_train, Y_train),(X_test, Y_test) = mnist.load_data()
K.set_image_dim_ordering('tf')  # channel last

# normalizing dataset
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
X_train = X_train.reshape(X_train.shape[0],IMG_ROW,IMG_COL,1)
X_test = X_test.reshape(X_test.shape[0],IMG_ROW,IMG_COL,1)

print(X_train.shape[0],'train samples')
print(X_test.shape[0],'test samples')

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(Y_train,NB_CLASS)
Y_test =  np_utils.to_categorical(Y_test,NB_CLASS)

# init the optimizer and model
model = CreateLeNet.createLeNet(input_shape=INPUT_SHAPE, nb_class=NB_CLASS)
model.summary()

model.compile(loss='categorical_crossentropy', optimizer=OPTIMIZER, metrics=['accuracy'])

hist = LossOfKeras()
model.fit(X_train,Y_train,batch_size=BATCH_SIZE,epochs=NB_EPOCH,verbose=VERBOSE,
          validation_split=VALIDATION_SPLIT,callbacks=[hist])

# plot accuracy-loss
hist.loss_plot('epoch')
hist.loss_plot('batch')

# save history of the loss in each batch
with open('loss_log.txt','w') as f:
    f.write(str(hist.loss)+str(hist.acc)+str(hist.val_acc)+str(hist.val_loss))

score = model.evaluate(X_test,Y_test,verbose=VERBOSE)

print("Test score:", score[0])
print("Test accuracy:", score[1])

model.save('mymodel.h5')

plot_model(model,to_file='model.png',show_shapes=True,show_layer_names=True) # you have to install Graphviz.msi
# change the path for Method 3 in __init__.py of pydot_ng if it is necessary (if you don't install the Graphviz in deafault path)