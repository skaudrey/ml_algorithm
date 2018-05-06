#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/5/6 22:13
# @Author  : MiaFeng
# @Site    : 
# @File    : LeNet-tf.py
# @Software: PyCharm
__author__ = 'MiaFeng'


import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)
sess = tf.InteractiveSession()

def weight_varaible(shape):
    initial = tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(initial)

def bias_varaible(shape):
    initial = tf.constant(0.1,shape=shape)
    return tf.Variable(initial)

def con2d(x,W):
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

x = tf.placeholder(tf.float32,[None,784])
y_ = tf.placeholder(tf.float32,[None,10])

x_img = tf.reshape(x,[-1,28,28,1])

# Layer Conv1
W_conv1 = weight_varaible([5,5,1,32])
b_conv1 = bias_varaible([32])
h_conv1 = tf.nn.relu(con2d(x_img, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# Layer Conv2
W_conv2 = weight_varaible([5,5,32,64])
b_conv2 = bias_varaible([64])
h_conv2 = tf.nn.relu(con2d(h_pool1,W_conv2)+b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# Layer FC
W_fc1 = weight_varaible([7*7*64,1024])
b_fc1 = bias_varaible([1024])
h_pool2_flat = tf.reshape(h_pool2,[-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat,W_fc1)+b_fc1)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1,keep_prob)

W_fc2 = weight_varaible([1024,10])
b_fc2 = bias_varaible([10])
y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop,W_fc2) + b_fc2)

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv),reduction_indices=[1]))
train_step = tf.train.AdadeltaOptimizer(1e-4).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y_conv,1),tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

tf.global_variables_initializer().run()
for i in range(20000):
    batch = mnist.train.next_batch(50)
    if i % 1000 == 0:
        train_accuracy = accuracy.eval(feed_dict={x:batch[0],y_:batch[1],keep_prob:1.0})
        print("Step %d, training accuracy %g" % (i, train_accuracy))
    train_step.run(feed_dict={x:batch[0],y_:batch[1],keep_prob:0.5})

print("test accuracy %g" %(accuracy.eval(feed_dict = {x:mnist.test.images, y_:mnist.test.labels, keep_prob:1.0})))

