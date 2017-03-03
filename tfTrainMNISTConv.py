#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

import numpy as np
import tensorflow as tf
import pickle
from TFCrossprop import *
from TFBackprop import *
from load_mnist import *

train_x, train_y = load_mnist('training')
test_x, test_y = load_mnist('testing')

epochs = 200
batch_size = 100
learning_rate = 0.0001
dim_hidden = 1024
dim_out = 10

train_x = train_x.reshape([-1, 28, 28, 1]) / 255.0
train_y = dense_to_one_hot(train_y)
test_x = test_x.reshape([-1, 28, 28, 1]) / 255.0
test_y = dense_to_one_hot(test_y)

train_examples = 10000
test_examples = 100
train_x = train_x[: train_examples, :, :, :]
train_y = train_y[: train_examples, :]
test_x = test_x[: test_examples, :, :, :]
test_y = test_y[: test_examples, :]

_, width, height, _ = train_x.shape
depth_0 = 1 # channels
filter_1 = 5
depth_1 = 64
filter_2 = 5
depth_2 = 64

dim_in = width * height * depth_2 // 16
dims = [width, height, depth_0, filter_1, depth_1, filter_2, depth_2]

gate = Tanh()
initialzer = tf.random_normal_initializer()
labels = ['cp', 'bp']

cpConvLayers = ConvLayers('cp-conv', dims, gate, initialzer)
cp = CrossPropClassification(dim_in, dim_hidden, dim_out, learning_rate,
                             gate=gate, initializer=initialzer, bottom_layer=cpConvLayers)
bpConvLayers = ConvLayers('bp-conv', dims, gate, initialzer)
bp = BackPropClissification(dim_in, dim_hidden, dim_out, learning_rate,
                            gate=gate, initializer=initialzer, bottom_layer=bpConvLayers)
methods = [cp, bp]
with tf.Session() as sess:
    for var in tf.global_variables():
        sess.run(var.initializer)
    for ep in range(epochs):
        cur = 0
        total_loss = 0.0
        while cur < train_x.shape[0]:
            end = min(cur + batch_size, train_x.shape[0])
            for i in range(len(labels)):
                loss, _ = methods[i].train(sess, train_x[cur: end, :, :, :], train_y[cur: end, :])
                total_loss += loss
            cur = end

        for i in range(len(labels)):
            loss, acc = methods[i].test(sess, test_x, test_y)
            print labels[i], 'epoch', ep, 'loss', loss, 'accuracy', acc
