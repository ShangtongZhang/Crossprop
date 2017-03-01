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
learning_rate = 0.01
dim_in = 28 * 28
dim_hidden = 1024
dim_out = 10

train_x = np.matrix(train_x.reshape([-1, dim_in])) / 255.0
train_y = np.matrix(dense_to_one_hot(train_y))
test_x = np.matrix(test_x.reshape([-1, dim_in])) / 255.0
test_y = np.matrix(dense_to_one_hot(test_y))

train_examples = 10000
test_examples = 1000
train_x = train_x[: train_examples, :]
train_y = train_y[: train_examples, :]
test_x = test_x[: test_examples, :]
test_y = test_y[: test_examples, :]

# lables = ['cp', 'bp']
lables = ['cp']
initialzer = tf.random_normal_initializer()
cp = CrossPropClassification(dim_in, dim_hidden, dim_out, learning_rate, gate=Tanh(), initializer=initialzer)
bp = BackPropClissification(dim_in, dim_hidden, dim_out, learning_rate, gate=Tanh(), initializer=initialzer)
methods = [cp, bp]
with tf.Session() as sess:
    for var in tf.global_variables():
        sess.run(var.initializer)
    for ep in range(epochs):
        cur = 0
        while cur < train_x.shape[0]:
            end = min(cur + batch_size, train_x.shape[0])
            for i in range(len(lables)):
                methods[i].train(sess, train_x[cur: end, :], train_y[cur: end, :])
            cur = end

        for i in range(len(lables)):
            loss, acc = methods[i].test(sess, test_x, test_y)
            print lables[i], 'epoch', ep, 'loss', loss, 'accuracy', acc
