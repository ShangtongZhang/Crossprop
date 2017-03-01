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

nSample = 3500
fr = open('GEOFF' + str(nSample) + '.bin', 'rb')
__, _, data = pickle.load(fr)
fr.close()
sep = nSample - 500

run = 0
X, Y = data[run]
train_x = np.matrix(X[: sep, : -1])
train_y = np.matrix(Y[: sep]).T
test_x = np.matrix(X[sep:, : -1])
test_y = np.matrix(Y[sep:]).T

epochs = 200
batch_size = 1
learning_rate = 0.0001
dim_in = 20
dim_hidden = 500

cp = CrossPropRegression(dim_in, dim_hidden, learning_rate)
bp = BackPropRegression(dim_in, dim_hidden, learning_rate)
with tf.Session() as sess:
    for var in tf.global_variables():
        sess.run(var.initializer)
    for ep in range(epochs):
        cur = 0
        while cur < train_x.shape[0]:
            end = min(cur + batch_size, train_x.shape[0])
            cp.train(sess, train_x[cur: end, :], train_y[cur: end, :])
            bp.train(sess, train_x[cur: end, :], train_y[cur: end, :])
            cur = end
        loss = cp.test(sess, test_x, test_y)
        print 'cp epoch', ep, loss
        loss = bp.test(sess, test_x, test_y)
        print 'bp epoch', ep, loss
