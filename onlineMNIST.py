#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

import numpy as np
import pickle
import logging
from load_mnist import *
from Backprop import *

tag = 'online_MNIST'
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
fh = logging.FileHandler('log/%s.txt' % tag)
fh.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)
logger.addHandler(fh)
logger.addHandler(ch)

train_x, train_y = load_mnist('training')
test_x, test_y = load_mnist('testing')

dim_in = 28 * 28

train_examples = 1000
test_examples = 100
train_x = train_x[: train_examples, :].reshape([-1, dim_in])
train_x = np.concatenate((train_x, np.ones((train_examples, 1))), axis=1)
train_y = train_y[: train_examples, :]
test_x = test_x[: test_examples, :].reshape([-1, dim_in])
test_x = np.concatenate((test_x, np.ones((test_examples, 1))), axis=1)
test_y = test_y[: test_examples, :]

train_x = np.asarray(train_x)
train_y = np.asarray(train_y)
test_x = np.asarray(test_x)
test_y = np.asarray(test_y)

dims = [dim_in, 1024, 10]
labels = ['SMD', 'BP']
window_size = 50

train_acc = np.zeros((len(labels), train_examples))

def train(learning_rate):
    init_fn = orthogonal_init
    gate_type = Relu
    smd = Backprop(dims, learning_rate, gate_type, SMDLayer, init_fn)
    bp = Backprop(dims, learning_rate, gate_type, BPLayer, init_fn)
    methods = [smd, bp]
    for train_index in range(len(train_x)):
        for method_ind in range(len(methods)):
            method = methods[method_ind]
            x = train_x[train_index, :].reshape(1, -1)
            y = train_y[train_index, :].reshape(1, -1)
            correct_labels = method.train(x, y)
            train_acc[method_ind, train_index] = correct_labels

            if train_index - window_size >= 0:
                train_acc[method_ind, train_index] = \
                    np.mean(train_acc[method_ind, train_index - window_size : train_index])
                logger.info('%s, %dth example, average accuracy %f' %
                            (labels[method_ind], train_index, train_acc[method_ind, train_index]))
            else:
                logger.info('%s, %dth example %d' %
                            (labels[method_ind], train_index, correct_labels))

train(0.0001)