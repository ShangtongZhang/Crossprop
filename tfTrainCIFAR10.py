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
from load_cifar10 import *
from TFAllCNN import *

train_x, train_y, test_x, test_y = load_cifar10('cifar10')

train_examples = train_x.shape[0]
test_examples = test_x.shape[0]
train_x = train_x[: train_examples, :, :, :]
train_y = train_y[: train_examples, :]
test_x = test_x[: test_examples, :, :, :]
test_y = test_y[: test_examples, :]

learning_rates = np.power(2.0, np.arange(-13, -8))

for learning_rate in learning_rates:
    dim_in = 8 * 8 * 10
    dim_hidden = 320
    dim_out = 10
    gate = Relu()
    epochs = 200
    batch_size = 200
    initialzer = tf.random_normal_initializer()
    labels = ['cp', 'bp']

    tf.reset_default_graph()
    cpAllCNN = AllCNN('cp-AllCNN', gate, initialzer)
    cp = CrossPropClassification(dim_in, dim_hidden, dim_out, learning_rate,
                                 gate=gate, initializer=initialzer, bottom_layer=cpAllCNN)
    bpAllCNN = AllCNN('bp-AllCNN', gate, initialzer)
    bp = BackPropClissification(dim_in, dim_hidden, dim_out, learning_rate,
                                gate=gate, initializer=initialzer, bottom_layer=bpAllCNN)
    methods = [cp, bp]
    train_loss = np.zeros((len(labels), epochs))
    test_loss = np.zeros((len(labels), epochs))
    test_acc = np.zeros((len(labels), epochs))
    with tf.Session() as sess:
        for var in tf.global_variables():
            sess.run(var.initializer)
        for ep in range(epochs):
            cur = 0
            while cur < train_x.shape[0]:
                end = min(cur + batch_size, train_x.shape[0])
                for i in range(len(labels)):
                    loss, _ = methods[i].train(sess, train_x[cur: end, :, :, :], train_y[cur: end, :])
                    train_loss[i, ep] += loss * (end - cur)
                    print labels[i], loss
                cur = end

            for i in range(len(labels)):
                loss, acc = methods[i].test(sess, test_x, test_y)
                test_loss[i, ep] = loss
                test_acc[i, ep] = acc
                print labels[i], 'epoch', ep, 'loss', loss, 'accuracy', acc
    train_loss /= train_examples

    fr = open('data/cifar10_AllCNN_'+str(learning_rate)+'.bin', 'wb')
    pickle.dump({'lr': learning_rate,
             'bs': batch_size,
             'stats': [train_loss, test_loss, test_acc]}, fr)
    fr.close()