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
from CrosspropAlternate import *
import sys
import logging

train_x, train_y = load_mnist('training')
test_x, test_y = load_mnist('testing')

epochs = 10
batch_size = 100
# learning_rate = 0.0001
learning_rate = 1.0 / (2 ** 16)
dim_in = 28 * 28
dim_hidden = 1024
dim_out = 10

# file_name = 'logs/mnist_complete_hdim_' + str(dim_hidden) + '_batchsize_' + str(batch_size) + \
#             '_ss_' + str(learning_rate) + '.txt'
# logging.basicConfig(filename=file_name, level=logging.DEBUG)

train_x = np.matrix(train_x.reshape([-1, dim_in])) / 255.0
train_y = np.matrix(dense_to_one_hot(train_y))
test_x = np.matrix(test_x.reshape([-1, dim_in])) / 255.0
test_y = np.matrix(dense_to_one_hot(test_y))

train_examples = train_x.shape[0]  # 60,000
test_examples = test_x.shape[0]  # 10,000
# train_examples = 1000
# test_examples = 100
train_x = train_x[: train_examples, :]
train_y = train_y[: train_examples, :]
test_x = test_x[: test_examples, :]
test_y = test_y[: test_examples, :]

def main():
    labels = ['cp', 'bp']
    # labels = ['cp']
    initialzer = tf.random_normal_initializer()
    # initialzer = tf.ones_initializer()
    cp = CrossPropClassification(dim_in, dim_hidden, dim_out, learning_rate, gate=Tanh(), initializer=initialzer)
    bp = BackPropClissification(dim_in, dim_hidden, dim_out, learning_rate, gate=Tanh(), initializer=initialzer)
    methods = [cp, bp]

    tf.reset_default_graph()
    with tf.Session() as sess:
        for var in tf.global_variables():
            sess.run(var.initializer)
        for i in range(len(labels)):
            logging.info('method: {}'.format(labels[i]))
            for ep in range(epochs):
                logging.info('epoch: {}'.format(ep))
                cur = 0
                total_loss = 0.0
                while cur < train_x.shape[0]:
                    end = min(cur + batch_size, train_x.shape[0])
                    loss, _ = methods[i].train(sess, train_x[cur: end, :], train_y[cur: end, :])
                    total_loss += loss
                    cur = end
                # print (total_loss / train_examples) * batch_size
                logging.info('training_loss: {}'.format((total_loss / train_examples) * batch_size))

                loss, acc = methods[i].test(sess, test_x, test_y)
                # print labels[i], 'epoch', ep, 'loss', loss, 'accuracy', acc
                logging.info('testing_accuracy: {}'.format(acc))
                logging.info('testing_loss: {}'.format(loss))
        sess.close()

if __name__ == '__main__':
    main()