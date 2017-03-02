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

train_x, train_y = load_mnist('training')
test_x, test_y = load_mnist('testing')

epochs = 200
batch_size = 1
learning_rate = 0.0001
dim_in = 28 * 28
dim_hidden = 1024
dim_out = 10

train_x = np.matrix(train_x.reshape([-1, dim_in])) / 255.0
train_y = np.matrix(dense_to_one_hot(train_y))
test_x = np.matrix(test_x.reshape([-1, dim_in])) / 255.0
test_y = np.matrix(dense_to_one_hot(test_y))

train_examples = 1000
test_examples = 100
train_x = train_x[: train_examples, :]
train_y = train_y[: train_examples, :]
test_x = test_x[: test_examples, :]
test_y = test_y[: test_examples, :]

# labels = ['cp', 'bp']
# labels = ['cp']
# initialzer = tf.random_normal_initializer(seed=0)
# # initialzer = tf.zeros_initializer()
# cp = CrossPropClassification(dim_in, dim_hidden, dim_out, learning_rate, gate=Tanh(), initializer=initialzer)
# bp = BackPropClissification(dim_in, dim_hidden, dim_out, learning_rate, gate=Tanh(), initializer=initialzer)
# methods = [cp, bp]
# with tf.Session() as sess:
#     for var in tf.global_variables():
#         sess.run(var.initializer)
#     for ep in range(epochs):
#         cur = 0
#         while cur < train_x.shape[0]:
#             end = min(cur + batch_size, train_x.shape[0])
#             for i in range(len(labels)):
#                 methods[i].train(sess, train_x[cur: end, :], train_y[cur: end, :])
#             cur = end
#
#         for i in range(len(labels)):
#             loss, acc = methods[i].test(sess, test_x, test_y)
#             print labels[i], 'epoch', ep, 'loss', loss, 'accuracy', acc

input_dim = dim_in + 1
output_units = dim_out
cp = Crossprop_Alternate(input_dim, dim_hidden + 1, output_units,
                         alpha_step_size=learning_rate, beta_step_size=learning_rate)
train_x = np.asarray(train_x)
train_y = np.asarray(train_y)
test_x = np.asarray(test_x)
test_y = np.asarray(test_y)
for ep in range(epochs):
    training_indices = np.arange(train_examples)
    for example_i in training_indices:
        tx = np.concatenate((train_x[example_i, :].flatten(), [1]))
        input_vector = tx.reshape(input_dim, 1)
        cp.set_input_vector(input_vector)
    training_indices = np.arange(train_examples)
    for example_i in training_indices:
        tx = np.concatenate((train_x[example_i, :].flatten(), [1]))
        input_vector = tx.reshape(input_dim, 1)
        cp.set_input_vector(input_vector)
        true_label_encoded = np.zeros((output_units, 1))
        true_label_encoded[np.argmax(train_y[example_i, :])] = 1
        cp_estimate = cp.make_estimate()
        cp.crosspropagate_errors(true_label_encoded)

    cp_error = 0.0
    acc = 1.0
    testing_indices = np.arange(test_examples)
    for ex_i in testing_indices:
        tx = np.concatenate((test_x[ex_i, :].flatten(), [1]))
        input_vector = tx.reshape(input_dim, 1)
        cp.set_input_vector(input_vector)
        true_label_encoded = np.zeros((output_units, 1))
        true_label_encoded[np.argmax(train_y[ex_i, :])] = 1
        cp_estimate = cp.make_estimate()
        if (np.argmax(cp_estimate) == np.argmax(true_label_encoded)):
            acc += 1
        cp_error += -np.sum(np.multiply(np.log(cp_estimate), true_label_encoded))
    cp_error /= test_examples
    acc /= test_examples
    print 'epoch', ep,  cp_error, acc