#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

import tensorflow as tf
import numpy as np
from TFCommon import *

class BackPropRegression:
    def __init__(self, dim_in, dim_hidden, learning_rate, gate=Relu(),
                 initializer=tf.random_normal_initializer(), optimizer=None):
        dim_out = 1
        if optimizer is None:
            optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        self.x = tf.placeholder(tf.float32, shape=(None, dim_in))
        self.target = tf.placeholder(tf.float32, shape=(None, dim_out))
        _, __, ___, phi = \
            fully_connected('backprop', 'fully_connected_layer1', self.x, dim_in, dim_hidden, initializer, gate.gate_fun)
        _, __, ___, y = \
            fully_connected('backprop', 'fully_connected_layer2', phi, dim_hidden, dim_out, initializer, tf.identity)
        self.loss = tf.scalar_mul(0.5, tf.reduce_mean(tf.squared_difference(y, self.target)))
        self.all_gradients = optimizer.compute_gradients(self.loss)
        self.train_op = optimizer.apply_gradients(self.all_gradients)

    def train(self, sess, train_x, train_y):
        _, loss = sess.run([self.train_op, self.loss], feed_dict={
            self.x: train_x,
            self.target: train_y
        })
        return loss

    def test(self, sess, test_x, test_y):
        return sess.run(self.loss, feed_dict={
            self.x: test_x,
            self.target: test_y
        })

class BackPropClissification:
    def __init__(self, dim_in, dim_hidden, dim_out, learning_rate, gate=Relu(),
                 initializer=tf.random_normal_initializer(), bottom_layer=None, optimizer=None):
        if bottom_layer is None:
            self.x = tf.placeholder(tf.float32, shape=(None, dim_in))
            var_in = self.x
        else:
            self.x = bottom_layer.x
            var_in = bottom_layer.var_out

        if optimizer is None:
            optimizer = tf.train.GradientDescentOptimizer(learning_rate)

        self.target = tf.placeholder(tf.float32, shape=(None, dim_out))
        U, __, ___, phi = \
            fully_connected('backprop', 'fully_connected_layer1', var_in, dim_in, dim_hidden, initializer, gate.gate_fun)
        W, __, ___, y = \
            fully_connected('backprop', 'fully_connected_layer2', phi, dim_hidden, dim_out, initializer, tf.identity)
        self.pred = tf.nn.softmax(y)
        ce_loss = tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=self.target)
        self.loss = tf.reduce_mean(ce_loss)
        self.total_loss = tf.reduce_sum(ce_loss)
        correct_prediction = tf.equal(tf.argmax(self.target, 1), tf.argmax(y, 1))
        self.correct_labels = tf.reduce_sum(tf.cast(correct_prediction, "float"))
        self.all_gradients = optimizer.compute_gradients(self.loss)
        self.train_op = optimizer.apply_gradients(self.all_gradients)
        self.other_info = [self.total_loss, self.correct_labels]
        self.U = U
        self.W = W

    def train(self, sess, train_x, train_y):
        _, total_loss, correct_labels = \
            sess.run([self.train_op, self.total_loss, self.correct_labels],
                     feed_dict={
                         self.x: train_x,
                         self.target: train_y
                     })
        return total_loss, correct_labels

    def test(self, sess, test_x, test_y):
        return sess.run([self.total_loss, self.correct_labels], feed_dict={
            self.x: test_x,
            self.target: test_y
        })
