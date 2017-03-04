#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

import tensorflow as tf
import numpy as np
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import nn_ops
from TFCommon import *

class BackPropRegression:
    def __init__(self, dim_in, dim_hidden, learning_rate, gate=Relu(),
                 initializer=tf.random_normal_initializer()):
        dim_out = 1
        self.x = tf.placeholder(tf.float32, shape=(None, dim_in))
        self.target = tf.placeholder(tf.float32, shape=(None, dim_out))
        _, __, ___, phi = \
            fully_connected('fully_connected_layer1', self.x, dim_in, dim_hidden, initializer, gate.gate_fun)
        _, __, ___, y = \
            fully_connected('fully_connected_layer2', phi, dim_hidden, dim_out, initializer, tf.identity)
        self.loss = tf.scalar_mul(0.5, tf.reduce_mean(tf.squared_difference(y, self.target)))
        self.train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(self.loss)

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
                 initializer=tf.random_normal_initializer(), bottom_layer=None):
        if bottom_layer is None:
            self.x = tf.placeholder(tf.float32, shape=(None, dim_in))
            var_in = self.x
        else:
            self.x = bottom_layer.x
            var_in = bottom_layer.var_out

        self.target = tf.placeholder(tf.float32, shape=(None, dim_out))
        _, __, ___, phi = \
            fully_connected('fully_connected_layer1', var_in, dim_in, dim_hidden, initializer, gate.gate_fun)
        _, __, ___, y = \
            fully_connected('fully_connected_layer2', phi, dim_hidden, dim_out, initializer, tf.identity)
        self.pred = tf.nn.softmax(y)
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=self.target))
        correct_prediction = tf.equal(tf.argmax(self.target, 1), tf.argmax(self.pred, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        self.train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(self.loss)

    def train(self, sess, train_x, train_y):
        _, loss, accuracy, pred = \
            sess.run([self.train_op, self.loss, self.accuracy, self.pred],
                     feed_dict={
                         self.x: train_x,
                         self.target: train_y
                     })
        return loss, accuracy

    def test(self, sess, test_x, test_y):
        return sess.run([self.loss, self.accuracy], feed_dict={
            self.x: test_x,
            self.target: test_y
        })
