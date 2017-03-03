#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

import numpy as np
import tensorflow as tf
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import nn_ops


def fully_connected(name, var_in, dim_in, dim_out,
                    initializer, transfer, reuse=False):
    """Standard fully connected layer"""
    with variable_scope.variable_scope(name, reuse=reuse):
        if reuse:
            W = variable_scope.get_variable("W", [dim_in, dim_out])
            b = variable_scope.get_variable("b", [dim_out])
        else:  # new
            W = variable_scope.get_variable("W", [dim_in, dim_out],
                                            initializer=initializer)
            b = variable_scope.get_variable("b", [dim_out],
                                            initializer=initializer)
    z_hat = math_ops.matmul(var_in, W)
    z_hat = nn_ops.bias_add(z_hat, b)
    y_hat = transfer(z_hat)
    return W, b, z_hat, y_hat


def crossprop_layer(name, var_in, dim_in, dim_hidden, dim_out, gate_fun, initializer, reuse=False):
    with variable_scope.variable_scope(name, reuse=False):
        if reuse:
            U = variable_scope.get_variable('U', [dim_in, dim_hidden])
            b_hidden = variable_scope.get_variable('b_hidden', [dim_hidden])
            W = variable_scope.get_variable('W', [dim_hidden, dim_out])
            b_out = variable_scope.get_variable('b_out', [dim_out])
        else:
            U = variable_scope.get_variable('U', [dim_in, dim_hidden],
                                            initializer=initializer)
            b_hidden = variable_scope.get_variable('b_hidden', [dim_hidden],
                                                   initializer=initializer)
            W = variable_scope.get_variable('W', [dim_hidden, dim_out],
                                            initializer=initializer)
            b_out = variable_scope.get_variable('b_out', [dim_out],
                                                initializer=initializer)
        net = math_ops.matmul(var_in, U)
        net = nn_ops.bias_add(net, b_hidden)
        phi = gate_fun(net)
        y = math_ops.matmul(phi, W)
        y = nn_ops.bias_add(y, b_out)
        return U, b_hidden, net, phi, W, b_out, y


def convolution_2d(name, label, var_in, f, dim_in, dim_out,
                   initializer, transfer, reuse=False):
    """Standard convolutional layer"""
    with variable_scope.variable_scope(name, reuse=reuse):
        with variable_scope.variable_scope(label, reuse=reuse):
            if reuse:
                W = variable_scope.get_variable("W", [f, f, dim_in, dim_out])
                b = variable_scope.get_variable("b", [dim_out])
            else:  # new
                W = variable_scope.get_variable("W", [f, f, dim_in, dim_out],
                                                initializer=initializer)
                b = variable_scope.get_variable("b", [dim_out],
                                                initializer=initializer)
    z_hat = nn_ops.conv2d(var_in, W, strides=[1, 1, 1, 1], padding="SAME")
    z_hat = nn_ops.bias_add(z_hat, b)
    y_hat = transfer(z_hat)
    return W, b, z_hat, y_hat


class Relu:
    def __init__(self):
        self.gate_fun = nn_ops.relu
        self.gate_fun_gradient = \
            lambda phi, net: tf.where(net >= 0, tf.ones(tf.shape(net)), tf.zeros(tf.shape(net)))


class Tanh:
    def __init__(self):
        self.gate_fun = tf.tanh
        self.gate_fun_gradient = \
            lambda phi, net: tf.subtract(1.0, tf.pow(phi, 2))


class ConvLayers:
    def __init__(self, name, dims, gate, initializer):
        n1, n2, d0, f1, d1, f2, d2 = dims
        self.x = tf.placeholder(tf.float32, shape=(None, n1, n2, d0))  # input
        # layer 1: conv
        W_1, b_1, z_hat_1, r_hat_1 = convolution_2d(
            name, "layer_1", self.x, f1, d0, d1,
            tf.random_normal_initializer(),
            gate.gate_fun)
        # layer 1.5: pool
        s_hat_1 = tf.nn.max_pool(
            r_hat_1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
        # layer 2: conv
        W_2, b_2, z_hat_2, r_hat_2 = convolution_2d(
            name, "layer_2", s_hat_1, f2, d1, d2,
            tf.random_normal_initializer(),
            gate.gate_fun)
        # layer 2.5: pool
        s_hat_2 = tf.nn.max_pool(
            r_hat_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
        shape_2 = s_hat_2.get_shape().as_list()
        y_hat_2 = tf.reshape(s_hat_2, [-1, shape_2[1] * shape_2[2] * shape_2[3]])

        self.var_out = y_hat_2
        self.trainable_vars = [W_1, b_1, W_2, b_2]
