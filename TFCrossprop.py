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

class CrossPropRegression:
    def __init__(self, dim_in, dim_hidden, learning_rate, gate=Relu(),
                 initializer=tf.random_normal_initializer()):
        dim_out = 1
        self.learning_rate = learning_rate
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)

        self.x = tf.placeholder(tf.float32, shape=(None, dim_in))
        self.u_mom = tf.placeholder(tf.float32, shape=(dim_in, dim_hidden))
        self.b_hidden_mom = tf.placeholder(tf.float32, shape=(dim_hidden))
        self.target = tf.placeholder(tf.float32, shape=(None, dim_out))

        U, b_hidden, net, phi, W, b_out, y =\
            crossprop_layer('crossprop_layer', self.x, dim_in, dim_hidden, dim_out, gate.gate_fun, initializer)
        delta = tf.subtract(self.target, y)
        self.loss = tf.scalar_mul(0.5, tf.reduce_mean(tf.pow(delta, 2)))

        u_mom_decay = tf.subtract(1.0, -tf.scalar_mul(learning_rate, tf.pow(phi, 2)))
        u_mom_decay = tf.reshape(tf.tile(u_mom_decay, [1, tf.shape(U)[0]]), [-1, tf.shape(U)[0], tf.shape(U)[1]])
        self.u_mom_decay = tf.reduce_mean(u_mom_decay, axis=0)

        u_mom_delta = tf.matmul(tf.transpose(self.x), tf.diag(tf.reshape(delta, shape=[-1])))
        self.u_mom_delta = tf.matmul(u_mom_delta, gate.gate_fun_gradient(phi, net))
        self.u_mom_delta = tf.scalar_mul(1.0 / tf.cast(tf.shape(self.x)[0], tf.float32), self.u_mom_delta)

        b_hidden_mom_decay = tf.subtract(1.0, -tf.scalar_mul(self.learning_rate, tf.pow(phi, 2)))
        self.b_hidden_mom_decay = tf.reduce_mean(b_hidden_mom_decay, axis=0)

        b_hidden_mom_delta = tf.matmul(tf.transpose(delta), gate.gate_fun_gradient(phi, net))
        self.b_hidden_mom_delta = tf.reshape(b_hidden_mom_delta, shape=[-1])
        self.b_hidden_mom_delta = tf.scalar_mul(1.0 / tf.cast(tf.shape(self.x)[0], tf.float32), self.b_hidden_mom_delta)

        new_grads = []
        weighted_phi = tf.matmul(tf.diag(tf.reshape(delta, shape=[-1])), phi)
        new_u_grad = tf.reshape(tf.tile(weighted_phi, [1, tf.shape(U)[0]]), [-1, tf.shape(U)[0], tf.shape(U)[1]])
        new_u_grad = tf.multiply(tf.reduce_mean(new_u_grad, axis=0), self.u_mom)
        new_grads.append(new_u_grad)
        new_b_hidden_grad = tf.multiply(tf.reduce_mean(weighted_phi, axis=0), self.b_hidden_mom)
        new_grads.append(new_b_hidden_grad)

        old_grads = optimizer.compute_gradients(self.loss, var_list=[U, b_hidden])
        for i, (grad, var) in enumerate(old_grads):
            old_grads[i] = (new_grads[i], var)
        other_grads = optimizer.compute_gradients(self.loss, var_list=[W, b_out])
        self.train_op = optimizer.apply_gradients(old_grads + other_grads)

        self.u_mom_var = np.zeros((dim_in, dim_hidden))
        self.b_hidden_mom_var = np.zeros(dim_hidden)

    def train(self, sess, train_x, train_y):
        _, self.u_mom_decay_var, self.u_mom_delta_var, self.b_hidden_mom_decay_var, self.b_hidden_mom_delta_var = \
            sess.run([self.train_op, self.u_mom_decay, self.u_mom_delta, self.b_hidden_mom_decay, self.b_hidden_mom_delta],
                     feed_dict={
                         self.x: train_x,
                         self.target: train_y,
                         self.u_mom: self.u_mom_var,
                         self.b_hidden_mom: self.b_hidden_mom_var
                     })
        self.u_mom_var = np.multiply(self.u_mom_decay_var, self.u_mom_var) + self.learning_rate * self.u_mom_delta_var
        self.b_hidden_mom_var = np.multiply(self.b_hidden_mom_decay_var, self.b_hidden_mom_var) + \
                           self.learning_rate * self.b_hidden_mom_delta_var

    def test(self, sess, test_x, test_y):
        return sess.run(self.loss, feed_dict={
            self.x: test_x,
            self.target: test_y
        })

class CrossPropClassification:
    def __init__(self, dim_in, dim_hidden, dim_out, learning_rate, gate=Relu(),
                 initializer=tf.random_normal_initializer()):
        self.learning_rate = learning_rate
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)

        self.x = tf.placeholder(tf.float32, shape=(None, dim_in))
        self.h = tf.placeholder(tf.float32, shape=(dim_hidden, dim_out))

        self.target = tf.placeholder(tf.float32, shape=(None, dim_out))

        U, b_hidden, net, phi, W, b_out, y =\
            crossprop_layer('crossprop_layer', self.x, dim_in, dim_hidden, dim_out, gate.gate_fun, initializer)
        max_y = tf.reduce_max(y, axis=1, keep_dims=True)
        y = tf.subtract(y, max_y)
        exp_y = tf.exp(y)
        den_y = tf.reduce_sum(exp_y, axis=1, keep_dims=True)
        y = tf.multiply(exp_y, 1.0 / den_y)
        self.loss = -tf.reduce_mean(tf.reduce_sum(
            tf.multiply(tf.log(y), self.target), axis=1))
        delta = tf.subtract(self.target, y)

        h_decay = tf.subtract(1.0, -tf.scalar_mul(learning_rate, tf.pow(phi, 2)))
        h_decay = tf.reshape(tf.tile(h_decay, [1, tf.shape(self.h)[1]]), [-1, tf.shape(self.h)[1], tf.shape(self.h)[0]])
        h_decay = tf.transpose(h_decay, [0, 2, 1])
        self.h_decay = tf.reduce_mean(h_decay, axis=0)

        h_delta = tf.reshape(tf.tile(delta, [1, tf.shape(self.h)[0]]), [-1, tf.shape(self.h)[0], tf.shape(self.h)[1]])
        self.h_delta = tf.reduce_mean(h_delta, axis=0)

        new_grads = []
        phi_phi_grad = tf.multiply(phi, gate.gate_fun_gradient(phi, net))
        weight = tf.reshape(tf.tile(delta, [1, tf.shape(self.h)[0]]), [-1, tf.shape(self.h)[0], tf.shape(self.h)[1]])
        weight = tf.multiply(weight, tf.reshape(tf.tile(self.h, [tf.shape(self.x)[0], 1]), [-1, tf.shape(self.h)[0], tf.shape(self.h)[1]]))
        weight = tf.reduce_sum(weight, axis=2)
        phi_phi_grad = tf.multiply(phi_phi_grad, weight)
        new_u_grad = tf.matmul(tf.transpose(self.x), phi_phi_grad)
        new_u_grad = tf.scalar_mul(1.0 / tf.cast(tf.shape(self.x)[0], tf.float32), new_u_grad)
        new_grads.append(new_u_grad)

        new_b_hidden_grad = tf.reduce_mean(phi_phi_grad, axis=0)
        new_grads.append(new_b_hidden_grad)

        old_grads = optimizer.compute_gradients(self.loss, var_list=[U, b_hidden])
        for i, (grad, var) in enumerate(old_grads):
            old_grads[i] = (new_grads[i], var)
        other_grads = optimizer.compute_gradients(self.loss, var_list=[W, b_out])
        self.train_op = optimizer.apply_gradients(old_grads + other_grads)

        self.h_var = np.zeros((dim_hidden, dim_out))

    def train(self, sess, train_x, train_y):
        _, self.h_var, self.h_decay_var, self.h_delta_var = \
            sess.run([self.train_op, self.h, self.h_decay, self.h_delta],
                     feed_dict={
                         self.x: train_x,
                         self.target: train_y,
                         self.h: self.h_var
                     })
        self.h_var = np.multiply(self.h_decay_var, self.h_var) + self.learning_rate * self.h_delta_var

    def test(self, sess, test_x, test_y):
        return sess.run(self.loss, feed_dict={
            self.x: test_x,
            self.target: test_y
        })