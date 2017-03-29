#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

import numpy as np
import tensorflow as tf
from TFCommon import *

class SMDLayer:
    def __init__(self, x, dim_in, dim_out, gate, learning_rate, name):
        self.leaning_rate = learning_rate
        self.epsilon = 1e-4
        self.name = name
        # with tf.variable_scope(name):
        #     self.W = tf.get_variable('W', [dim_in, dim_out], initializer=initializer)
        #     self.b = tf.get_variable('b', [dim_out], initializer=initializer)

        self.W = tf.placeholder(tf.float32, shape=[dim_in, dim_out])
        self.b = tf.placeholder(tf.float32, shape=dim_out)
        net = tf.matmul(x, self.W)
        net = tf.nn.bias_add(net, self.b)
        self.phi = gate.gate_fun(net)

        # self.alpha_W = tf.placeholder(tf.float32, shape=[dim_in, dim_out])
        # self.h_W = tf.placeholder(tf.float32, shape=[dim_in, dim_out])
        #
        # self.alpha_b = tf.placeholder(tf.float32, shape=dim_out)
        # self.h_b = tf.placeholder(tf.float32, shape=dim_out)


        self.W_var = np.random.randn(dim_in, dim_out)
        self.alpha_W = np.random.randn(dim_in, dim_out) * self.leaning_rate
        self.h_W = np.zeros((dim_in, dim_out))

        self.b_var = np.random.randn(dim_out)
        self.alpha_b = np.random.randn(dim_out) * self.leaning_rate
        self.h_b = np.zeros(dim_out)

        self.feed_dict = {self.W: self.W_var, self.b: self.b_var}

    def compute_grads(self, loss):
        self.W_grad = tf.gradients(loss, self.W)
        self.b_grad = tf.gradients(loss, self.b)

    def update(self, sess, feed_dict):
        W_grad_var = sess.run(self.W_grad, feed_dict=feed_dict)
        W_grad_var = -np.asarray(W_grad_var[0])
        alpha_W_decay = np.maximum(0.1, 1 + self.leaning_rate * W_grad_var * self.h_W)
        delta = self.h_W * self.epsilon
        feed_dict[self.W] = self.W_var + delta
        W_plus_grad = sess.run(self.W_grad, feed_dict=feed_dict)
        W_plus_grad = np.asarray(W_plus_grad[0])
        feed_dict[self.W] = self.W_var - delta
        W_minus_grad = sess.run(self.W_grad, feed_dict=feed_dict)
        W_minus_grad = np.asarray(W_minus_grad[0])
        feed_dict[self.W] = self.W_var
        hessian_W_h_W = (W_plus_grad - W_minus_grad) / (2 * self.epsilon)
        self.h_W += self.alpha_W * (W_grad_var - hessian_W_h_W)
        self.W_var += self.alpha_W * W_grad_var
        self.alpha_W *= alpha_W_decay

        b_grad_var = sess.run(self.b_grad, feed_dict=feed_dict)
        b_grad_var = -np.asarray(b_grad_var[0])
        alpha_b_decay = np.maximum(0.1, 1 + self.leaning_rate * b_grad_var * self.h_b)
        delta = self.h_b * self.epsilon
        feed_dict[self.b] = self.b_var + delta
        b_plus_grad = sess.run(self.b_grad, feed_dict=feed_dict)
        b_plus_grad = np.asarray(b_plus_grad[0])
        feed_dict[self.b] = self.b_var - delta
        b_minus_grad = sess.run(self.b_grad, feed_dict=feed_dict)
        b_minus_grad = np.asarray(b_minus_grad[0])
        feed_dict[self.b] = self.b_var
        hessian_b_h_b = (b_plus_grad - b_minus_grad) / (2 * self.epsilon)
        self.h_b += self.alpha_b * (b_grad_var - hessian_b_h_b)
        self.b_var += self.alpha_b * b_grad_var
        self.alpha_b *= alpha_b_decay

        # [[W_grad, W_var]] = optimizer.compute_gradients(loss, var_list=[self.W])
        # delta = tf.scalar_mul(self.epsilon, self.h_W)
        # tf.assign(self.W, tf.add(self.W, delta))
        # W_plus = tf.add(self.W, delta)
        # [[W_grad_plus, _]] = optimizer.compute_gradients(loss, var_list=[W_plus])
        # tf.assign(self.W, tf.subtract(self.W, tf.scalar_mul(2.0, delta)))
        # [[W_grad_minus, _]] = optimizer.compute_gradients(loss, var_list=[self.W])
        # tf.assign(self.W, tf.add(self.W, delta))
        #
        # hessian_dot_h_W = tf.scalar_mul(1.0 / (2 * self.epsilon), tf.subtract(W_grad_plus, W_grad_minus))
        # self.next_h_W = tf.add(self.h_W, tf.multiply(self.alpha_W, tf.subtract(W_grad, hessian_dot_h_W)))
        #
        # pre_exp = tf.add(1.0, tf.scalar_mul(self.leaning_rate, tf.multiply(W_grad, self.h_W)))
        # lower_bound = tf.ones(tf.shape(pre_exp))
        # alpha_decay = tf.where(pre_exp > lower_bound, pre_exp, lower_bound)
        # self.next_alpha_W = tf.multiply(self.alpha_W, alpha_decay)
        #
        # W_grad = tf.multiply(W_grad, self.alpha_W)
        # grads.append((W_grad, W_var))
        #
        # [[b_grad, b_var]] = optimizer.compute_gradients(loss, var_list=[self.b])
        # delta = tf.scalar_mul(self.epsilon, self.h_b)
        # tf.assign(self.b, tf.add(self.b, delta))
        # [[b_grad_plus, _]] = optimizer.compute_gradients(loss, var_list=[self.b])
        # tf.assign(self.b, tf.subtract(self.b, tf.scalar_mul(2.0, delta)))
        # [[b_grad_minus, _]] = optimizer.compute_gradients(loss, var_list=[self.b])
        # tf.assign(self.b, tf.add(self.b, delta))
        #
        # hessian_dot_h_b = tf.scalar_mul(1.0 / (2 * self.epsilon), tf.subtract(b_grad_plus, b_grad_minus))
        # self.next_h_b = tf.add(self.h_b, tf.multiply(self.alpha_b, tf.subtract(b_grad, hessian_dot_h_b)))
        #
        # pre_exp = tf.add(1.0, tf.scalar_mul(self.leaning_rate, tf.multiply(b_grad, self.h_b)))
        # lower_bound = tf.scalar_mul(0.1, tf.ones(tf.shape(pre_exp)))
        # alpha_decay = tf.where(pre_exp > lower_bound, pre_exp, lower_bound)
        # self.next_alpha_b = tf.multiply(self.alpha_b, alpha_decay)
        #
        # b_grad = tf.multiply(b_grad, self.alpha_b)
        # grads.append((b_grad, b_var))
        #
        # self.placeholders = [self.alpha_W, self.h_W, self.alpha_b, self.h_b]

class SMD:
    def __init__(self, dims, learning_rate, name):
        self.layers = []
        x = tf.placeholder(tf.float32, shape=(None, dims[0]))
        var_in = x
        for i in range(1, len(dims)):
            if i < len(dims) - 1:
                gate = Relu()
            else:
                gate = Identity()
            layer = SMDLayer(var_in, dims[i - 1], dims[i], gate,
                             learning_rate, '%s_layer_%d' % (name, i))
            self.layers.append(layer)
            var_in = layer.phi

        phi = var_in
        pred = tf.nn.softmax(phi)
        target = tf.placeholder(tf.float32, shape=(None, dims[-1]))
        correct_prediction = tf.equal(tf.argmax(target, 1), tf.argmax(pred, 1))
        self.correct_labels = tf.reduce_sum(tf.cast(correct_prediction, "float"))
        cross_entropy_loss = tf.nn.softmax_cross_entropy_with_logits(logits=phi, labels=target)
        # cross_entropy_loss = tf.reduce_sum(tf.squared_difference(pred, target))
        # optimizer = tf.train.GradientDescentOptimizer(learning_rate=1.0)

        self.loss = cross_entropy_loss
        self.feed_dict = dict()
        for layer in self.layers:
            self.feed_dict.update(layer.feed_dict)
            layer.compute_grads(self.loss)

        self.x = x
        self.target = target

    def train(self, sess, train_x, train_y):
        self.feed_dict[self.x] = train_x
        self.feed_dict[self.target] = train_y
        result = sess.run(self.correct_labels, feed_dict=self.feed_dict)
        for layer in self.layers:
            layer.update(sess, self.feed_dict)
        for layer in self.layers:
            self.feed_dict[layer.W] = layer.W_var
            self.feed_dict[layer.b] = layer.b_var
        return result
