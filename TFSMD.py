#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

import numpy as np
import tensorflow as tf
from TFCommon import *

class TFSMDLayer:
    def __init__(self, x, dim_in, dim_out, gate, learning_rate, initializer, name):
        self.leaning_rate = learning_rate
        # self.epsilon = 1e-4
        self.epsilon = 0.1
        self.name = name
        with tf.variable_scope(name):
            self.W = tf.get_variable('W', [dim_in, dim_out], initializer=initializer)
            self.alpha_W = tf.get_variable('alpha_W', [dim_in, dim_out],
                                           initializer=tf.random_normal_initializer(
                                               mean=0.0, stddev=learning_rate, seed=0
                                                # mean=0.0, stddev=learning_rate
                                           ))
            self.alpha_W_decay = tf.get_variable('alpha_W_decay', [dim_in, dim_out],
                                                 initializer=tf.zeros_initializer())
            # self.h_W = tf.get_variable('h_W', [dim_in, dim_out], initializer=tf.zeros_initializer())
            self.h_W = tf.get_variable('h_W', [dim_in, dim_out], initializer=initializer)
            self.h_W_delta = tf.get_variable('h_W_delta', [dim_in, dim_out],
                                             initializer=tf.zeros_initializer())
            self.b = tf.get_variable('b', [dim_out], initializer=initializer)
            self.alpha_b = tf.get_variable('alpha_b', [dim_out],
                                           initializer=tf.random_normal_initializer(
                                               mean=0.0, stddev=learning_rate, seed=0
                                                # mean=0.0, stddev=learning_rate
                                           ))
            self.alpha_b_decay = tf.get_variable('alpha_b_decay', [dim_out],
                                                 initializer=tf.zeros_initializer())
            # self.h_b = tf.get_variable('h_b', [dim_out], initializer=tf.zeros_initializer())
            self.h_b = tf.get_variable('h_b', [dim_out], initializer=initializer)
            self.h_b_delta = tf.get_variable('h_b_delta', [dim_out],
                                             initializer=tf.zeros_initializer())

        self.hessian_W_h_W = tf.placeholder(tf.float32, [dim_in, dim_out])
        self.hessian_b_h_b = tf.placeholder(tf.float32, [dim_out])

        net = tf.matmul(x, self.W)
        net = tf.nn.bias_add(net, self.b)
        self.phi = gate.gate_fun(net)

        delta_W = self.epsilon * self.h_W
        W_plus = self.W + delta_W
        W_minus = self.W - 2 * delta_W
        W_restore = self.W + delta_W

        self.W_plus_op = self.W.assign(W_plus)
        self.W_minus_op = self.W.assign(W_minus)
        self.W_restore_op = self.W.assign(W_restore)

        delta_b = self.epsilon * self.h_b
        b_plus = self.b + delta_b
        b_minus = self.b - 2 * delta_b
        b_resotre = self.b + delta_b

        self.b_plus_op = self.b.assign(b_plus)
        self.b_minus_op = self.b.assign(b_minus)
        self.b_restore_op = self.b.assign(b_resotre)

    def backprop(self, optimizer, loss):
        [[self.W_grad, W_var]] = optimizer.compute_gradients(loss, var_list=[self.W])
        scaled_W_grad = tf.multiply(self.W_grad, self.alpha_W)
        pre_exp_W = 1.0 + self.leaning_rate * tf.multiply(-self.W_grad, self.h_W)
        lower_bound_W = tf.ones(tf.shape(pre_exp_W)) * 0.1

        self.pre_update_alpha_W_op = \
            self.alpha_W_decay.assign(tf.where(pre_exp_W > lower_bound_W, pre_exp_W, lower_bound_W))
        self.pre_update_h_W_op = \
            self.h_W_delta.assign(tf.multiply(self.alpha_W, -self.W_grad - self.hessian_W_h_W))
        # self.update_W_op = optimizer.apply_gradients([(scaled_W_grad, W_var)])
        self.update_W_op = self.W.assign(self.W - scaled_W_grad)
        self.update_h_W_op = self.h_W.assign(self.h_W + self.h_W_delta)
        self.update_alpha_W_op = self.alpha_W.assign(tf.multiply(self.alpha_W_decay, self.alpha_W))

        [[self.b_grad, b_var]] = optimizer.compute_gradients(loss, var_list=[self.b])
        scaled_b_grad = tf.multiply(self.b_grad, self.alpha_b)
        pre_exp_b = 1.0 + self.leaning_rate * tf.multiply(-self.b_grad, self.h_b)
        lower_bound_b = tf.ones(tf.shape(pre_exp_b)) * 0.1

        self.pre_update_alpha_b_op = \
            self.alpha_b_decay.assign(tf.where(pre_exp_b > lower_bound_b, pre_exp_b, lower_bound_b))
        self.pre_update_h_b_op = \
            self.h_b_delta.assign(tf.multiply(self.alpha_b, -self.b_grad - self.hessian_b_h_b))
        # self.update_b_op = optimizer.apply_gradients([(scaled_b_grad, b_var)])
        self.update_b_op = self.b.assign(self.b - scaled_b_grad)
        self.update_h_b_op = self.h_b.assign(self.h_b + self.h_b_delta)
        self.update_alpha_b_op = self.alpha_b.assign(tf.multiply(self.alpha_b_decay, self.alpha_b))

    def update(self, sess, feed_dict):
        # print sess.run(self.W)
        _, W_plus = sess.run([self.W_plus_op, self.W_grad], feed_dict=feed_dict)
        # print sess.run(self.W)
        _, W_minus = sess.run([self.W_minus_op, self.W_grad], feed_dict=feed_dict)
        # print sess.run(self.W)
        sess.run(self.W_restore_op, feed_dict=feed_dict)
        # print sess.run(self.W)
        hessian_W_h_W = (W_plus - W_minus) / (2 * self.epsilon)
        _, b_plus = sess.run([self.b_plus_op, self.b_grad], feed_dict=feed_dict)
        _, b_minus = sess.run([self.b_minus_op, self.b_grad], feed_dict=feed_dict)
        sess.run(self.b_restore_op, feed_dict=feed_dict)
        hessian_b_h_b = (b_plus - b_minus) / (2 * self.epsilon)
        feed_dict[self.hessian_W_h_W] = hessian_W_h_W
        feed_dict[self.hessian_b_h_b] = hessian_b_h_b
        # print 'W', sess.run(self.W)
        # print 'alpha_W', sess.run(self.alpha_W)
        # print 'grad_W', sess.run(self.W_grad, feed_dict=feed_dict)
        # print 'alpha_W', sess.run(self.alpha_W)
        # sess.run(self.pre_update_alpha_W_op, feed_dict=feed_dict)
        # sess.run(self.pre_update_h_W_op, feed_dict=feed_dict)
        # sess.run(self.update_h_W_op, feed_dict=feed_dict)
        # sess.run(self.update_W_op, feed_dict=feed_dict)
        # sess.run(self.update_alpha_W_op, feed_dict=feed_dict)
        # sess.run(self.pre_update_alpha_b_op, feed_dict=feed_dict)
        # sess.run(self.pre_update_h_b_op, feed_dict=feed_dict)
        # sess.run(self.update_h_b_op, feed_dict=feed_dict)
        # sess.run(self.update_b_op, feed_dict=feed_dict)
        # sess.run(self.update_alpha_b_op, feed_dict=feed_dict)
        ops = [self.pre_update_alpha_W_op, self.pre_update_h_W_op,
               self.update_h_W_op, self.update_W_op, self.update_alpha_W_op,
               self.pre_update_alpha_b_op, self.pre_update_h_b_op,
               self.update_h_b_op, self.update_b_op, self.update_alpha_b_op]
        for op in ops:
            sess.run(op, feed_dict=feed_dict)
        # print 'grad_W', sess.run(self.W_grad, feed_dict=feed_dict)
        print 'alpha_W', sess.run(self.alpha_W)
        # print 'h_W_delta', sess.run(self.h_W_delta)
        # print 'alpha_W_decay', sess.run(self.alpha_W_decay)
        # print 'alpha_W', sess.run(self.alpha_W)
        # print 'h_W_delta', sess.run(self.h_W_delta)
        print 'h_W', sess.run(self.h_W)
        print 'W', sess.run(self.W)
        print ''

class TFSMD:
    def __init__(self, dims, learning_rate, initializer, gate_type, name):
        self.layers = []
        x = tf.placeholder(tf.float32, shape=(None, dims[0]))
        var_in = x
        for i in range(1, len(dims)):
            if i < len(dims) - 1:
                gate = gate_type()
            else:
                gate = Identity()
            layer = TFSMDLayer(var_in, dims[i - 1], dims[i], gate,
                               learning_rate, initializer, '%s_layer_%d' % (name, i))
            self.layers.append(layer)
            var_in = layer.phi

        phi = var_in
        pred = tf.nn.softmax(phi)
        target = tf.placeholder(tf.float32, shape=(None, dims[-1]))
        correct_prediction = tf.equal(tf.argmax(target, 1), tf.argmax(pred, 1))
        self.correct_labels = tf.reduce_sum(tf.cast(correct_prediction, "float"))
        cross_entropy_loss = tf.nn.softmax_cross_entropy_with_logits(logits=phi, labels=target)
        # cross_entropy_loss = tf.reduce_sum(tf.squared_difference(pred, target))
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=1.0)

        self.ce_loss = cross_entropy_loss
        self.loss = tf.reduce_mean(cross_entropy_loss)
        for layer in self.layers:
            layer.backprop(optimizer, self.loss)

        self.phi = phi
        self.pred = pred
        self.x = x
        self.target = target

    def train(self, sess, train_x, train_y):
        feed_dict = dict({
            self.x: train_x,
            self.target: train_y
        })
        # l = self.layers[0]
        # print 'x', train_x
        # print 'y', train_y
        # print 'W', sess.run(l.W)
        # print 'b', sess.run(l.b)
        # print 'alpha_W', sess.run(l.alpha_W)
        # print 'h_W', sess.run(l.h_W)
        # print 'alpha_b', sess.run(l.alpha_b)
        # print 'h_b', sess.run(l.h_b)
        # print 'phi', sess.run(self.phi, feed_dict=feed_dict)
        # print 'pred', sess.run(self.pred, feed_dict=feed_dict)
        # print 'ce_loss', sess.run(self.ce_loss, feed_dict=feed_dict)
        result = sess.run(self.correct_labels, feed_dict=feed_dict)
        # print result
        # print sess.run(self.pred, feed_dict=feed_dict)
        # print train_y
        for layer in self.layers:
            layer.update(sess, feed_dict)
        return result
