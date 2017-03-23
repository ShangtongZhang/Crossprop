#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

import numpy as np
import tensorflow as tf
from TFCommon import *

class SMDLayer:
    def __init__(self, x, dim_in, dim_out, gate,
                 initializer, learning_rate, name):
        self.leaning_rate = learning_rate
        self.name = name
        with tf.variable_scope(name):
            W = tf.get_variable('W', [dim_in, dim_out], initializer=initializer)
            b = tf.get_variable('b', [dim_out], initializer=initializer)

        net = tf.matmul(x, W)
        net = tf.nn.bias_add(net, b)
        self.phi = gate.gate_fun(net)

        self.W = W
        self.b = b

        self.alpha_W = tf.placeholder(tf.float32, shape=[dim_in, dim_out])
        self.h_W = tf.placeholder(tf.float32, shape=[dim_in, dim_out])

        self.alpha_b = tf.placeholder(tf.float32, shape=dim_out)
        self.h_b = tf.placeholder(tf.float32, shape=dim_out)

    def backprop(self, loss, optimizer, grads):
        [W_grad, W_var] = optimizer.compute_gradients(loss, var_list=[self.W])
        W_grad = tf.multiply(W_grad, self.alpha_W)
        grads.append((W_grad, W_var))

        pre_exp = tf.add(1.0, tf.scalar_mul(self.leaning_rate, tf.multiply(W_grad, self.h_W)))
        lower_bound = tf.ones(tf.shape(pre_exp))
        alpha_decay = tf.where(pre_exp > lower_bound, pre_exp, lower_bound)
        self.next_alpha_W = tf.multiply(self.alpha_W, alpha_decay)

        hessian = lambda _, w: tf.transpose(tf.hessians(loss, w))

        hessians = tf.scan(hessian, self.W)

        h_W_hessian = tf.reshape(tf.transpose(self.h_W), shape=[tf.shape(hessians)[0], -1, 1])
        h_W_hessian = tf.multiply(h_W_hessian, hessians)
        h_W_hessian = tf.reduce_sum(h_W_hessian, axis=1)

        self.next_h_W = self.h_W + tf.multiply(self.alpha_W, tf.subtract(W_grad, h_W_hessian))

        [b_grad, b_var] = optimizer.compute_gradients(loss, var_list=[self.b])
        b_grad = tf.multiply(b_grad, self.alpha_b)
        grads.append((b_grad, b_var))

        pre_exp = tf.add(1.0, tf.scalar_mul(self.leaning_rate, tf.multiply(b_grad, self.h_b)))
        lower_bound = tf.ones(tf.shape(pre_exp))
        alpha_decay = tf.where(pre_exp > lower_bound, pre_exp, lower_bound)
        self.next_alpha_b = tf.multiply(self.alpha_b, alpha_decay)

        hessians = hessian(None, self.b)
        h_b_hession = tf.matmul(self.h_b, hessians)

        self.next_h_b = self.h_b + tf.multiply(self.alpha_b, tf.subtract(b_grad, h_b_hession))

        self.placeholders = [self.alpha_W, self.h_W, self.alpha_b, self.h_b]

class SMD:
    def __init__(self, dims, initializer, learning_rate, name):
        self.layers = []
        x = tf.placeholder(tf.float32, shape=(None, dims[0]))
        var_in = x
        for i in range(1, len(dims)):
            if i < len(dims) - 1:
                gate = Relu()
            else:
                gate = Identity()
            layer = SMDLayer(var_in, dims[i - 1], dims[i], gate, initializer,
                             learning_rate, '%s_layer_%d' % (name, i))
            self.layers.append(layer)
            var_in = layer.phi

        phi = self.layers[-1].phi
        pred = tf.nn.softmax(phi)
        target = tf.placeholder(tf.float32, shape=(None, dims[-1]))
        correct_prediction = tf.equal(tf.argmax(target, 1), tf.argmax(pred, 1))
        self.correct_labels = tf.reduce_sum(tf.cast(correct_prediction, "float"))
        cross_entropy_loss = tf.nn.softmax_cross_entropy_with_logits(logits=phi, labels=target)

        optimizer = tf.train.GradientDescentOptimizer(learning_rate=1.0)

        layers_placeholder_var = []
        for i in range(1, len(dims)):
            layers_placeholder_var.append([np.zeros((dims[i - 1], dims[i])),
                                           np.zeros((dims[i - 1], dims[i])),
                                           np.zeros(dims[i]),
                                           np.zeros(dims[i])])
        grads = []
        self.feed_dict = dict()
        self.layers_output = []
        for i in range(1, len(dims)):
            layer = self.layers[i - 1]
            layer.backprop(cross_entropy_loss, optimizer, grads)
            for j in range(len(layers_placeholder_var[i - 1])):
                self.feed_dict[layer.placeholders[j]] = layers_placeholder_var[i - 1][j]

            self.layers_output.extend([layer.next_alpha_W, layer.next_h_W,
                                       layer.next_alpha_b, layer.next_h_b])

        self.layers_placeholder_var = layers_placeholder_var
        self.train_op = optimizer.apply_gradients(grads)
        self.x = x
        self.target = target

    def train(self, sess, train_x, train_y):
        self.feed_dict[self.x] = train_x
        self.feed_dict[self.target] = train_y
        result = sess.run([self.train_op, self.correct_labels] + self.layers_output,
                          feed_dict=self.feed_dict)
        for i in range(len(self.layers_placeholder_var)):
            self.layers_placeholder_var[i][:] = result[i + 2]
        return result[1]
