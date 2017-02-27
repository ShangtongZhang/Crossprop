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
import pickle

def fully_connected(name, label, var_in, dim_in, dim_out,
                    initializer, transfer, reuse=False):
    with variable_scope.variable_scope(name, reuse=reuse):
        with variable_scope.variable_scope(label, reuse=reuse):
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


nSample = 3500
fr = open('GEOFF' + str(nSample) + '.bin', 'rb')
__, _, data = pickle.load(fr)
fr.close()
sep = nSample - 500

run = 0
X, Y = data[run]
trainX = np.matrix(X[: sep, : -1])
trainY = np.matrix(Y[: sep]).T
testX = np.matrix(X[sep:, : -1])
testY = np.matrix(Y[sep:]).T

def relu_gradient(phi, net):
    return tf.where(net >= 0, tf.ones(tf.shape(net)), tf.zeros(tf.shape(net)))

epochs = 200
batch_size = 100
learning_rate = 0.0001
dim_in = 20
dim_hidden = 500
dim_out = 1
gate_fun = nn_ops.relu
# initializer = tf.random_normal_initializer()
initializer = tf.zeros_initializer()
loss_fun = lambda x, y: tf.scalar_mul(0.5, tf.squared_difference(x, y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
x = tf.placeholder(tf.float32, shape=(None, dim_in))
u_mom = tf.placeholder(tf.float32, shape=(dim_in, dim_hidden))
b_hidden_mom = tf.placeholder(tf.float32, shape=(dim_hidden))
target = tf.placeholder(tf.float32, shape=(None, dim_out))

U, b_hidden, net, phi, W, b_out, y = crossprop_layer('crossprop_layer', x, dim_in, dim_hidden, dim_out, gate_fun,
                                                     initializer)
delta = tf.subtract(target, y)
loss = tf.scalar_mul(0.5, tf.reduce_mean(tf.pow(delta, 2)))

u_mom_decay = tf.subtract(1.0, -tf.scalar_mul(learning_rate, tf.pow(phi, 2)))
u_mom_decay = tf.reshape(tf.tile(u_mom_decay, [1, tf.shape(U)[0]]), [-1, tf.shape(U)[0], tf.shape(U)[1]])
u_mom_decay = tf.reduce_mean(u_mom_decay, axis=0)
u_mom_delta = tf.matmul(tf.transpose(x), tf.diag(tf.reshape(delta, shape=[-1])))
u_mom_delta = tf.matmul(u_mom_delta, relu_gradient(phi, net))

b_hidden_mom_decay = tf.subtract(1.0, -tf.scalar_mul(learning_rate, tf.pow(phi, 2)))
b_hidden_mom_decay = tf.reduce_mean(b_hidden_mom_decay, axis=0)
b_hidden_mom_delta = tf.matmul(tf.transpose(delta), relu_gradient(phi, net))
b_hidden_mom_delta = tf.reshape(b_hidden_mom_delta, shape=[-1])

new_grads = []
weighted_phi = tf.matmul(tf.diag(tf.reshape(delta, shape=[-1])), phi)

new_u_grad = tf.reshape(tf.tile(weighted_phi, [1, tf.shape(U)[0]]), [-1, tf.shape(U)[0], tf.shape(U)[1]])
new_u_grad = tf.multiply(tf.reduce_mean(new_u_grad, axis=0), u_mom)
new_grads.append(new_u_grad)
new_b_hidden_grad = tf.multiply(tf.reduce_mean(weighted_phi, axis=0), b_hidden_mom)
new_grads.append(new_b_hidden_grad)

old_grads = optimizer.compute_gradients(loss, var_list=[U, b_hidden])
for i, (grad, var) in enumerate(old_grads):
    old_grads[i] = (new_grads[i], var)
other_grads = optimizer.compute_gradients(loss, var_list=[W, b_out])
train_op = optimizer.apply_gradients(old_grads + other_grads)

with tf.Session() as sess:
    for var in tf.global_variables():
        sess.run(var.initializer)
    for ep in range(epochs):
        u_mom_var = np.zeros((dim_in, dim_hidden))
        b_hidden_mom_var = np.zeros(dim_hidden)
        cur = 0
        while cur < trainX.shape[0]:
            end = min(cur + batch_size, trainX.shape[0])
            _, u_mom_decay_var, u_mom_delta_var, b_hidden_mom_decay_var, b_hidden_mom_delta_var = \
                sess.run([train_op, u_mom_decay, u_mom_delta, b_hidden_mom_decay, b_hidden_mom_delta],
                     feed_dict={
                         x: trainX[cur: end, :],
                         target: trainY[cur: end, :],
                         u_mom: u_mom_var,
                         b_hidden_mom: b_hidden_mom_var
                     })
            u_mom_var = np.multiply(u_mom_decay_var, u_mom_var) + learning_rate * u_mom_delta_var
            b_hidden_mom_var = np.multiply(b_hidden_mom_decay_var, b_hidden_mom_var) + \
                learning_rate * b_hidden_mom_delta_var
            cur = end
        cur_loss = sess.run(loss, feed_dict={
            x: testX,
            target: testY
        })
        print 'epoch', ep, cur_loss