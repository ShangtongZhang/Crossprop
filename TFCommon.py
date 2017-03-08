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
      else: # new
           W = variable_scope.get_variable("W", [dim_in, dim_out],
                                           initializer=initializer)
           b = variable_scope.get_variable("b", [dim_out],
                                           initializer=initializer)
  z_hat = math_ops.matmul(var_in, W)
  z_hat = nn_ops.bias_add(z_hat, b)
  y_hat = transfer(z_hat)
  return W, b, z_hat, y_hat

def crossprop_layer(name, var_in, dim_in, dim_hidden, dim_out, gate_fun, initializer, keep_prob_input, keep_prob_hidden, reuse=False):
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

        # dropout
        U_drop = tf.nn.dropout(U, keep_prob_input)
        
        net = math_ops.matmul(var_in, U_drop)

        net = nn_ops.bias_add(net, b_hidden)
        phi = gate_fun(net)
        
        # dropout
        W_drop = tf.nn.dropout(W, keep_prob_hidden)
        
        y = math_ops.matmul(phi, W_drop)
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