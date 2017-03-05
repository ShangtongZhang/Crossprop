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
from TFCommon import *

class AllCNN:
    def __init__(self, name, gate, initializer):
        self.x = tf.placeholder(tf.float32, shape=[None, 32, 32, 3])
        self.trainable_vars = []
        W, b, _, y_hat = convolution_2d(name, 'conv1', self.x, 3, 3, 96, initializer, gate.gate_fun)
        self.trainable_vars.extend([W, b])
        W, b, _, y_hat = convolution_2d(name, 'conv2', y_hat, 3, 96, 96, initializer, gate.gate_fun)
        self.trainable_vars.extend([W, b])
        W, b, _, y_hat = convolution_2d(name, 'pool1', y_hat, 3, 96, 96, initializer, gate.gate_fun, stride=2)
        self.trainable_vars.extend([W, b])
        W, b, _, y_hat = convolution_2d(name, 'conv3', y_hat, 3, 96, 192, initializer, gate.gate_fun)
        self.trainable_vars.extend([W, b])
        W, b, _, y_hat = convolution_2d(name, 'conv4', y_hat, 3, 192, 192, initializer, gate.gate_fun)
        self.trainable_vars.extend([W, b])
        W, b, _, y_hat = convolution_2d(name, 'pool2', y_hat, 3, 192, 192, initializer, gate.gate_fun, stride=2)
        self.trainable_vars.extend([W, b])
        W, b, _, y_hat = convolution_2d(name, 'conv5', y_hat, 3, 192, 192, initializer, gate.gate_fun)
        self.trainable_vars.extend([W, b])
        W, b, _, y_hat = convolution_2d(name, 'conv6', y_hat, 1, 192, 192, initializer, gate.gate_fun)
        self.trainable_vars.extend([W, b])
        W, b, _, y_hat = convolution_2d(name, 'conv7', y_hat, 1, 192, 10, initializer, gate.gate_fun)
        self.trainable_vars.extend([W, b])
        shape = y_hat.get_shape().as_list()
        self.var_out = tf.reshape(y_hat, [-1, shape[1] * shape[2] * shape[3]])

