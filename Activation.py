#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

from __future__ import print_function
import numpy as np
from functools import partial
from multiprocessing import Pool
import pickle

def tanh(net):
    return (np.exp(2 * net) - 1) / (np.exp(2 * net) + 1)

def gradientTanh(phi, net):
    return 1 - np.power(phi, 2)

def relu(net):
    return np.maximum(net, 0)

def gradientRelu(phi, net):
    return np.where(net >= 0, 1, 0)

def sigmoid(net):
    return 1.0 / (1.0 + np.exp(-net))

def gradientSigmoid(phi, net):
    return phi * (1 - phi)

class Tanh:
    def __init__(self):
        self.gate_fun = tanh
        self.gate_fun_grad = gradientTanh

class Relu:
    def __init__(self):
        self.gate_fun = relu
        self.gate_fun_grad = gradientRelu

class Sigmoid:
    def __init__(self):
        self.gate_fun = sigmoid
        self.gate_fun_grad = gradientSigmoid

class Identity:
    def __init__(self):
        self.gate_fun = lambda x: x
        self.gate_fun_grad = lambda phi, net: np.ones(phi.shape)