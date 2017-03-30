#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

import numpy as np

from Activation import *
from Initialization import *

class BPLayer:
    def __init__(self, dim_in, dim_out, gate, learning_rate, init_fn, bias_term_in_phi=True):
        self.W = init_fn(dim_in, dim_out)
        self.gate = gate
        self.learning_rate = learning_rate
        self.bias_term_in_phi = bias_term_in_phi

    def set_next_layer(self, next_layer):
        self.next_layer = next_layer

    def forward(self, x):
        self.x = x
        self.net = np.dot(x, self.W)
        self.phi = self.gate.gate_fun(self.net)
        if self.bias_term_in_phi:
            self.phi[:, -1] = 1
        self.next_layer.forward(self.phi)

    def backward(self):
        error = self.next_layer.backward()
        net_error = error * self.gate.gate_fun_grad(self.phi, self.net)
        if self.bias_term_in_phi:
            net_error[:, -1] = 0
        self.net_error = net_error
        x_error = np.dot(self.net_error, self.W.T)
        return x_error

    def compute_grad_W(self):
        self.backward()
        grad_W = np.dot(self.x.T, self.net_error)
        return grad_W

    def update(self):
        grad_W = -self.compute_grad_W()
        self.W += self.learning_rate * grad_W
        self.next_layer.update()

class SMDLayer(BPLayer):
    def __init__(self, dim_in, dim_out, gate, learning_rate, init_fn, bias_term_in_phi=True):
        BPLayer.__init__(self, dim_in, dim_out, gate, learning_rate, init_fn, bias_term_in_phi)
        self.alpha = np.random.rand(dim_in, dim_out) * learning_rate
        self.h = np.zeros(self.W.shape)
        self.epsilon = 1e-4

    def compute_hessian_W_dot_h(self):
        delta = self.h * self.epsilon
        self.W += delta
        grad_W_plus = self.compute_grad_W()
        self.W -= 2 * delta
        grad_W_minus = self.compute_grad_W()
        self.W += delta
        hessian_W_dot_h = (grad_W_plus - grad_W_minus) / (2 * self.epsilon)
        return hessian_W_dot_h

    def update(self):
        hessian_W_dot_h = self.compute_hessian_W_dot_h()
        self.forward(self.x)
        grad_W = -self.compute_grad_W()
        alpha_decay = np.maximum(0.1, 1 + self.learning_rate * grad_W * self.h)
        h_delta = self.alpha * (grad_W - hessian_W_dot_h)
        self.W += self.alpha * grad_W
        self.alpha *= alpha_decay
        self.h += h_delta
        self.next_layer.update()

class SoftmaxOutputLayer:
    def set_target(self, target):
        self.target = target

    def forward(self, x):
        exp_x = np.exp(x - np.max(x))
        self.pred = exp_x / np.sum(exp_x)

    def correct_labels(self):
        compare = np.where(np.argmax(self.pred, 1) == np.argmax(self.target, 1), 1, 0)
        return np.sum(compare)

    def backward(self):
        return self.pred - self.target

    def update(self):
        return

class Backprop:
    def __init__(self, dims, learning_rate, gate_type, layer_type, init_fn):
        self.layers = []
        for i in range(1, len(dims)):
            if i == len(dims) - 1:
                gate = Identity()
                bias_term_in_phi = False
                dim_in = dims[i - 1] + 1
                dim_out = dims[i]
            else:
                gate = gate_type()
                bias_term_in_phi = True
                dim_in = dims[i - 1] + 1
                dim_out = dims[i] + 1
            self.layers.append(layer_type(dim_in, dim_out, gate, learning_rate, init_fn,
                                          bias_term_in_phi))
        self.layers.append(SoftmaxOutputLayer())

        for i in range(1, len(self.layers)):
            self.layers[i - 1].set_next_layer(self.layers[i])

    def train(self, x, y):
        self.layers[-1].set_target(y)
        self.layers[0].forward(x)
        correct_labels = self.layers[-1].correct_labels()
        self.layers[0].update()
        return correct_labels
