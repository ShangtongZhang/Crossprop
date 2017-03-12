#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

import numpy as np
from functools import partial
from multiprocessing import Pool
import pickle
from Activation import *
from Initialization import *

class BasicLearner:
    def __init__(self, stepSize, dims, bias, activation,
                 init, asOutput, use_norm, lr_decay_factor):
        self.stepSize = stepSize
        self.asOutput = asOutput
        self.bias = bias
        dims[0] += int(bias[0])
        dims[1] += int(bias[1])

        if init == 'orthogonal':
            self.U = np.matrix(orthogonalInit(dims[0], dims[1]))
        else:
            self.U = np.matrix(np.random.randn(dims[0], dims[1]))

        self.use_norm = use_norm
        self.step_size_norm = 0.0
        self.step_count = 0
        self.initial_lr = self.stepSize

        self.lr_decay_factor = lr_decay_factor

        self.W = np.matrix(np.random.randn(dims[1], 1))

        self.initial_W = np.copy(self.W)
        self.initial_U = np.copy(self.U)

        if activation == 'relu':
            self.act = relu
            self.gradientAct = gradientRelu
        elif activation == 'tanh':
            self.act = tanh
            self.gradientAct = gradientTanh
        elif activation == 'sigmoid':
            self.act = sigmoid
            self.gradientAct = gradientSigmoid

    def predict(self, X):
        self.X = np.matrix(X)
        self.net = self.X * self.U
        self.phi = self.act(self.net)
        if self.bias[1]:
            self.phi[:, -1] = 1
        self.y = self.phi * self.W

        if self.use_norm:
            self.step_count += 1
            self.step_size_norm += (np.sum(np.power(np.asarray(self.phi), 2)) - self.step_size_norm) / float(self.step_count)
            self.stepSize = self.initial_lr / self.step_size_norm

        return self.y

    def learn(self, target, epoch):
        if self.lr_decay_factor is not None:
            self.stepSize = self.initial_lr / (1 + self.lr_decay_factor * epoch)

    def get_params_changes(self):
        W_delta = np.sum(np.power(self.W - self.initial_W, 2))
        U_delta = np.sum(np.power(self.U - self.initial_U, 2))
        return U_delta, W_delta