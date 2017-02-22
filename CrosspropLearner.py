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
from Activation import *
from Initialization import *

class CrossPropLearner:
    def __init__(self, stepSize, dims, bias=[True, True], activation='relu', init='orthogonal'):
        self.stepSize = stepSize
        self.bias = bias
        dims[0] += int(bias[0])
        dims[1] += int(bias[1])

        if init == 'orthogonal':
            self.U = orthogonalInit(dims[0], dims[1])
        else:
            self.U = np.random.randn(dims[0], dims[1])

        self.W = np.random.randn(dims[1])
        self.H = np.zeros((dims[0], dims[1]))
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
        self.X = X
        self.net = np.dot(self.X, self.U)
        self.phi = self.act(self.net)
        if self.bias[1]:
            self.phi[-1] = 1
        self.y = np.dot(self.phi, self.W)
        return self.y

    def learn(self, target):
        error = target - self.y
        self.U += self.stepSize * error * np.multiply(np.repeat(np.matrix(self.phi), self.H.shape[0], 0), self.H)
        deltaUij = np.dot(np.matrix(self.X).T,
                          np.matrix(self.gradientAct(self.phi, self.net)))
        self.H = np.multiply(self.H, 1 - self.stepSize * np.repeat(np.matrix(np.power(self.phi, 2)), self.H.shape[0], 0)) + \
            error * self.stepSize * deltaUij
        self.W += self.stepSize * error * self.phi
        return 0.5 * error * error

class CrossPropLearnerV2:
    def __init__(self, stepSize, dims, activation='relu'):
        self.stepSize = stepSize
        dims[0] += 1
        dims[1] += 1
        self.U = np.random.randn(dims[0], dims[1])
        self.W = np.random.randn(dims[1])
        self.H = np.zeros(dims[1])
        if activation == 'relu':
            self.act = relu
            self.gradientAct = gradientRelu
        elif activation == 'tanh':
            self.act = tanh
            self.gradientAct = gradientTanh

    def predict(self, X):
        self.X = X
        self.net = np.dot(self.X, self.U)
        # self.phi = (np.exp(2 * self.net) - 1) / (np.exp(2 * self.net) + 1)
        self.phi = self.act(self.net)
        self.phi[-1] = 1
        self.y = np.dot(self.phi, self.W)
        return self.y

    def learn(self, target):
        error = target - self.y
        self.W += self.stepSize * error * self.phi
        self.H = self.H * (1 - self.stepSize * np.power(self.phi, 2)) + self.stepSize * error
        gradientU = np.multiply(np.repeat(np.matrix(self.phi * self.H), self.U.shape[0], 0),
                                np.dot(np.matrix(self.X).T,
                                       # np.matrix(1 - np.power(self.phi, 2))))
                                       np.matrix(self.gradientAct(self.phi, self.net))))
        self.U += self.stepSize * gradientU
        return 0.5 * error * error

