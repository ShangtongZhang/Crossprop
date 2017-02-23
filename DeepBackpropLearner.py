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
from CrosspropLearner import *
from BackpropLearner import *

class DeepBackpropLearner:
    def __init__(self, stepSize, dims, outputLayer='cp', bias=[True, True, True], activation='relu', init='orthogonal'):
        self.stepSize = stepSize
        self.bias = bias
        if outputLayer == 'cp':
            self.outputLayer = CrossPropLearner(stepSize, list(dims[1:]), bias=bias[1:], activation=activation, init=init)
        elif outputLayer == 'bp':
            self.outputLayer = BackpropLearner(stepSize, list(dims[1:]), bias=bias[1:], activation=activation, init=init)

        for i in range(len(dims)):
            dims[i] += int(bias[i])

        if init == 'orthogonal':
            self.W = orthogonalInit(dims[0], dims[1])
        else:
            self.W = np.random.randn(dims[0], dims[1])

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
        self.net = np.dot(self.X, self.W)
        self.phi = self.act(self.net)
        if self.bias[1]:
            self.phi[-1] = 1
        self.y = self.outputLayer.predict(self.phi)
        return self.y

    def learn(self, target):
        loss = self.outputLayer.learn(target)
        gradient = self.outputLayer.inputGradient
        gradient = np.multiply(gradient, np.matrix(self.gradientAct(self.phi, self.net)))
        gradientW = np.matrix(self.X).T * gradient
        self.W -= self.stepSize * gradientW
        return loss

