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

class BackpropLearner:
    def __init__(self, stepSize, dims, activation='relu', init='orthogonal'):
        self.stepSize = stepSize
        dims[0] += 1
        dims[1] += 1

        if init == 'orthogonal':
            self.U = orthogonalInit(dims[0], dims[1])
        else:
            self.U = np.random.randn(dims[0], dims[1])

        self.W = np.random.randn(dims[1])
        if activation == 'relu':
            self.act = relu
            self.gradientAct = gradientRelu
        elif activation == 'tanh':
            self.act = tanh
            self.gradientAct = gradientTanh

    def predict(self, X=None):
        if X is not None:
            self.X = X
        self.net = np.dot(self.X, self.U)
        # self.phi = (np.exp(2 * self.net) - 1) / (np.exp(2 * self.net) + 1)
        self.phi = self.act(self.net)
        self.phi[-1] = 1
        self.y = np.dot(self.phi, self.W)
        return self.y

    def getGradient(self, error):
        gradientW = -error * self.phi
        gradientU = -error * np.multiply(np.repeat(np.matrix(self.W), self.U.shape[0], 0),
                               np.dot(np.matrix(self.X).T,
                                      # np.matrix(1 - np.power(self.phi, 2))))
                                      np.matrix(self.gradientAct(self.phi, self.net))))
        return [gradientW, gradientU]


    def learn(self, target):
        self.target = target
        error = target - self.y
        gradientW, gradientU = self.getGradient(error)
        self.gradientW = gradientW
        self.gradientU = gradientU
        # self.checkGradient()
        self.W -= self.stepSize * gradientW
        self.U -= self.stepSize * gradientU
        return 0.5 * error * error

    def checkGradient(self):
        epsilon = 1e-5
        for i in range(len(self.W)):
            self.W[i] += epsilon
            y1 = self.predict()
            error1 = self.target - y1
            self.W[i] -= 2 * epsilon
            y2 = self.predict()
            error2 = self.target - y2
            self.W[i] += epsilon
            g = 0.5 * (error1 * error1 - error2 * error2) / (2 * epsilon)
            if np.abs(self.gradientW[i] - g) > epsilon:
                print('W[', i, ']Gradient Error:')

        for i in range(self.U.shape[0]):
            for j in range(self.U.shape[1]):
                self.U[i, j] += epsilon
                y1 = self.predict()
                error1 = self.target - y1
                self.U[i, j] -= 2 * epsilon
                y2 = self.predict()
                error2 = self.target - y2
                self.U[i, j] += epsilon
                g = 0.5 * (error1 * error1 - error2 * error2) / (2 * epsilon)
                if np.abs(self.gradientU[i, j] - g) > epsilon:
                    print('U[', i, j, ']Gradient Error:')
