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

class IDBDLearner:
    def __init__(self, stepSize, dims):
        self.stepSize = stepSize
        dims[0] += 1
        dims[1] += 1
        self.U = np.random.randn(dims[0], dims[1])
        self.W = np.random.randn(dims[1])
        self.H = np.zeros(dims[1])
        self.Beta = np.zeros(dims[1])

    def predict(self, X):
        self.X = np.concatenate((X, [1]))
        self.net = np.dot(self.X, self.U)
        self.phi = (np.exp(2 * self.net) - 1) / (np.exp(2 * self.net) + 1)
        self.phi[-1] = 1
        self.y = np.dot(self.phi, self.W)
        return self.y

    def learn(self, target):
        self.target = target
        error = target - self.y
        print(self.Beta)
        # self.Beta += self.stepSize * error * self.phi * self.H
        deltaBeta = 0.01 * error * self.phi * self.H
        deltaBeta = np.where(deltaBeta > 2, 2, deltaBeta)
        deltaBeta = np.where(deltaBeta < -2, -2, deltaBeta)
        self.Beta += 0.01 * error * self.phi * self.H
        print(self.Beta)
        Alpha = np.exp(self.Beta)
        # gradientU = error * np.multiply(np.repeat(np.matrix(self.H * self.phi), self.U.shape[0], 0),
        #                                  np.dot(np.matrix(self.X).T,
        #                                         np.matrix(1 - np.power(self.phi, 2))))
        gradientU = error * np.multiply(np.repeat(np.matrix(self.W), self.U.shape[0], 0),
                                        np.dot(np.matrix(self.X).T,
                                               np.matrix(1 - np.power(self.phi, 2))))
        # print(gradientU)
        self.H = self.H * np.maximum(1 - Alpha * np.power(self.phi, 2), 0) + \
            Alpha * error * self.phi
        # self.H = self.H * (1 - Alpha * np.power(self.phi, 2)) + \
        #     Alpha * error * self.phi
        self.W += Alpha * error * self.phi
        self.U += self.stepSize * gradientU
        return 0.5 * error * error


