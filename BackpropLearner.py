#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

from __future__ import print_function
# import numpy as np
import minpy.numpy as np
from functools import partial
from multiprocessing import Pool
import pickle
from Activation import *
from Initialization import *

class GradientAdjuster:
    def __init__(self, stepSize):
        self.stepSize = stepSize

    def adjust(self, gradient):
        return self.stepSize * gradient

class AdamGradientAdjuster:
    def __init__(self, stepSize):
        self.stepSize = stepSize
        self.time = 0
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.beta1t = self.beta2t = 1.0
        self.epsilon = 1e-8
        self.m = 0
        self.v = 0

    def adjust(self, gradient):
        self.time += 1
        self.beta1t *= self.beta1
        self.beta2t *= self.beta2
        lr = self.stepSize * np.sqrt(1 - self.beta2t) / (1 - self.beta1t)
        self.m = self.beta1 * self.m + (1 - self.beta1) * gradient
        self.v = self.beta2 * self.v + (1 - self.beta2) * np.power(gradient, 2)
        return lr * np.divide(self.m, np.sqrt(self.v) + self.epsilon)

class RMSPropGradientAdjuster:
    def __init__(self, stepSize):
        self.stepSize = stepSize
        self.eta = 1e-3
        self.Eg2 = 0
        self.epsilon = 1e-8

    def adjust(self, gradient):
        self.Eg2 = 0.9 * self.Eg2 + 0.1 * np.power(gradient, 2)
        return np.multiply(self.eta / np.sqrt(self.Eg2 + self.epsilon), gradient)


class BackpropLearner:
    def __init__(self, stepSize, dims, bias=[True, True], activation='relu', init='orthogonal', gradient='', asOutput=False):
        self.stepSize = stepSize
        self.asOutput = asOutput
        self.bias = bias
        dims[0] += int(bias[0])
        dims[1] += int(bias[1])

        if init == 'orthogonal':
            self.U = np.matrix(orthogonalInit(dims[0], dims[1]))
        else:
            self.U = np.matrix(np.random.randn(dims[0], dims[1]))

        self.W = np.matrix(np.random.randn(dims[1], 1))
        if activation == 'relu':
            self.act = relu
            self.gradientAct = gradientRelu
        elif activation == 'tanh':
            self.act = tanh
            self.gradientAct = gradientTanh
        elif activation == 'sigmoid':
            self.act = sigmoid
            self.gradientAct = gradientSigmoid

        if gradient == 'adam':
            self.WAdjustor = AdamGradientAdjuster(self.stepSize)
            self.UAdjustor = AdamGradientAdjuster(self.stepSize)
        elif gradient == 'RMSProp':
            self.WAdjustor = RMSPropGradientAdjuster(self.stepSize)
            self.UAdjustor = RMSPropGradientAdjuster(self.stepSize)
        else:
            self.WAdjustor = GradientAdjuster(self.stepSize)
            self.UAdjustor = GradientAdjuster(self.stepSize)

    def predict(self, X=None):
        if X is not None:
            self.X = np.matrix(X)
        self.net = self.X * self.U
        self.phi = self.act(self.net)
        if self.bias[1]:
            self.phi[:, -1] = 1
        self.y = self.phi * self.W
        return self.y

    def getGradient(self, error):
        gradientW = -np.matrix(np.diag(error.flat)) * self.phi
        gradientW = np.mean(gradientW, axis=0).T

        errorPhi = -error * self.W.T
        errorNet = np.multiply(errorPhi, self.gradientAct(self.phi, self.net))
        if self.bias[1]:
            errorNet[:, -1] = 0
        gradientU = self.X.T * errorNet / self.X.shape[0]

        if self.asOutput:
            self.errorInput = errorNet * self.U.T
            if self.bias[0]:
                self.errorInput[:, -1] = 0
        return [gradientW, gradientU]

    def learn(self, target):
        self.target = target
        error = target - self.y
        gradientW, gradientU = self.getGradient(error)
        self.gradientW = gradientW
        self.gradientU = gradientU
        # self.checkGradient()
        self.W -= self.WAdjustor.adjust(self.gradientW)
        self.U -= self.UAdjustor.adjust(self.gradientU)
        return 0.5 * np.sum(np.power(error, 2))

    def checkGradient(self):
        epsilon = 1e-8
        for i in range(len(self.W)):
            self.W[i, 0] += epsilon
            y1 = self.predict()
            error1 = self.target - y1
            self.W[i, 0] -= 2 * epsilon
            y2 = self.predict()
            error2 = self.target - y2
            self.W[i, 0] += epsilon
            g = 0.5 * (error1 * error1 - error2 * error2) / (2 * epsilon)
            if np.abs(self.gradientW[i, 0] - g) > 0.01:
                print('W[', i, ']Gradient Error:', g, self.gradientW[i])

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
                if np.abs(self.gradientU[i, j] - g) > 0.01:
                    print('U[', i, j, ']Gradient Error:', g, self.gradientU[i, j])
