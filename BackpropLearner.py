#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

import numpy as np
from functools import partial
from multiprocessing import Pool
import pickle
from BasicLearner import *
from Activation import *
from Initialization import *
from GradientAdjuster import *

class BackpropLearner(BasicLearner):
    def __init__(self, stepSize, dims, bias=[True, True], activation='relu',
                 init='orthogonal', gradient='', asOutput=False,
                 use_normal=False, lr_decay_factor=None, step_size_W=None, step_size_U=None):
        BasicLearner.__init__(self, stepSize, dims, bias, activation, init, asOutput,
                              use_normal, lr_decay_factor, step_size_W, step_size_U)

        if gradient == 'adam':
            self.WAdjustor = AdamGradientAdjuster(self.step_size_W)
            self.UAdjustor = AdamGradientAdjuster(self.step_size_U)
        elif gradient == 'RMSProp':
            self.WAdjustor = RMSPropGradientAdjuster(self.step_size_W)
            self.UAdjustor = RMSPropGradientAdjuster(self.step_size_U)
        else:
            self.WAdjustor = GradientAdjuster(self.step_size_W)
            self.UAdjustor = GradientAdjuster(self.step_size_U)

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

    def learn(self, target, epoch=None):
        BasicLearner.learn(self, target, epoch)
        if self.stepSize is not None:
            self.WAdjustor.stepSize = self.stepSize
            self.UAdjustor.stepSize = self.stepSize

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

# TODO: refactor classification learner
class BackpropLearnerClassification:
    def __init__(self, stepSize, dims, bias=[True, True], activation='relu', init='orthogonal', gradient=''):
        self.stepSize = stepSize
        self.bias = bias
        dims[0] += int(bias[0])
        dims[1] += int(bias[1])

        if init == 'orthogonal':
            self.U = np.matrix(orthogonalInit(dims[0], dims[1]))
        else:
            self.U = np.matrix(np.random.randn(dims[0], dims[1]))

        self.W = np.matrix(np.random.randn(dims[1], dims[2]))

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
        exp_y = np.exp(self.y - np.max(self.y))
        self.prob = exp_y / np.sum(exp_y)
        return self.prob

    def getGradient(self, error):
        gradientW = -self.phi.T * error
        phiError = -error * self.W.T
        netError = np.multiply(phiError, self.gradientAct(self.phi, self.net))
        gradientU = self.X.T * netError
        return [gradientW, gradientU]

    def learn(self, target):
        self.target = target
        error = np.matrix(target - self.y)
        gradientW, gradientU = self.getGradient(error)
        self.gradientW = gradientW
        self.gradientU = gradientU
        # self.checkGradient()
        self.W -= self.WAdjustor.adjust(self.gradientW)
        self.U -= self.UAdjustor.adjust(self.gradientU)
        return -np.sum(np.multiply(np.log(self.prob), target))

    def checkGradient(self):
        epsilon = 1e-8
        for i in range(self.W.shape[0]):
            for j in range(self.W.shape[1]):
                self.W[i, j] += epsilon
                y1 = self.predict()
                error1 = 0.5 * np.sum(np.power(self.target - y1, 2))
                self.W[i, j] -= 2 * epsilon
                y2 = self.predict()
                error2 = 0.5 * np.sum(np.power(self.target - y2, 2))
                self.W[i, j] += epsilon
                g = (error1 - error2) / (2 * epsilon)
                if np.abs(self.gradientW[i, j] - g) > 0.01:
                    print('W[', i, j, ']Gradient Error:', g, self.gradientW[i, j])

        for i in range(self.U.shape[0]):
            for j in range(self.U.shape[1]):
                self.U[i, j] += epsilon
                y1 = self.predict()
                error1 = 0.5 * np.sum(np.power(self.target - y1, 2))
                self.U[i, j] -= 2 * epsilon
                y2 = self.predict()
                error2 = 0.5 * np.sum(np.power(self.target - y2, 2))
                self.U[i, j] += epsilon
                g = (error1 - error2) / (2 * epsilon)
                if np.abs(self.gradientU[i, j] - g) > 0.01:
                    print('U[', i, j, ']Gradient Error:', g, self.gradientU[i, j])