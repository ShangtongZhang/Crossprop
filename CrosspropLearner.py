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

class CrossPropLearner(BasicLearner):
    def __init__(self, stepSize, dims, bias=[True, True],
                 activation='relu', init='orthogonal', asOutput=False,
                 use_normal=False, lr_decay_factor=None, step_size_W=None, step_size_U=None):
        BasicLearner.__init__(self, stepSize, dims, bias, activation, init, asOutput,
                              use_normal, lr_decay_factor, step_size_W, step_size_U)

        self.H = np.matrix(np.zeros((dims[0], dims[1])))

    def learn(self, target, epoch=None):
        BasicLearner.learn(self, target, epoch)

        error = target - self.y
        phi = np.asarray(np.matrix(np.diag(error.flat)) * self.phi)
        wDelta = np.mean(phi, axis=0)
        self.W += self.step_size_W * np.matrix(wDelta).T

        phi = np.repeat(phi, self.U.shape[0], 0).reshape((-1, self.U.shape[0], self.U.shape[1]))
        phi = np.matrix(np.mean(phi, axis=0))
        uDelta = np.multiply(phi, self.H)
        self.U += self.step_size_U * uDelta

        hDecay = np.asarray(1 - self.step_size_W * np.power(self.phi, 2))
        hDecay = np.repeat(hDecay, self.U.shape[0], 0).reshape((-1, self.U.shape[0], self.U.shape[1]))
        hDecay = np.matrix(np.mean(hDecay, axis=0))
        hDelta = self.X.T * np.matrix(np.diag(error.flat)) * \
                 self.gradientAct(self.phi, self.net) / self.X.shape[0]
        self.H = np.multiply(self.H, hDecay) + self.step_size_W * hDelta

        errorPhi = -error * self.W.T
        errorNet = np.multiply(errorPhi, self.gradientAct(self.phi, self.net))
        if self.bias[1]:
            errorNet[:, -1] = 0
        if self.asOutput:
            self.errorInput = errorNet * self.U.T
            if self.bias[0]:
                self.errorInput[:, -1] = 0

        return 0.5 * np.sum(np.power(error, 2))

class CrossPropLearnerAlt(BasicLearner):
    def __init__(self, stepSize, dims, bias=[True, True], activation='relu',
                 init='orthogonal', asOutput=False, use_normal=False,
                 lr_decay_factor=None, step_size_W=None, step_size_U=None):
        BasicLearner.__init__(self, stepSize, dims, bias, activation, init,
                              asOutput, use_normal, lr_decay_factor, step_size_W, step_size_U)

        self.H = np.matrix(np.zeros((dims[1], 1)))

    def learn(self, target, epoch=None):
        BasicLearner.learn(self, target, epoch)

        error = self.y - target

        gradientW = self.phi.T * error
        phi_phi_grad = np.multiply(self.phi, self.gradientAct(self.phi, self.net))
        phi_phi_grad = np.multiply(phi_phi_grad, (self.H * error.T).T)
        phi_phi_grad[:, -1] = 0
        gradientU = self.X.T * phi_phi_grad

        h_decay = 1 - self.step_size_W * np.power(self.phi, 2)
        h_decay = np.repeat(h_decay, self.H.shape[1], 0).T

        h_gradient = np.repeat(error, self.H.shape[0], 0)

        self.W -= self.step_size_W * gradientW
        self.U -= self.step_size_U * gradientU
        self.H = np.multiply(h_decay, self.H) - self.step_size_W * h_gradient

        if self.asOutput:
            errorPhi = error * self.W.T
            errorNet = np.multiply(errorPhi, self.gradientAct(self.phi, self.net))
            if self.bias[1]:
                errorNet[:, -1] = 0
            self.errorInput = errorNet

        return 0.5 * np.sum(np.power(error, 2))

class CrosspropUtility(BasicLearner):
    def __init__(self, stepSize, dims, bias=[True, True], activation='relu', init='orthogonal',
                 asOutput=False, use_normal=False, lr_decay_factor=None):
        BasicLearner.__init__(self, stepSize, dims, bias, activation, init,
                              asOutput, use_normal, lr_decay_factor)

        self.H = np.zeros((dims[0], dims[1]))

    def learn(self, target, epoch=None):
        BasicLearner.learn(self, target, epoch)
        error = target - self.y

        delta_U = np.asarray(self.W).flatten() * self.H
        self.U += self.stepSize * delta_U

        h_deacy = np.asarray(1 - self.stepSize * np.power(self.phi, 2)).flatten()
        h_delta = np.asarray(error).flatten() * np.asarray(self.X.T * self.gradientAct(self.phi, self.net))
        self.H = h_deacy * np.asarray(self.H) + self.stepSize * h_delta

        self.W += self.stepSize * self.phi.T * error
        return 0.5 * np.sum(np.power(error, 2))

class CrosspropUtilityAlt(BasicLearner):
    def __init__(self, stepSize, dims, bias=[True, True], activation='relu', init='orthogonal',
                 asOutput=False, use_normal=False, lr_decay_factor=None):
        BasicLearner.__init__(self, stepSize, dims, bias, activation, init,
                              asOutput, use_normal, lr_decay_factor)

        self.H = np.zeros(dims[1])

    def learn(self, target, epoch=None):
        BasicLearner.learn(self, target, epoch)
        error = target - self.y

        delta_U = np.asarray(self.W).flatten() * self.H * np.asarray(self.X.T * self.gradientAct(self.phi, self.net))
        self.U += self.stepSize * delta_U

        h_decay = np.asarray(1 - self.stepSize * np.power(self.phi, 2)).flatten()
        h_delta = np.asarray(error) - np.asarray(self.W).flatten() * np.asarray(self.phi).flatten()
        self.H = h_decay * self.H + self.stepSize * h_delta

        self.W += self.stepSize * self.phi.T * error
        return 0.5 * np.sum(np.power(error, 2))

class CrosspropUtilityAlpha(BasicLearner):
    def __init__(self, stepSize, dims, bias=[True, True], activation='relu', init='orthogonal',
                 asOutput=False, use_normal=False, lr_decay_factor=None, theta=1e-4):
        BasicLearner.__init__(self, stepSize, dims, bias, activation, init,
                              asOutput, use_normal, lr_decay_factor)
        self.m = np.zeros((dims[0], dims[1]))
        self.h = np.zeros(dims[1])
        self.beta = np.log(np.ones(dims[1]) * 0.001)
        self.alpha = np.zeros(dims[1])
        self.theta = theta

    def learn(self, target, epoch=None):
        BasicLearner.learn(self, target, epoch)
        error = np.asscalar(target - self.y)

        self.beta += self.theta * error * self.h * np.asarray(self.phi).flatten()
        self.alpha = np.exp(self.beta)

        self.W += np.matrix(self.alpha * error * np.asarray(self.phi).flatten()).T
        self.U += self.stepSize * self.m

        phi_2 = np.asarray(np.power(self.phi, 2)).flatten()
        m_decay = 1 - self.theta * np.power(self.h, 2) * phi_2
        m_delta = error * self.h * np.asarray(self.X.T * self.gradientAct(self.phi, self.net))
        self.m = m_decay * self.m + self.theta * m_delta

        h_decay = 1 - self.alpha * phi_2
        h_delta = error * self.alpha * np.asarray(self.phi).flatten()
        self.h = h_decay * self.h + h_delta

        return 0.5 * error * error

# TODO: refactor classification learner
class CrossPropLearnerClassification:
    def __init__(self, stepSize, dims, bias=[True, True], activation='relu', init='orthogonal'):
        self.stepSize = stepSize
        self.bias = bias
        dims[0] += int(bias[0])
        dims[1] += int(bias[1])
        self.dims = dims

        if init == 'orthogonal':
            self.U = np.matrix(orthogonalInit(dims[0], dims[1]))
        else:
            self.U = np.matrix(np.random.randn(dims[0], dims[1]))

        self.W = np.matrix(np.random.randn(dims[1], dims[2]))
        self.H = np.array(np.zeros((dims[2], dims[0], dims[1])))
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
        exp_y = np.exp(self.y - np.max(self.y))
        self.prob = exp_y / np.sum(exp_y)
        return self.prob

    def learn(self, target):
        error = np.matrix(target - self.y)
        phi = np.asarray(np.repeat(self.phi, self.H.shape[1], 0))
        phi_h = phi * self.H * np.asarray(error).reshape([-1, 1, 1])
        self.U += self.stepSize * np.sum(phi_h, axis=0)

        h_decay = np.asarray(1 - self.stepSize * np.repeat(np.power(self.phi, 2), self.H.shape[1], 0))
        deltaUij = self.X.T * self.gradientAct(self.phi, self.net)
        self.H = h_decay * self.H + self.stepSize * np.asarray(error).reshape([-1, 1, 1]) * np.asarray(deltaUij)
        self.W += self.stepSize * self.phi.T * error
        return -np.sum(np.multiply(np.log(self.prob), target))

class CrossPropUtilityClassification:
    def __init__(self, stepSize, dims, bias=[True, True], activation='relu', init='orthogonal'):
        self.stepSize = stepSize
        self.bias = bias
        dims[0] += int(bias[0])
        dims[1] += int(bias[1])
        self.dims = dims

        if init == 'orthogonal':
            self.U = np.matrix(orthogonalInit(dims[0], dims[1]))
        else:
            self.U = np.matrix(np.random.randn(dims[0], dims[1]))

        self.W = np.matrix(np.random.randn(dims[1], dims[2]))
        self.H = np.array(np.zeros((dims[2], dims[0], dims[1])))
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
        exp_y = np.exp(self.y - np.max(self.y))
        self.prob = exp_y / np.sum(exp_y)
        return self.prob

    def learn(self, target):
        error = np.matrix(target - self.y)

        delta_U = np.asarray(self.W) * self.H.transpose([1, 2, 0])
        delta_U = np.sum(delta_U, axis=2)
        self.U += self.stepSize * delta_U

        h_decay = np.asarray(1 - self.stepSize * np.repeat(np.power(self.phi, 2), self.H.shape[1], 0))
        deltaUij = self.X.T * self.gradientAct(self.phi, self.net)
        self.H = h_decay * self.H + self.stepSize * np.asarray(error).reshape([-1, 1, 1]) * np.asarray(deltaUij)
        self.W += self.stepSize * self.phi.T * error
        return -np.sum(np.multiply(np.log(self.prob), target))