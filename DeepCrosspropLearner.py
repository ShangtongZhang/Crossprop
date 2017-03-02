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
from CrosspropLearner import *

class DeepCrosspropLearner:
    def __init__(self, stepSize, dims, bias=[True, True, True], activation='relu', init='orthogonal'):
        self.stepSize = stepSize
        self.bias = bias
        self.outputLayer = CrossPropLearnerAlt(stepSize, list(dims[1:]), bias=bias[1:],
                                            activation=activation, init=init, asOutput=True)

        for i in range(len(dims)):
            dims[i] += int(bias[i])

        if init == 'orthogonal':
            self.W = np.matrix(orthogonalInit(dims[0], dims[1]))
        else:
            self.W = np.matrix(np.random.randn(dims[0], dims[1]))

        self.H = np.matrix(np.zeros((dims[1], dims[2])))
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
        self.net = self.X * self.W
        self.phi = self.act(self.net)
        if self.bias[1]:
            self.phi[:, -1] = 1
        self.y = self.outputLayer.predict(self.phi)
        return self.y


    def learn(self, target):
        loss = self.outputLayer.learn(target)
        error = self.outputLayer.errorInput
        phi_phi_grad = np.multiply(self.phi, self.gradientAct(self.phi, self.net))
        phi_phi_grad = np.multiply(phi_phi_grad, (self.H * error.T).T)
        gradientW = self.X.T * phi_phi_grad

        h_deacy = 1 - self.stepSize * np.power(self.phi, 2)
        h_deacy = np.repeat(h_deacy, self.H.shape[1], 0).T

        h_gradient = np.repeat(error, self.H.shape[0], 0)

        self.W -= self.stepSize * gradientW
        self.H = np.multiply(h_deacy, self.H) - self.stepSize * h_gradient
        return loss
