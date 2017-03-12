#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

import numpy as np
from functools import partial
from multiprocessing import Pool
import pickle

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