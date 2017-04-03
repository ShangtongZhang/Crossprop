#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

import numpy as np

class DynamicGEOFF:
    beta = 0.6

    def __init__(self, input_dim, n_target_feature, n_examples):
        self.input_dim = input_dim + 1
        self.n_target_feature = n_target_feature + 1
        self.n_examples = n_examples

    def reset_X(self):
        self.X = None

    def generate(self, reset=False):
        if reset or (self.X is None):
            X = np.random.rand(self.n_examples, self.input_dim)
            X = np.where(X > 0.5, 0, 1)
            X[:, -1] = 1
            self.X = X

        if reset or (self.U is None):
            U = np.random.rand(self.input_dim, self.n_target_feature)
            U = np.where(U > 0.5, 1, -1)
            self.U = U
            self.theta = self.input_dim * self.beta - np.sum(np.where(self.U < 0, 1, 0), 0)

        if reset or (self.W is None):
            self.W = np.array([np.random.randint(-1, 2) for _ in range(self.n_target_feature)])

        self.net = np.dot(self.X, self.U)
        self.phi = np.zeros(self.net.shape)
        for i in range(self.phi.shape[0]):
            self.phi[i, :] = np.where(self.net[i, :] > self.theta, 1, 0)
        self.phi[:, -1] = 1

        Y = np.sum(np.asarray(self.phi) * np.asarray(self.W), axis= 1)
        Y += np.random.randn(len(Y))
        return self.X, Y

    def W_mutate(self, factor):
        positions = np.arange(len(self.W))
        np.random.shuffle(positions)
        positions = positions[: int(len(self.W) * factor)]
        self.W[positions] = np.random.randint(-1, 2, size=len(positions))
        # self.W[positions] += 1

    def target_feature_size_mutate(self, n_target_feature):
        self.n_target_feature = n_target_feature + 1
        self.U = None
        self.W = None






