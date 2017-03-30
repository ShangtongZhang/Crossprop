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

def genDataset(nSample, dims, beta=0.6):
    n, m = dims
    X = np.random.rand(nSample, n)
    X = np.where(X > 0.5, 0, 1)
    X[:, -1] = 1
    U = np.random.rand(n, m)
    U = np.where(U > 0.5, 1, -1)
    theta = n * beta - np.sum(np.where(U < 0, 1, 0), 0)
    W = np.array([np.random.randint(-1, 2) for _ in range(m)])
    Y = np.zeros(nSample)
    net = np.dot(X, U)
    phi = np.zeros(net.shape)
    for i in range(nSample):
        phi[i, :] = np.where(net[i, :] > theta, 1, 0)
    phi[:, -1] = 1
    for i in range(nSample):
        Y[i] = np.dot(phi[i, :], W) + np.random.randn()
    return X, Y, U, phi, W

if __name__ == '__main__':
    # data = []
    # for i in range(50):
    #     dims = [21, 1001]
    #     nSample = 40500
    #     X, Y = genDataset(nSample, dims)
    #     data.append((X, Y))
    # fw = open('GEOFF'+str(nSample)+'.bin', 'wb')
    # pickle.dump([nSample, dims, data], fw)
    # fw.close()
    fr = open('GEOFF.bin', 'rb')
    data = pickle.load(fr)
    for X, Y in data[2]:
        print(np.mean(Y))
    fr.close()