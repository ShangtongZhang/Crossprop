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
from CrosspropLearner import *
from BackpropLearner import *
from IDBDLearner import *
from GEOFF import *

alphas = [0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001]
features = [50, 500, 1000]

def train(nFeatures, alpha):
    dims = [20, nFeatures]
    nSample = 10000
    learnerDims = [dims[0], dims[1] * 2]
    # irrelevantInput = 0
    # dims[0] += irrelevantInput
    # X = np.concatenate((X, np.ones((X.shape[0], irrelevantInput))), 1)

    # labels = ['Backprop']
    labels = ['IDBD', 'Crossprop', 'Backprop']
    # labels = ['Crossprop', 'Backprop']

    runs = 1
    errors = np.zeros((len(labels), nSample))

    for run in range(runs):
        X, Y = genDataset(nSample, dims)
        # learners = [
        learners = [IDBDLearner(0.00001, list(learnerDims)),
                    CrossPropLearner(alpha, list(learnerDims)),
                    BackpropLearner(alpha, list(learnerDims))]
        for i in range(nSample):
            # print('Run', run, 'Example', i)
            for ind in range(len(labels)):
                learners[ind].predict(X[i, :])
                errors[ind, i] += learners[ind].learn(Y[i])

    errors /= runs
    print('feature', nFeatures, 'alpha', alpha, labels, np.sum(errors, 1) / nSample)

    fr = open('data/onlineD' + str(nFeatures) + '.bin', 'wb')
    pickle.dump({'errors': errors,
                 'nExamples': nSample,
                 'runs': runs,
                 'dims': dims,
                 'labels': labels}, fr)
    fr.close()

train(50, 0.005)
# train(500, 0.0005)
# train(1000, 0.0001)

# for feature in features:
#     for alpha in alphas:
#         train(feature, alpha)