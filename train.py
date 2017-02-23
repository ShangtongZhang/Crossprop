#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

from __future__ import print_function
import numpy as np
from functools import partial
from multiprocessing import Pool, Process
import pickle
from CrosspropLearner import *
from BackpropLearner import *
from DeepBackpropLearner import *
from GEOFF import *

# runs = len(data)
runs = 6
epochs = 200
# labels = ['Backprop', 'Crossprop', 'CrosspropV2']
labels = ['Backprop', 'Crossprop']
# labels = ['Backprop-Adam']
# labels = ['CrosspropV2']
# labels = ['Backprop-RMSProp']

def test(learner, testX, testY):
    error = 0.0
    for i in range(testX.shape[0]):
        error += 0.5 * np.power(learner.predict(testX[i, :]) - testY[i], 2)
    return error

def trainUnitWrapper(args):
   return trainUnit(*args)

def trainUnit(data, stepSize, learnerFeatures, nSample, startRun, endRun, trainErrors, testErrors):
    sep = nSample - 500
    for run in range(startRun, endRun):
        X, Y = data[run]
        trainX = X[: sep]
        trainY = Y[: sep]
        testX = X[sep:]
        testY = Y[sep:]
        dims = [20, learnerFeatures, 50]
        # cp = CrossPropLearner(stepSize, list(dims))
        # bp = BackpropLearner(stepSize, list(dims))
        bpAdam = BackpropLearner(stepSize, list(dims), init='normal', gradient='RMSProp')
        cpv2 = CrossPropLearnerV2(stepSize, list(dims))
        bp = DeepBackpropLearner(stepSize, list(dims), outputLayer='bp')
        cp = DeepBackpropLearner(stepSize, list(dims), outputLayer='cp')
        # learners = [bp, cp, cpv2]
        # learners = [bpAdam]
        learners = [bp, cp]

        for ind in range(len(labels)):
            print('Run', run, labels[ind], stepSize, learnerFeatures, nSample)
            for ep in range(epochs):
                indices = np.arange(trainX.shape[0])
                np.random.shuffle(indices)
                for i in indices:
                    learners[ind].predict(trainX[i, :])
                    trainErrors[ind, run, ep] += learners[ind].learn(trainY[i])
                testErrors[ind, run, ep] = test(learners[ind], testX, testY)
                print('Run', run, labels[ind], 'Epoch', ep, testErrors[ind, run, ep])
    return [trainErrors, testErrors]

def train(stepSize, learnerFeatures, nSample):
    trainErrors = np.zeros((len(labels), runs, epochs))
    testErrors = np.zeros(trainErrors.shape)
    fr = open('GEOFF'+str(nSample)+'.bin', 'rb')
    __, _, data = pickle.load(fr)
    fr.close()

    nThreads = 6
    step = runs // nThreads
    startRun = []
    endRun = []
    for i in range(nThreads):
        startRun.append(i * step)
        endRun.append((i + 1) * step)
    endRun[-1] = runs
    args = []
    for i in range(len(startRun)):
        args.append((data, stepSize, learnerFeatures, nSample, startRun[i], endRun[i], trainErrors, testErrors))
    results = Pool(nThreads).map(trainUnitWrapper, args)
    for trError, teError in results:
        trainErrors += trError
        testErrors += teError

    fw = open('tmp/deep_'+str(learnerFeatures)+'_'+str(stepSize)+'_'+str(nSample)+'.bin', 'wb')
    pickle.dump({'errors': [trainErrors, testErrors],
                 'stepSize': stepSize,
                 'learnerFeatures': learnerFeatures}, fw)
    fw.close()

# samples = [1500, 3500, 6500, 9500, 12500, 15500, 18500]
# learnerFeatures = [100, 300, 500, 700, 900, 1000, 2000]
# stepSizes = [0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05]

samples = [3500, 6500, 15500, 24500]
learnerFeatures = [100, 500, 900]
stepSizes = np.power(2., np.arange(-16, -10))
for step in stepSizes[: 3]:
    train(step, 500, 3500)

# train(stepSizes[-1], 500, 3500)
# train(stepSizes[0], learnerFeatures[0], 1500)

# for units in learnerFeatures:
#     train(stepSizes[1], units, 1500)

# def multiTrainWrapper(args):
#    return train(*args)

# args = [(stepSizes[3], feat) for feat in learnerFeatures]
# args = [(0.0001, 300, sample) for sample in samples]
# print(args)
# Pool(5).map(multiTrainWrapper, args)

# for sample in samples:
#     train(0.0001, 300, sample)
# train(0.0001, 500, 1500)
# for step in stepSizes[:3]:
#     train(step, 500, 1500)