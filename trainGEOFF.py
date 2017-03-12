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
from DeepCrosspropLearner import *
from GEOFF import *
import logging

tag = 'utility_tanh_decay_'
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
fh = logging.FileHandler('log/%s.txt' % tag)
fh.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)
logger.addHandler(fh)
logger.addHandler(ch)

runs = 1
epochs = 100
# labels = ['Backprop', 'Crossprop', 'CrosspropV3']
labels = ['CrosspropAlt', 'CrosspropUtility', 'Backprop', 'Crossprop']
# labels = ['Backprop']

def test(learner, testX, testY):
    error = 0.0
    for i in range(testX.shape[0]):
        error += 0.5 * np.sum(np.power(learner.predict(testX[i, :]) - testY[i, :], 2))
    # error = 0.5 * np.sum(np.power(learner.predict(testX) - testY, 2))
    return error

def trainUnitWrapper(args):
   return trainUnit(*args)

def trainUnit(data, stepSize, learnerFeatures, nSample, startRun, endRun, trainErrors, testErrors, UTrack=None, WTrack=None, batchSize=1):
    np.random.seed()
    sep = nSample - 500
    for run in range(startRun, endRun):
        X, Y = data[run]
        trainX = np.matrix(X[: sep])
        trainY = np.matrix(Y[: sep]).T
        testX = np.matrix(X[sep:])
        testY = np.matrix(Y[sep:]).T
        # dims = [20, learnerFeatures, 50]
        dims = [20, learnerFeatures]
        act = 'tanh'
        init = 'normal'
        # act = 'relu'
        # init = 'orthogonal'
        use_norm = False
        lr_decay = 1.0

        bp = BackpropLearner(stepSize, list(dims), init=init, activation=act, use_normal=use_norm, lr_decay_factor=lr_decay)
        cp = CrossPropLearner(stepSize, list(dims), init=init, activation=act, use_normal=use_norm, lr_decay_factor=lr_decay)
        cpAlt = CrossPropLearnerAlt(stepSize, list(dims), init=init, activation=act, use_normal=use_norm, lr_decay_factor=lr_decay)
        cpUtility = CrosspropUtility(stepSize, list(dims), init=init, activation=act, use_normal=use_norm, lr_decay_factor=lr_decay)
        learners = [cpAlt, cpUtility, bp, cp]

        for ind in range(len(labels)):
            # print('Run', run, labels[ind], stepSize, learnerFeatures, nSample)
            for ep in range(epochs):
                # cur = 0
                # while (cur < trainX.shape[0]):
                #     end = min(cur + batchSize, trainX.shape[0])
                #     learners[ind].predict(trainX[cur: end, :])
                #     trainErrors[ind, run, ep] += learners[ind].learn(trainY[cur: end, :])
                #     cur = end
                # learners[ind].predict(trainX)
                # trainErrors[ind, run, ep] += learners[ind].learn(trainY)
                indices = np.arange(trainX.shape[0])
                np.random.shuffle(indices)
                for i in indices:
                    learners[ind].predict(trainX[i, :])
                    trainErrors[ind, run, ep] += learners[ind].learn(trainY[i, :], ep)
                if WTrack is not None:
                    U_delta, W_delta= learners[ind].get_params_changes()
                    WTrack[ind, run, ep] = W_delta
                    UTrack[ind, run, ep] = U_delta
                testErrors[ind, run, ep] = test(learners[ind], testX, testY)
                logger.info('Run %d %s Epoch %d %f' % (run, labels[ind], ep, testErrors[ind, run, ep] / 500))
    return [trainErrors, testErrors]

def train(stepSize, learnerFeatures, nSample):
    trainErrors = np.zeros((len(labels), runs, epochs))
    testErrors = np.zeros(trainErrors.shape)
    UTrack = np.zeros(trainErrors.shape)
    WTrack = np.zeros(trainErrors.shape)
    fr = open('GEOFF'+str(nSample)+'.bin', 'rb')
    __, _, data = pickle.load(fr)
    fr.close()

    logger.info('***%s, %d, %d' % (str(stepSize), learnerFeatures, nSample))
    nThreads = 4
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
    # results = Pool(nThreads).map(trainUnitWrapper, args)
    trainUnit(data, stepSize, learnerFeatures, nSample, 0, 1, trainErrors, testErrors, UTrack, WTrack)
    # for trError, teError in results:
    #     trainErrors += trError
    #     testErrors += teError

    fw = open('tmp/'+tag+str(learnerFeatures)+'_'+str(stepSize)+'_'+str(nSample)+'.bin', 'wb')
    pickle.dump({'errors': [trainErrors, testErrors],
                 'track': [UTrack, WTrack],
                 'stepSize': stepSize,
                 'learnerFeatures': learnerFeatures}, fw)
    fw.close()

# samples = [1500, 3500, 6500, 9500, 12500, 15500, 18500]
# learnerFeatures = [100, 300, 500, 700, 900, 1000, 2000]
# stepSizes = [0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05]

# samples = [3500, 6500, 15500, 24500]
# learnerFeatures = [100, 500, 900]
# stepSizes = np.power(2., np.arange(-16, -10))
# for step in stepSizes[: 3]:
#     train(step, 500, 3500)
# train(stepSizes[0], 500, 3500)
# train(0.0001, 500, 3500)
# train(stepSizes[-1], 500, 3500)
# train(stepSizes[0], learnerFeatures[0], 1500)
# stepSizes = np.power(2., np.arange(-16, -11))
stepSizes = [0.001]
samples = [3500, 6500, 9500]
# samples = [12500, 15500, 18500]
learnerFeatures = [100, 500, 900]
for lr in stepSizes:
    for sample in samples:
        for hidden_unit in learnerFeatures:
            train(lr, hidden_unit, sample)

# train(0.01, 500, 3500)

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