from __future__ import print_function
import numpy as np
from functools import partial
from multiprocessing import Pool
import pickle
from CrosspropLearner import *
from BackpropLearner import *
from GEOFF import *
import matplotlib.pyplot as plt

figureInd = 0
def draw(learnerFeatures, stepSize, nSample):

    # labels = ['Backprop', 'Crossprop', 'CrosspropV2']
    labels = ['Backprop', 'Crossprop']
    # labels = ['CrosspropV2']
    # epochs = 1000
    epochs = 1000
    runs = 1

    nTestExamples = 500
    nTrainExamples = nSample -  nTestExamples
    fr = open('data/one_relu_offline_'+str(learnerFeatures)+'_'+str(stepSize)+'_'+str(nSample)+'.bin', 'rb')
    data = pickle.load(fr)
    fr.close()

    trainErrors, testErrors = data['errors']
    trainErrors /= nTrainExamples
    testErrors /= nTestExamples

    trainMean = np.mean(trainErrors, 1)
    testMean = np.mean(testErrors, 1)

    trainStd = np.std(trainErrors, 1) / np.sqrt(runs)
    testStd = np.std(testErrors, 1) / np.sqrt(runs)

    # print(trainMean[0, :10])

    global figureInd
    plt.figure(figureInd)
    figureInd += 1
    for i in range(len(labels)):
        # plt.errorbar(np.arange(epochs), trainMean[i, :], trainStd[i, :], label=labels[i])
        plt.plot(np.arange(epochs), trainMean[i, :], label=labels[i])
    plt.xlabel('Epoch')
    plt.ylabel('Average MSE')
    plt.title('Train_'+str(learnerFeatures)+'_'+str(stepSize)+'_'+str(nTrainExamples))
    plt.legend()
    plt.savefig('figure/one_relu_train_'+str(learnerFeatures)+'_'+str(stepSize)+'_'+str(nSample)+'.png')
    plt.close()

    plt.figure(figureInd)
    figureInd += 1
    for i in range(len(labels)):
        # plt.errorbar(np.arange(epochs), testMean[i, :], testStd[i, :], label=labels[i])
        plt.plot(np.arange(epochs), testMean[i, :], label=labels[i])
    plt.xlabel('Epoch')
    plt.ylabel('Average MSE')
    plt.title('Test_'+str(learnerFeatures)+'_'+str(stepSize)+'_'+str(nTrainExamples))
    plt.legend()
    plt.savefig('figure/one_relu_test_'+str(learnerFeatures)+'_'+str(stepSize)+'_'+str(nSample)+'.png')
    plt.close()

samples = [1500, 3500, 6500, 10500, 12500, 15500, 18500]
learnerFeatures = [100, 300, 500, 700, 900]
stepSizes = [0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05]
# args = [(feat, stepSizes[0]) for feat in learnerFeatures] + \
#     [(feat, stepSizes[3]) for feat in learnerFeatures]
# args = [(300, 0.0001, 1500)]
# args = [(300, 0.0001, sample) for sample in samples]
# args = [(2000, step, 1500) for step in stepSizes[: 3]]

args = []
stepSizes = [0.0001, 0.0005, 0.001]
# stepSizes = [0.001]
units = [100, 300, 500, 700, 900]
# units = [700]
for step in stepSizes:
    for unit in units:
        args.append((unit, step, 1500))

# args = [(1100, 0.00005, 1500), (1100, 0.0001, 1500), (1100, 0.0005, 1500)]
args = [(500, 0.0001, 1500)]
for arg in args:
    draw(arg[0], arg[1], arg[2])