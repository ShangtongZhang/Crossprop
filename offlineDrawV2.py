from __future__ import print_function
import numpy as np
from functools import partial
from multiprocessing import Pool
import pickle
from CrosspropLearner import *
from BackpropLearner import *
from GEOFF import *
import matplotlib.pyplot as plt
import math

figureInd = 0
bpTestError = []
cpTestError = []

def draw(learnerFeatures, stepSize, nSample):

    # labels = ['Backprop', 'Crossprop']
    # labels = ['CrosspropV2']
    epochs = 1000

    nTrainExamples = 1000
    nTestExamples = 500
    # fr = open('data/cpv2_offline_'+str(learnerFeatures)+'_'+str(stepSize)+'_'+str(nSample)+'.bin', 'rb')
    # dataV2 = pickle.load(fr)
    # fr.close()

    fr = open('data/new_offline_'+str(learnerFeatures)+'_'+str(stepSize)+'.bin', 'rb')
    data = pickle.load(fr)
    fr.close()

    # trainErrors = np.concatenate((data['errors'][0], dataV2['errors'][0]))
    # testErrors = np.concatenate((data['errors'][1], dataV2['errors'][1]))
    trainErrors, testErrors = data['errors']
    # labels = ['Backprop', 'Crossprop', 'CrosspropV2']
    labels = ['Backprop', 'Crossprop']

    trainErrors /= nTrainExamples
    testErrors /= nTestExamples

    trainMean = np.mean(trainErrors, 1)
    testMean = np.mean(testErrors, 1)

    trainStd = np.std(trainErrors, 1) / np.sqrt(50)
    testStd = np.std(testErrors, 1) / np.sqrt(50)

    print(testMean[0, :10])
    print(testMean[1, :10])

    bpTestError.append([testMean[0, -1], stepSize, learnerFeatures, testStd[0, -1]])
    cpTestError.append([testMean[1, -1], stepSize, learnerFeatures, testStd[1, -1]])

    # global figureInd
    # plt.figure(figureInd)
    # figureInd += 1
    # for i in range(len(labels)):
    #     plt.errorbar(np.arange(epochs), trainMean[i, :], trainStd[i, :], label=labels[i])
    # plt.xlabel('Epoch')
    # plt.ylabel('Average MSE')
    # plt.title('Train_'+str(learnerFeatures)+'_'+str(stepSize))
    # plt.legend()
    # plt.savefig('figure/train_'+str(learnerFeatures)+'_'+str(stepSize)+'_'+str(nSample)+'.png')
    #
    # plt.figure(figureInd)
    # figureInd += 1
    # for i in range(len(labels)):
    #     plt.errorbar(np.arange(epochs), testMean[i, :], testStd[i, :], label=labels[i])
    # plt.xlabel('Epoch')
    # plt.ylabel('Average MSE')
    # plt.title('Test_'+str(learnerFeatures)+'_'+str(stepSize))
    # plt.legend()
    # plt.savefig('figure/test_'+str(learnerFeatures)+'_'+str(stepSize)+'_'+str(nSample)+'.png')

samples = [1500, 3500, 6500, 10500, 12500, 15500, 18500]
learnerFeatures = [100, 300, 500, 700, 900]
stepSizes = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05]
# args = [(feat, stepSizes[0]) for feat in learnerFeatures] + \
#     [(feat, stepSizes[3]) for feat in learnerFeatures]
# args = [(300, 0.0001, 1500)]
# args = [(300, 0.0001, sample) for sample in samples]
# for arg in args:
#     draw(arg[0], arg[1], arg[2])
# for unit in learnerFeatures:
#     for step in stepSizes:
#         draw(unit, step, 1500)

draw(300, 0.005, 1500)

# bpTestError = [expt for expt in bpTestError if not math.isnan(expt[0])]
# cpTestError = [expt for expt in cpTestError if not math.isnan(expt[0])]
# bpTestError = sorted(bpTestError, key=lambda x:x[0])
# cpTestError = sorted(cpTestError, key=lambda x:x[0])
# fw = open('analysis.txt', 'w')
# fw.write('[test error at last epoch, step size, hidden units, std error]\n')
# fw.write('bp test error:\n')
# for expt in bpTestError:
#     fw.write(str(expt))
#     fw.write('\n')
# fw.write('cp test error:\n')
# for expt in cpTestError:
#     fw.write(str(expt))
#     fw.write('\n')
# fw.close()
