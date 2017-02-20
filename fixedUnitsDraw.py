from __future__ import print_function
import numpy as np
from functools import partial
from multiprocessing import Pool
import pickle
from CrosspropLearner import *
from BackpropLearner import *
from GEOFF import *
import matplotlib.pyplot as plt
import os

def getData(hiddenUnits, stepSize, nSample):
    # path = 'data/relu_total_offline_'+str(hiddenUnits)+'_'+str(stepSize)+'_'+str(nSample)+'.bin'
    path = 'data/YAD_total_'+str(hiddenUnits)+'_'+str(stepSize)+'_'+str(nSample)+'.bin'
    # path = 'data/new_offline_'+str(hiddenUnits)+'_'+str(stepSize)+'.bin'
    if not os.path.isfile(path):
        return None
    fr = open(path, 'rb')
    data = pickle.load(fr)
    fr.close()
    return data['errors']

# units = [100, 300, 500, 700, 900, 1100]
# units = [100, 500, 900]
# units = [100, 500]
units = [200]
# units = [100, 300, 500, 700, 900]
# stepSizes = [0.00005, 0.0001, 0.0005, 0.001]
# stepSizes = np.power(2., np.arange(-16, -10))
# stepSizes = np.power(2., np.arange(-17, -5))
stepSizes = np.power(2., np.arange(-17, -10))
# stepSizes = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05]
# stepSizes = [0]

labels = ['Backprop', 'Crossprop']
epochs = 200
runs = 3
# samples = [3500, 6500, 15500, 24500]
# samples = [6500, 9500, 13500]
# samples = [3500, 6500, 9500]
# samples = [13500, 18500, 23500]
samples = [23500]

for nSample in samples:
    # nSample = 1500
    nTestExamples = 500
    nTrainExamples = nSample - nTestExamples

    # infoBP = dict()
    # infoCP = dict()
    # infos = [infoBP, infoCP]
    for unit in units:
        # infoBP[unit] = []
        # infoCP[unit] = []
        for stepInd, step in enumerate(stepSizes):
            data = getData(unit, step, nSample)
            if data is not None:
                trainErrors, testErrors = data
                trainErrors /= nTrainExamples
                testErrors /= nTestExamples

                trainMean = np.mean(trainErrors, 1)
                testMean = np.mean(testErrors, 1)

                print(nSample, step, testMean[:, -1])

                # trainMean = np.sum(trainErrors, 1) / runs
                # testMean = np.sum(testErrors, 1) / runs

                trainStd = np.std(trainErrors, 1) / np.sqrt(runs)
                testStd = np.std(testErrors, 1) / np.sqrt(runs)

                # infoBP[unit].append([testMean[0, -1], step])
                # infoCP[unit].append([testMean[1, -1], step])

                for i in range(len(labels)):
                    if i == 0:
                        line = 'dashed'
                        color = 'b'
                    else:
                        line = 'solid'
                        color = 'r'
                    # plt.errorbar(np.arange(epochs), testMean[i, :], testStd[i, :], label=labels[i]+str(step))
                    # plt.plot(np.arange(epochs), testMean[i, :], linestyle=line, label=labels[i]+str(step))
                    # plt.plot(np.arange(epochs), testMean[i, :], linestyle=line, color=color)
                    plt.plot(np.arange(epochs), testMean[i, :], linestyle=line, color=color, label=labels[i]+str(step))
                    # plt.plot(np.arange(epochs), trainMean[i, :], linestyle=line, label=labels[i]+str(step))

    plt.xlabel('Sweep')
    plt.ylabel('Average MSE')
    plt.title('relu_'+str(unit)+'_'+str(nTrainExamples))
    plt.ylim([0, 1])
    plt.legend()
    plt.savefig('figure/YAD_test_'+str(unit)+'_'+str(nTrainExamples)+'.png')
    # plt.savefig('figure/relu_train_'+str(unit)+'_'+str(nTrainExamples)+'.png')
    plt.close()

# for info in infos:
#     for unit in info.keys():
#         info[unit] = sorted(info[unit], key=lambda x:x[0])[0][1]
        # print(unit, info[unit][0][1])

# print(infos)
# bestStep = [{500: 0.00048828125, 900: 0.00048828125, 100: 3.0517578125e-05},
#             {500: 6.103515625e-05, 900: 6.103515625e-05, 100: 6.103515625e-05}]
