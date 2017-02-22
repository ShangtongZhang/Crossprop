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

def getData(hiddenUnits, stepSize, nSample, prefix):
    # path = 'data/relu_total_offline_'+str(hiddenUnits)+'_'+str(stepSize)+'_'+str(nSample)+'.bin'
    path = 'data/'+prefix+str(hiddenUnits)+'_'+str(stepSize)+'_'+str(nSample)+'.bin'
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
units = [60]
# units = [100, 300, 500, 700, 900]
# stepSizes = [0.00005, 0.0001, 0.0005, 0.001]
# stepSizes = np.power(2., np.arange(-16, -10))
stepSizes = np.power(2., np.arange(-17, -5))
# stepSizes = np.power(2., np.arange(-17, -10))
# stepSizes = np.power(2., np.arange(-16, -7))
# stepSizes = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05]
# stepSizes = [0]

labels = ['Backprop', 'Crossprop', 'Backprop-Adam']
# epochs = 1000
epochs = 200
runs = 30
# samples = [3500, 6500, 15500, 24500]
samples = [3500, 6500, 9500]
# samples = [23500]

for nSample in samples:
    nTestExamples = 500
    nTrainExamples = nSample - nTestExamples

    for unit in units:
        # infoBP[unit] = []
        # infoCP[unit] = []
        asymptoticError = [[] for _ in range(len(labels))]
        for stepInd, step in enumerate(stepSizes):
            # data = getData(unit, step, nSample, 'relu_total_offline_')
            # data2 = getData(unit, step, nSample, 'adam_total_')
            data = getData(unit, step, nSample, 'YAD_total_')
            data2 = getData(unit, step, nSample, 'YAD_adam_total_')
            if data is not None:
                trainErrors, testErrors = data
                if data2 is not None:
                    trainErrorsAdam, testErrorsAdam = data2
                    trainErrors = np.concatenate((trainErrors, trainErrorsAdam))
                    testErrors = np.concatenate((testErrors, testErrorsAdam))

                trainErrors /= nTrainExamples
                testErrors /= nTestExamples

                trainMean = np.mean(trainErrors, 1)
                testMean = np.mean(testErrors, 1)

                print(unit, nSample, step, testMean[:, -1])

                trainStd = np.std(trainErrors, 1) / np.sqrt(runs)
                testStd = np.std(testErrors, 1) / np.sqrt(runs)

                for i in range(testErrors.shape[0]):
                    asymptoticError[i].append([testMean[i, -1], stepInd, step, testMean[i, :], testStd[i, :]])

        for i in range(len(labels)):
            asymptoticError[i] = sorted(asymptoticError[i], key=lambda x: x[0])[0]
        for i in range(len(labels)):
            if labels[i] == 'Backprop':
                color = 'b'
            elif labels[i] == 'Crossprop':
                color = 'r'
            elif labels[i] == 'Backprop-Adam':
                color = 'g'
            plt.errorbar(np.arange(epochs), asymptoticError[i][3], asymptoticError[i][4], color=color, label=labels[i]+str(asymptoticError[i][2]))

        diff = str(asymptoticError[1][0] - asymptoticError[0][0])
        plt.xlabel('Sweep')
        plt.ylabel('Average MSE')
        plt.ylim([0, 1])
        plt.title('relu_'+str(unit)+'_'+str(nTrainExamples)+'_'+diff)
        plt.legend()
        # plt.savefig('figure/tanh_test_' + str(unit)+ '.png')
        # plt.savefig('figure/tanh_train_' + str(unit)+ '.png')
        # plt.savefig('figure/relu_test_'+str(unit)+'_'+str(nTrainExamples)+'.png')
        plt.savefig('figure/YAD_test_'+str(unit)+'_'+str(nTrainExamples)+'.png')
        plt.close()
