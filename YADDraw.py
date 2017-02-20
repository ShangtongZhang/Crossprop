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
    path = 'data/YAD_'+str(hiddenUnits)+'_'+str(stepSize)+'_'+str(nSample)+'.bin'
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
# units = [100, 300, 500, 700, 900]
# stepSizes = [0.00005, 0.0001, 0.0005, 0.001]
stepSizes = np.power(2., np.arange(-18, -10))
# stepSizes = np.power(2., np.arange(-16, -7))
# stepSizes = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05]
# stepSizes = [0]

labels = ['Backprop', 'Crossprop']
# epochs = 1000
epochs = 200
runs = 6
# samples = [3500, 6500, 15500, 24500]
units = [500]
samples = [3500]

for nSample in samples:
    nTestExamples = 500
    nTrainExamples = nSample - nTestExamples
    for unit in units:
        for stepInd, step in enumerate(stepSizes):
            data = getData(unit, step, nSample)
            if data is not None:
                trainErrors, testErrors = data
                trainErrors /= nTrainExamples
                testErrors /= nTestExamples

                trainMean = np.mean(trainErrors, 1)
                testMean = np.mean(testErrors, 1)

                trainStd = np.std(trainErrors, 1) / np.sqrt(runs)
                testStd = np.std(testErrors, 1) / np.sqrt(runs)

                for i in range(len(labels)):
                    if i == 0:
                        line = 'dashed'
                        color = 'b'
                    else:
                        line = 'solid'
                        color = 'r'
                    # plt.errorbar(np.arange(epochs), testMean[i, :], testStd[i, :], label=labels[i]+str(step))
                    plt.plot(np.arange(epochs), testMean[i, :], linestyle=line, label=labels[i]+str(step))
                    # plt.plot(np.arange(epochs), testMean[i, :], linestyle=line, color=color, label=labels[i]+str(step))
                    # plt.plot(np.arange(epochs), trainMean[i, :], label=labels[i]+str(step))

    plt.xlabel('Sweep')
    plt.ylabel('Average MSE')
    plt.ylim([0, 50])
    plt.title('YAD_'+str(unit)+'_'+str(nTrainExamples))
    plt.legend()
    # plt.savefig('figure/tanh_test_' + str(unit)+ '.png')
    # plt.savefig('figure/tanh_train_' + str(unit)+ '.png')
    plt.savefig('figure/YAD_test_'+str(unit)+'_'+str(nTrainExamples)+'.png')
    plt.close()
