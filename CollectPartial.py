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

def collect(stride, units, stepSize, nSample):
    # labels = ['Backprop', 'Crossprop', 'CrosspropV2']
    # labels = ['Backprop', 'Crossprop']
    labels = ['BP-Adam']
    epochs = 200
    runs = 30
    trainErrors = np.zeros((len(labels), runs, epochs))
    testErrors = np.zeros(trainErrors.shape)
    for startRun in range(0, runs, stride):
        path = 'partial/YAD_adam_partial_'+str(startRun)+'_'+str(units)+'_'+str(stepSize)+'_'+str(nSample)+'.bin'
        # path = 'partial/YAD_partial_'+str(startRun) + str(units)+'_'+str(stepSize)+'_'+str(nSample)+'.bin'
        if not os.path.isfile(path):
            print(path)
            return
        fr = open(path, 'rb')
        data = pickle.load(fr)
        fr.close()
        trErrors, teErrors = data['errors']
        trainErrors += trErrors
        testErrors += teErrors

    fw = open('partial/YAD_adam_total_'+str(units)+'_'+str(stepSize)+'_'+str(nSample)+'.bin', 'wb')
    pickle.dump({'errors': [trainErrors, testErrors],
                 'stepSize': stepSize,
                 'learnerFeatures': units}, fw)
    fw.close()

# stepSizes = [0.00005, 0.0001, 0.0005]
# stepSizes = np.power(2., np.arange(-16, -7))
stepSizes = np.power(2., np.arange(-16, -10))
# stepSizes = np.power(2., np.arange(-17, -11))
# stepSizes = np.power(2., np.arange(-11, -5))
# stepSizes = np.power(2., np.arange(-17, -10))
# stepSizes = [0.001]
# units = [100, 300, 500, 700, 900]
# units = [300, 700]
# units = [100, 500, 900]

# units = [100, 500]
units = [60]
# samples = [13500, 18500, 23500]
samples = [3500, 6500, 9500]
# samples = [3500, 6500, 15500, 24500]
for step in stepSizes:
    for unit in units:
        for sample in samples:
            collect(3, unit, step, sample)

# collect(1, 500, 0.00005, 40500)
