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

# tag = 'tanh_W_mutation'
tag = 'tanh_feature_mutation'

def collect(stride, units, stepSize, nSample):
    # labels = ['Backprop', 'Crossprop', 'CrosspropV2']
    # labels = ['Backprop', 'Crossprop']
    # labels = ['BP-Adam']
    labels = ['Backprop', 'Crossprop', 'CrosspropAlt']
    epochs = 150
    runs = 30
    trainErrors = np.zeros((len(labels), runs, epochs))
    testErrors = np.zeros(trainErrors.shape)
    UTrack = np.zeros(trainErrors.shape)
    WTrack = np.zeros(trainErrors.shape)

    for startRun in range(0, runs, stride):
        path = 'partial/%s_%d_%d_%s_%d.bin' % (tag, startRun, units, str(stepSize), nSample)
        if not os.path.isfile(path):
            print(path)
            return
        fr = open(path, 'rb')
        data = pickle.load(fr)
        fr.close()
        trErrors, teErrors = data['errors']
        trainErrors += trErrors
        testErrors += teErrors
        UT, WT = data['track']
        UTrack += UT
        WTrack += WT

    path = 'partial/total_%s_%d_%s_%d.bin' % (tag, units, str(stepSize), nSample)
    fw = open(path, 'wb')
    pickle.dump({'errors': [trainErrors, testErrors],
                 'track': [UTrack, WTrack],
                 'stepSize': stepSize,
                 'learnerFeatures': units}, fw)
    fw.close()

# stepSizes = [0.00005, 0.0001, 0.0005]
# stepSizes = np.power(2., np.arange(-16, -7))
stepSizes = np.power(2., np.arange(-16, -10))
# stepSizes = np.power(2., np.arange(-17, -11))
# stepSizes = np.power(2., np.arange(-17, -11))
# stepSizes = np.power(2., np.arange(-11, -5))
# stepSizes = np.power(2., np.arange(-17, -10))
# stepSizes = [0.001]
# units = [100, 300, 500, 700, 900]
# units = [300, 700]
# units = [100, 500, 900]

units = [100, 500]
# units = [60]
# samples = [13500, 18500, 23500, 40500]
# samples = [3500, 6500, 9500]
# samples = [3500, 6500, 15500, 24500]
samples =[6500]
for step in stepSizes:
    for unit in units:
        for sample in samples:
            collect(3, unit, step, sample)

# collect(1, 500, 0.00005, 40500)
