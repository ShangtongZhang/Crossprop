#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

from __future__ import print_function
import numpy as np
from utils.utils import *
from functools import partial
from multiprocessing import Pool
import pickle
from CrosspropLearner import *
from BackpropLearner import *

import matplotlib.pyplot as plt

nFeatures = 50
# nFeatures = 500
# nFeatures = 1000

fr = open('data/online' + str(nFeatures) + '.bin', 'rb')
data = pickle.load(fr)
fr.close()

errors = data['errors']
nExamples = data['nExamples']

plt.figure(0)
for i in range(len(data['labels'])):
    plt.plot(np.arange(nExamples), errors[i, :], label=data['labels'][i])
plt.xlabel('examples')
plt.ylabel('SE')
plt.xscale('log')
plt.legend()
plt.savefig('figure/online_SE.png')

plt.figure(1)
errors = np.add.accumulate(errors, 1)
errors /= np.arange(1, nExamples + 1)
for i in range(len(data['labels'])):
    plt.plot(np.arange(nExamples), errors[i, :], label=data['labels'][i])
plt.xlabel('examples')
plt.ylabel('MSE')
plt.xscale('log')
plt.legend()
plt.savefig('figure/online_MSE.png')

plt.show()