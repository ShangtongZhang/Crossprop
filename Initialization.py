#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

from __future__ import print_function
import numpy as np
from functools import partial
from multiprocessing import Pool
import pickle

def orthogonalInit(dim0, dim1):
    random = np.random.RandomState()
    X = random.rand(dim0, dim1)
    U, _, V = np.linalg.svd(X, full_matrices=False)
    if dim0 >= dim1:
        W = U * np.sqrt(2.0)
    else:
        W = V * np.sqrt(2.0)
    return W

def orthogonal_init(dim0, dim1):
    return orthogonalInit(dim0, dim1)

def normal_init(dim0, dim1):
    random = np.random.RandomState()
    return random.randn(dim0, dim1)

