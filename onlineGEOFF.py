import numpy as np
import pickle
import logging
from Backprop import *
from DynamicGEOFF import *
from BackpropLearner import *
from CrosspropLearner import *

tag = 'online_GEOFF'
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

window_size = 50
dims = [20, 500, 1]

labels = ['SMD', 'BP']
def train(learning_rate):
    init_fn = normal_init
    gate_type = Tanh
    smd = Backprop(list(dims), learning_rate, gate_type, SMDLayer, init_fn, MSEOutputLayer)
    bp = Backprop(list(dims), learning_rate, gate_type, BPLayer, init_fn, MSEOutputLayer)
    methods = [smd, bp]
    train_examples = 50000
    GEOFF = DynamicGEOFF(20, 1000, train_examples)
    train_x, train_y = GEOFF.generate(True)
    train_error = np.zeros((len(labels), train_examples))
    for train_index in range(len(train_x)):
        for method_ind in range(len(methods)):
            method = methods[method_ind]
            x = train_x[train_index, :].reshape(1, -1)
            y = train_y[train_index].reshape(1, -1)
            error = method.train(x, y)
            train_error[method_ind, train_index] = error

            if train_index - window_size >= 0:
                err = np.mean(train_error[method_ind, train_index - window_size: train_index])
                logger.info('%s, %dth example, average error %f' %
                            (labels[method_ind], train_index, err))
            else:
                logger.info('%s, %dth example %f' %
                            (labels[method_ind], train_index, error))
    with open('tmp/%s_%s_%s.bin' % (tag, gate_type().name, str(learning_rate)), 'wb') as f:
        pickle.dump({'error': train_error}, f)

def train_cp(learning_rate, hidden_units, n_examples):
    labels = ['BP', 'CP']

    dims = [20, hidden_units]
    act = 'tanh'
    init = 'normal'
    # act = 'relu'
    # init = 'orthogonal'
    use_norm = False
    lr_decay = 0.0

    bp = BackpropLearner(learning_rate, list(dims), init=init, activation=act,
                         use_normal=use_norm, lr_decay_factor=lr_decay)
    cp = CrossPropLearner(learning_rate, list(dims), init=init, activation=act,
                          use_normal=use_norm, lr_decay_factor=lr_decay)

    methods = [bp, cp]
    stages = 2

    runs = 1
    train_error = np.zeros((len(methods), runs, stages * n_examples))
    GEOFF = DynamicGEOFF(20, 1000, n_examples)
    for stage in range(stages):
        if stage == 0:
            train_x, train_y = GEOFF.generate(True)
        else:
            GEOFF.W_mutate(0.5)
            GEOFF.reset_X()
            train_x, train_y = GEOFF.generate()
            for method in methods:
                method.W = np.matrix(np.random.randn(*method.W.shape))
        train_x = np.matrix(train_x)
        train_y = np.matrix(train_y).T
        for example_ind in range(train_x.shape[0]):
            for method_ind in range(len(methods)):
                methods[method_ind].predict(train_x[example_ind, :])
                error = methods[method_ind].learn(train_y[example_ind, :], 0)
                train_error[method_ind, 0, example_ind + stage * n_examples] = error
                logger.info('%s, stage %d, example %d, error %f' %
                           (labels[method_ind], stage, example_ind, error))

    file_path = 'tmp/%s_%d_%s.bin' % (tag, hidden_units, str(learning_rate))
    with open(file_path, 'wb') as f:
        pickle.dump({'error': train_error}, f)
    return train_error

train_cp(0.0001, 500, 10000)



