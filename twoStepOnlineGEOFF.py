import numpy as np
import pickle
import logging
from DynamicGEOFF import *
from CrosspropLearner import *
from BackpropLearner import *

tag = 'tf_two_step_online_GEOFF'
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

labels = ['BP', 'CP']
def train(learning_rate_W, learning_rate_U, hidden_units, n_examples):
    runs = 30
    stages = 4
    GEOFF = DynamicGEOFF(20, 1000, n_examples)
    cp = CrossPropLearner(None, [20, hidden_units], init='normal',
                          step_size_W=learning_rate_W, step_size_U=learning_rate_U)
    bp = BackpropLearner(None, [20, hidden_units], init='normal',
                         step_size_W=learning_rate_W, step_size_U=learning_rate_U)
    methods = [bp, cp]
    train_error = np.zeros((len(methods), runs, stages * n_examples))
    for run in range(runs):
        x0, y0 = GEOFF.generate(True)
        GEOFF.W_mutate(0.5)
        GEOFF.reset_X()
        x1, y1 = GEOFF.generate()
        GEOFF.W_mutate(0.5)
        GEOFF.reset_X()
        x2, y2 = GEOFF.generate()

        train_xs = [x0, x1, x0, x1]
        train_ys = [y0, y1, y0, y1]

        for stage in range(stages):
            train_x = train_xs[stage]
            train_y = train_ys[stage]
            train_y = np.matrix(train_y).T
            for example_ind in range(train_x.shape[0]):
                for method_ind, method in enumerate(methods):
                    method.predict(train_x[example_ind, :])
                    error = method.learn(train_y[example_ind, :])
                    index = example_ind + stage * n_examples
                    train_error[method_ind, run, index] = error
                    logger.info('W: %f, U:%f, run %d, stage %d, example %d, %s, error %f' %
                                (learning_rate_W, learning_rate_U,
                                 run, stage, example_ind, labels[method_ind], error))

    file_path = 'tmp/%s_%d_%s_%s.bin' % (tag, hidden_units, str(learning_rate_W), str(learning_rate_U))
    with open(file_path, 'wb') as f:
        pickle.dump({'error': train_error}, f)
    return train_error

learning_rates_W = [0.0005, 0.0001, 0.00005]
learning_rates_U = [0.0005, 0.0001, 0.00005]
for learning_rate_W in learning_rates_W:
    for learning_rate_U in learning_rates_U:
        train(learning_rate_W, learning_rate_U, 500, 5000)



