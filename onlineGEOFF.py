import numpy as np
import pickle
import logging
from DynamicGEOFF import *
from TFBackprop import *
from TFCrossprop import *

tag = 'tf_online_GEOFF'
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

labels = ['CP', 'CP-lambda', 'CP-ALT', 'CP-ALT-lambda', 'BP', 'BP-adam', 'BP-rms', 'BP-mom']
# labels = ['CP', 'CP-lambda', 'CP-ALT', 'CP-ALT-lambda', 'BP']
def train(learning_rate, hidden_units, n_examples):
    # gate = Relu()
    gate = Tanh()
    runs = 30
    stages = 6
    GEOFF = DynamicGEOFF(20, 1000, n_examples)
    cp = CrossPropRegression(dims[0], dims[1], learning_rate, gate,
                             name='cp', lam=0)
    cp_lam = CrossPropRegression(dims[0], dims[1], learning_rate, gate,
                                 name='cp_lam', lam=0.5)
    cp_alt = CrossPropAlt(dims[0], dims[1], dims[2], learning_rate, gate,
                          name='cp_alt', lam=0, output_layer='MSE')
    cp_alt_lam = CrossPropAlt(dims[0], dims[1], dims[2], learning_rate, gate,
                              name='cp_alt_lam', lam=0.5, output_layer='MSE')
    bp = BackPropRegression(dims[0], dims[1], learning_rate, gate, name='bp')
    bp_mom = BackPropRegression(dims[0], dims[1], learning_rate, gate, name='bp-mom',
                                optimizer=tf.train.MomentumOptimizer(
                                    learning_rate=learning_rate, momentum=0.9))
    bp_adam = BackPropRegression(dims[0], dims[1], learning_rate, gate, name='bp-adam',
                                 optimizer=tf.train.AdamOptimizer(learning_rate=learning_rate))
    bp_rms = BackPropRegression(dims[0], dims[1], learning_rate, gate, name='bp-rms',
                                optimizer=tf.train.RMSPropOptimizer(learning_rate=learning_rate))
    methods = [cp, cp_lam, cp_alt, cp_alt_lam, bp, bp_adam, bp_rms, bp_mom]
    # methods = [cp, cp_lam, cp_alt, cp_alt_lam, bp]
    train_error = np.zeros((len(methods), runs, stages * n_examples))
    outgoing_weight_track = np.zeros(train_error.shape)
    feature_matrix_track = np.zeros(train_error.shape)
    for run in range(runs):
        initial_outgoing_weight = []
        initial_feature_matrix = []

        x0, y0 = GEOFF.generate(True)
        GEOFF.W_mutate(0.5)
        GEOFF.reset_X()
        x1, y1 = GEOFF.generate()
        GEOFF.W_mutate(0.5)
        GEOFF.reset_X()
        x2, y2 = GEOFF.generate()

        train_xs = [x0, x1, x2, x0, x1, x2]
        train_ys = [y0, y1, y2, y0, y1, y2]

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for method in methods:
                initial_outgoing_weight.append(sess.run(method.outgoing_weight))
                initial_feature_matrix.append(sess.run(method.feature_matrix))
            for stage in range(stages):
                train_x = train_xs[stage]
                train_y = train_ys[stage]
                train_x = np.matrix(train_x[:, :-1])
                train_y = np.matrix(train_y).T
                for example_ind in range(train_x.shape[0]):
                    for method_ind, method in enumerate(methods):
                        resutls = method.train(sess, train_x[example_ind, :], train_y[example_ind, :])
                        error = resutls[0]
                        outgoing_weight, feature_matrix = sess.run([method.outgoing_weight, method.feature_matrix])
                        index = example_ind + stage * n_examples
                        outgoing_weight_track[method_ind, run, index] = \
                            np.sum(np.power(outgoing_weight - initial_outgoing_weight[method_ind], 2))
                        feature_matrix_track[method_ind, run, index] = \
                            np.sum(np.power(feature_matrix - initial_feature_matrix[method_ind], 2))
                        train_error[method_ind, run, index] = error
                        logger.info('run %d, stage %d, example %d, %s, error %f' %
                                    (run, stage, example_ind, labels[method_ind], error))

    file_path = 'tmp/%s_%d_%s.bin' % (tag, hidden_units, str(learning_rate))
    with open(file_path, 'wb') as f:
        pickle.dump({'error': train_error,
                     'outgoing_weight_track': outgoing_weight_track,
                     'feature_matrix_track': feature_matrix_track
                     }, f)
    return train_error

step_sizes = [0.0005]
for step_size in step_sizes:
    train(step_size, 500, 5000)



