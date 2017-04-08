import numpy as np
import pickle
import logging
from DynamicGEOFF import *
from TFBackprop import *
from TFCrossprop import *
from load_mnist import *

tag = 'tf_online_MNIST'
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

dim_in = 28 * 28
dim_hidden = 1024
dim_out = 10
dims = [dim_in, dim_hidden, dim_out]

train_x_total, train_y_total = load_mnist('training')
train_x_total = train_x_total.reshape([-1, dim_in])
train_y_total = train_y_total.reshape([-1, 10])

labels = ['CP-ALT', 'CP-ALT-lambda', 'BP', 'BP-adam', 'BP-rms', 'BP-mom']
# labels = ['CP', 'CP-lambda', 'CP-ALT', 'CP-ALT-lambda', 'BP']
def train(learning_rate, n_examples):
    gate = Tanh()
    runs = 1
    stages = 6
    cp_alt = CrossPropAlt(dim_in, dim_hidden, dim_out, learning_rate, gate,
                      output_layer='CE', lam=0, name='cp')
    cp_alt_lam = CrossPropAlt(dim_in, dim_hidden, dim_out, learning_rate, gate,
                          output_layer='CE', lam=0.5, name='cp-lam')
    bp = BackPropClissification(dim_in, dim_hidden, dim_out, learning_rate, gate,
                                name='bp')
    bp_mom = BackPropClissification(dim_in, dim_hidden, dim_out, learning_rate, gate, name='bp-mom',
                                    optimizer=tf.train.MomentumOptimizer(
                                        learning_rate=learning_rate, momentum=0.9
                                    ))
    bp_adam = BackPropClissification(dim_in, dim_hidden, dim_out, learning_rate, gate, name='bp-adam',
                                     optimizer=tf.train.AdamOptimizer(learning_rate=learning_rate))
    bp_rms = BackPropClissification(dim_in, dim_hidden, dim_out, learning_rate, gate, name='bp-rms',
                                    optimizer=tf.train.RMSPropOptimizer(learning_rate=learning_rate))
    methods = [cp_alt, cp_alt_lam, bp, bp_adam, bp_rms, bp_mom]
    # methods = [cp, cp_lam, cp_alt, cp_alt_lam, bp]
    train_error = np.zeros((len(methods), runs, stages * n_examples))
    outgoing_weight_track = np.zeros(train_error.shape)
    feature_matrix_track = np.zeros(train_error.shape)
    for run in range(runs):
        initial_outgoing_weight = []
        initial_feature_matrix = []

        train_x = train_x_total[:n_examples, :]
        train_y = train_y_total[:n_examples, :]

        y0 = train_y
        y1 = np.concatenate([train_y[:, 1:], train_y[:, :1]], 1)
        y2 = np.concatenate([y1[:, 1:], y1[:, :1]], 1)

        train_xs = [train_x] * 6
        train_ys = [y0, y1, y2, y0, y1, y2]

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for method in methods:
                initial_outgoing_weight.append(sess.run(method.outgoing_weight))
                initial_feature_matrix.append(sess.run(method.feature_matrix))
            for stage in range(stages):
                train_x = train_xs[stage]
                train_y = train_ys[stage]
                for example_ind in range(train_x.shape[0]):
                    for method_ind, method in enumerate(methods):
                        resutls = method.train(sess, train_x[[example_ind], :], train_y[[example_ind], :])
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

    file_path = 'tmp/%s_%s.bin' % (tag, str(learning_rate))
    with open(file_path, 'wb') as f:
        pickle.dump({'error': train_error,
                     'outgoing_weight_track': outgoing_weight_track,
                     'feature_matrix_track': feature_matrix_track
                     }, f)
    return train_error

step_sizes = [0.0005]
for step_size in step_sizes:
    train(step_size, 50000)



