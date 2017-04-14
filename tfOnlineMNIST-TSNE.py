import numpy as np
import pickle
import logging
from DynamicGEOFF import *
from TFBackprop import *
from TFCrossprop import *
from load_mnist import *
from tsne import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

tag = 'tf_online_MNIST_shift'
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
def draw(learning_rate, n_examples, repeats):
    gate = Tanh()
    runs = 1
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
    for run in range(runs):

        train_x = train_x_total[:n_examples, :]
        train_y = train_y_total[:n_examples, :]

        y0 = train_y
        y1 = np.concatenate([train_y[:, 1:], train_y[:, :1]], 1)
        y2 = np.concatenate([y1[:, 1:], y1[:, :1]], 1)

        train_xs = [train_x] * 6
        train_ys = [y0, y1, y2, y0, y1, y2]

        # np.random.seed(0)
        # x0 = train_x
        # perm = np.arange(dim_in)
        # np.random.shuffle(perm)
        # x1 = train_x[:, perm]
        # np.random.shuffle(perm)
        # x2 = train_x[:, perm]
        #
        # train_xs = [x0, x1, x2, x0, x1, x2]
        # train_ys = [train_y] * 6

        # features = np.zeros((stages, len(methods), n_examples, dim_hidden))
        saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            figure_index = 0
            candidate_stages = [0, 1, 2, 3]
            candidate_methods = [0, 2]
            target_dim = 2
            tsne_data = dict()
            for stage in candidate_stages:
                saver.restore(sess, 'tmp/saved/ffn_model/%s_stage_%d' % (tag, stage))
                features = np.zeros((len(methods), n_examples, dim_hidden))
                train_x = train_xs[stage]
                train_y = train_ys[stage]
                batch_size = 1000
                cur_example = 0
                while cur_example < n_examples:
                    logger.info('store features... stage %d, example %d' % (stage, cur_example))
                    end_example = min(n_examples, cur_example + batch_size)
                    for method_ind, method in enumerate(methods):
                        cur_features = sess.run(method.feature, feed_dict={
                            method.x: train_x[cur_example: end_example, :],
                            method.target: train_y[cur_example: end_example, :]
                        })
                        features[method_ind, cur_example: end_example, :] = cur_features
                    cur_example = end_example
                sample_indices = np.arange(2500)
                for repeat in range(repeats):
                    np.random.shuffle(sample_indices)
                    for method_ind in candidate_methods:
                        x_to_plot = features[method_ind, sample_indices, :]
                        y_to_plot = np.argmax(train_y[sample_indices, :], axis=1)
                        print x_to_plot.shape, y_to_plot.shape
                        x_prime = tsne(x_to_plot, target_dim, 50, 20.0)
                        tsne_data[(stage, method_ind)] = (x_prime, y_to_plot)
                        # fig = plt.figure(figure_index)
                        # figure_index += 1
                        # ax = Axes3D(fig)
                        # ax.scatter(x_prime[:, 0], x_prime[:, 1], x_prime[:, 2], c=y_to_plot)
                        # plt.scatter(x_prime[:, 0], x_prime[:, 1], 20, y_to_plot)
                        # plt.title('%s_%s_stage_%d' % (tag, labels[method_ind], stage))
                        # plt.show()
                        # plt.savefig('figure/%s_repeat_%d_%s_stage_%d.png' % (tag, repeat, labels[method_ind], stage))
                        # plt.close()
                        # plt.show()
            with open('tmp/tsne_dim_%d.bin' % target_dim, 'wb') as f:
                pickle.dump(tsne_data, f)

draw(0.0005, 50000, 1)
