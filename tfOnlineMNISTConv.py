import numpy as np
import pickle
import logging
from DynamicGEOFF import *
from TFBackprop import *
from TFCrossprop import *
from load_mnist import *

tag = 'tf_online_MNIST_conv_shift'
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

width = 28
height = 28
train_x_total, train_y_total = load_mnist('training')
train_x_total = train_x_total.reshape([-1, width, height, 1])
train_y_total = train_y_total.reshape([-1, 10])
depth_0 = 1 # channels
filter_1 = 5
depth_1 = 64
filter_2 = 5
depth_2 = 64
dim_in = width * height * depth_2 // 16
dim_hidden = 1024
dim_out = 10
dims = [width, height, depth_0, filter_1, depth_1, filter_2, depth_2]

labels = ['CP-ALT', 'CP-ALT-lambda', 'BP', 'BP-adam', 'BP-rms', 'BP-mom']
# labels = ['CP', 'CP-lambda', 'CP-ALT', 'CP-ALT-lambda', 'BP']
def train(learning_rate, n_examples):
    gate = Tanh()
    runs = 1
    stages = 4
    cp_alt_conv = ConvLayers('cp-alt-conv', dims, gate)
    cp_alt = CrossPropAlt(dim_in, dim_hidden, dim_out, learning_rate, gate,
                      output_layer='CE', lam=0, name='cp', bottom_layer=cp_alt_conv)
    cp_alt_lam_conv = ConvLayers('cp-alt-lam-conv', dims, gate)
    cp_alt_lam = CrossPropAlt(dim_in, dim_hidden, dim_out, learning_rate, gate,
                          output_layer='CE', lam=0.5, name='cp-lam',
                              bottom_layer=cp_alt_lam_conv)
    bp_conv = ConvLayers('bp-conv', dims, gate)
    bp = BackPropClissification(dim_in, dim_hidden, dim_out, learning_rate, gate,
                                name='bp', bottom_layer=bp_conv)
    bp_mom_conv = ConvLayers('bp-mom-conv', dims, gate)
    bp_mom = BackPropClissification(dim_in, dim_hidden, dim_out, learning_rate, gate, name='bp-mom',
                                    optimizer=tf.train.MomentumOptimizer(
                                        learning_rate=learning_rate, momentum=0.9
                                    ), bottom_layer=bp_mom_conv)
    bp_adam_conv = ConvLayers('bp-adam-conv', dims, gate)
    bp_adam = BackPropClissification(dim_in, dim_hidden, dim_out, learning_rate, gate, name='bp-adam',
                                     optimizer=tf.train.AdamOptimizer(learning_rate=learning_rate),
                                     bottom_layer=bp_adam_conv)
    bp_rms_conv = ConvLayers('bp-rms-conv', dims, gate)
    bp_rms = BackPropClissification(dim_in, dim_hidden, dim_out, learning_rate, gate, name='bp-rms',
                                    optimizer=tf.train.RMSPropOptimizer(learning_rate=learning_rate),
                                    bottom_layer=bp_rms_conv)
    methods = [cp_alt, cp_alt_lam, bp, bp_adam, bp_rms, bp_mom]
    # methods = [cp, cp_lam, cp_alt, cp_alt_lam, bp]
    train_error = np.zeros((len(methods), runs, stages * n_examples))
    outgoing_weight_track = np.zeros(train_error.shape)
    feature_matrix_track = np.zeros(train_error.shape)
    conv_kernel1_track = np.zeros(train_error.shape)
    conv_kernel2_track = np.zeros(train_error.shape)
    for run in range(runs):
        initial_outgoing_weight = []
        initial_feature_matrix = []
        initial_conv_kernel1 = []
        initial_conv_kernel2 = []

        train_x = train_x_total[:n_examples, :]
        train_y = train_y_total[:n_examples, :]

        y0 = train_y
        y1 = np.concatenate([train_y[:, 1:], train_y[:, :1]], 1)
        y2 = np.concatenate([y1[:, 1:], y1[:, :1]], 1)

        train_xs = [train_x] * stages
        # train_ys = [y0, y1, y2, y0, y1, y2]
        train_ys = [y0, y1, y0, y1]

        saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for method in methods:
                initial_outgoing_weight.append(sess.run(method.outgoing_weight))
                initial_feature_matrix.append(sess.run(method.feature_matrix))
                initial_conv_kernel1.append(sess.run(method.bottom_layer.conv_kernel1))
                initial_conv_kernel2.append(sess.run(method.bottom_layer.conv_kernel2))
            for stage in range(stages):
                features = np.zeros((len(methods), n_examples, dim_hidden))
                train_x = train_xs[stage]
                train_y = train_ys[stage]
                for example_ind in range(train_x.shape[0]):
                    for method_ind, method in enumerate(methods):
                        resutls = method.train(sess, train_x[[example_ind], :], train_y[[example_ind], :])
                        error = resutls[0]
                        outgoing_weight, feature_matrix, conv_kernel1, conv_kernel2 = \
                            sess.run([method.outgoing_weight, method.feature_matrix,
                                      method.bottom_layer.conv_kernel1,
                                      method.bottom_layer.conv_kernel2])
                        index = example_ind + stage * n_examples
                        outgoing_weight_track[method_ind, run, index] = \
                            np.sum(np.power(outgoing_weight - initial_outgoing_weight[method_ind], 2))
                        feature_matrix_track[method_ind, run, index] = \
                            np.sum(np.power(feature_matrix - initial_feature_matrix[method_ind], 2))
                        conv_kernel1_track[method_ind, run, index] = \
                            np.sum(np.power(conv_kernel1 - initial_conv_kernel1[method_ind], 2))
                        conv_kernel2_track[method_ind, run, index] = \
                            np.sum(np.power(conv_kernel2 - initial_conv_kernel2[method_ind], 2))

                        train_error[method_ind, run, index] = error
                        logger.info('run %d, stage %d, example %d, %s, error %f' %
                                    (run, stage, example_ind, labels[method_ind], error))
                # batch_size = 100
                # cur_example = 0
                # while cur_example < n_examples:
                #     logger.info('store features... stage %d, example %d' % (stage, cur_example))
                #     end_example = min(n_examples, cur_example + batch_size)
                #     for method_ind, method in enumerate(methods):
                #         cur_features = sess.run(method.feature, feed_dict={
                #             method.x: train_x[cur_example: end_example, :],
                #             method.target: train_y[cur_example: end_example, :]
                #         })
                #         features[method_ind, cur_example: end_example, :] = cur_features
                #     cur_example = end_example
                # with open('tmp/%s_stage_%d_saved_features.bin' % (tag, stage), 'wb') as f:
                #     pickle.dump(features, f)
                saver.save(sess, 'tmp/saved/conv_model/%s_stage_%d' % (tag, stage))

    file_path = 'tmp/%s_%s.bin' % (tag, str(learning_rate))
    with open(file_path, 'wb') as f:
        pickle.dump({'error': train_error,
                     'outgoing_weight_track': outgoing_weight_track,
                     'feature_matrix_track': feature_matrix_track,
                     'conv_kernel1_track': conv_kernel1_track,
                     'conv_kernel2_track': conv_kernel2_track
                     }, f)

step_sizes = [0.0005]
for step_size in step_sizes:
    train(step_size, 50000)



