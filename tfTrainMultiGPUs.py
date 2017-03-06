#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

import numpy as np
import tensorflow as tf
import pickle
from TFCrossprop import *
from TFBackprop import *
from load_cifar10 import *
from TFAllCNN import *
from load_mnist import *
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
fh = logging.FileHandler('log/multi_gpu.txt')
fh.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)
logger.addHandler(fh)
logger.addHandler(ch)

# MNIST = True
MNIST = False

if MNIST:
    train_x, train_y = load_mnist('training')
    test_x, test_y = load_mnist('testing')
else:
    train_x, train_y, test_x, test_y = load_cifar10('cifar10')

train_examples = train_x.shape[0]
test_examples = test_x.shape[0]
# train_examples = 1000
# test_examples = 200
train_x = train_x[: train_examples, :, :, :]
train_y = train_y[: train_examples, :]
test_x = test_x[: test_examples, :, :, :]
test_y = test_y[: test_examples, :]

if MNIST:
    _, width, height, _ = train_x.shape
    depth_0 = 1 # channels
    filter_1 = 5
    depth_1 = 64
    filter_2 = 5
    depth_2 = 64
    dim_in = width * height * depth_2 // 16
    dim_hidden = 1024
    dim_out = 10
    dims = [width, height, depth_0, filter_1, depth_1, filter_2, depth_2]
    # gate = Tanh()
    gate = Relu()
    tag = 'MNIST'
    initialzer = tf.random_normal_initializer()
else:
    dim_in = 8 * 8 * 10
    dim_hidden = 320
    dim_out = 10
    gate = Relu()
    # gate = Tanh()
    tag = 'CIFAR10'
    initialzer = orthogonal_initializer(np.sqrt(2.0))

num_gpus = 16
epochs = 200
batch_size = 200

def train_cp(learning_rate):
    with tf.Graph().as_default(), tf.device('/cpu:0'):
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        tower_grads = []
        tower_infos = []
        towers = []
        with tf.variable_scope(tf.get_variable_scope()):
            for i in range(num_gpus):
                with tf.device('/gpu:%d' % i):
                    with tf.name_scope('corssprop_%d' % i):
                        if MNIST:
                            bottom_layer = ConvLayers('cp-conv', dims, gate, initialzer)
                        else:
                            bottom_layer = AllCNN('cp-AllCNN', gate, initialzer)
                        cp = CrossPropClassification(dim_in, dim_hidden, dim_out, learning_rate,
                                                     gate=gate, initializer=initialzer,
                                                     bottom_layer=bottom_layer,
                                                     optimizer=optimizer)
                        towers.append(cp)
                        tf.get_variable_scope().reuse_variables()
                        tower_grads.append(cp.all_gradients)
                        tower_infos.append(cp.other_info)
        grads = average_gradients(tower_grads)
        total_loss, correct_labels, h_decay, h_delta = sum_info(tower_infos)
        train_op = optimizer.apply_gradients(grads)

        init = tf.global_variables_initializer()
        sess = tf.Session(config=tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=False))
        sess.run(init)

        h = np.zeros((dim_hidden, dim_out))

        train_loss = np.zeros(epochs)
        train_acc = np.zeros(epochs)
        test_loss = np.zeros(epochs)
        test_acc = np.zeros(epochs)
        for ep in range(epochs):
            batch_begin = 0
            while batch_begin < train_examples:
                batch_end = min(batch_begin + batch_size, train_examples)
                feed_dict = get_feed_dict(batch_begin, batch_end, num_gpus, train_x, train_y, h, towers)
                _, total_loss_var, correct_labels_var, h_decay_var, h_delta_var = \
                    sess.run([train_op, total_loss, correct_labels, h_decay, h_delta],
                             feed_dict=feed_dict)
                h = np.multiply(h, h_decay_var / batch_size) - learning_rate * h_delta_var / batch_size
                batch_begin = batch_end
                logger.debug('%f', total_loss_var)
                train_loss[ep] += total_loss_var
                train_acc[ep] += correct_labels_var
            train_loss[ep] /= train_examples
            train_acc[ep] /= train_examples

            batch_begin = 0
            while batch_begin < test_examples:
                batch_end = min(batch_begin + batch_size, test_examples)
                feed_dict = get_feed_dict(batch_begin, batch_end, num_gpus, test_x, test_y, h, towers)
                total_loss_var, correct_labels_var = \
                    sess.run([total_loss, correct_labels], feed_dict=feed_dict)
                batch_begin = batch_end
                test_loss[ep] += total_loss_var
                test_acc[ep] += correct_labels_var
            test_loss[ep] /= test_examples
            test_acc[ep] /= test_examples

            logger.info('CP: Epoch %d Train %f %f Test %f %f',
                        ep, train_loss[ep], train_acc[ep], test_loss[ep], test_acc[ep])
    return train_loss, train_acc, test_loss, test_acc

def train_bp(learning_rate):
    with tf.Graph().as_default(), tf.device('/cpu:0'):
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        tower_grads = []
        tower_infos = []
        towers = []
        with tf.variable_scope(tf.get_variable_scope()):
            for i in range(num_gpus):
                with tf.device('/gpu:%d' % i):
                    with tf.name_scope('corssprop_%d' % i):
                        if MNIST:
                            bottom_layer = ConvLayers('bp-conv', dims, gate, initialzer)
                        else:
                            bottom_layer = AllCNN('bp-AllCNN', gate, initialzer)
                        bp = BackPropClissification(dim_in, dim_hidden, dim_out, learning_rate,
                                                     gate=gate, initializer=initialzer,
                                                     bottom_layer=bottom_layer,
                                                     optimizer=optimizer)
                        towers.append(bp)
                        tf.get_variable_scope().reuse_variables()
                        tower_grads.append(bp.all_gradients)
                        tower_infos.append(bp.other_info)
        grads = average_gradients(tower_grads)
        total_loss, correct_labels = sum_info(tower_infos)
        train_op = optimizer.apply_gradients(grads)

        init = tf.global_variables_initializer()
        sess = tf.Session(config=tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=False))
        sess.run(init)

        train_loss = np.zeros(epochs)
        train_acc = np.zeros(epochs)
        test_loss = np.zeros(epochs)
        test_acc = np.zeros(epochs)
        for ep in range(epochs):
            batch_begin = 0
            while batch_begin < train_examples:
                batch_end = min(batch_begin + batch_size, train_examples)
                feed_dict = get_feed_dict(batch_begin, batch_end, num_gpus, train_x, train_y, None, towers)
                _, total_loss_var, correct_labels_var = \
                    sess.run([train_op, total_loss, correct_labels],
                             feed_dict=feed_dict)
                batch_begin = batch_end
                logger.debug('%f', total_loss_var)
                train_loss[ep] += total_loss_var
                train_acc[ep] += correct_labels_var
            train_loss[ep] /= train_examples
            train_acc[ep] /= train_examples

            batch_begin = 0
            while batch_begin < test_examples:
                batch_end = min(batch_begin + batch_size, test_examples)
                feed_dict = get_feed_dict(batch_begin, batch_end, num_gpus, test_x, test_y, None, towers)
                total_loss_var, correct_labels_var = \
                    sess.run([total_loss, correct_labels], feed_dict=feed_dict)
                batch_begin = batch_end
                test_loss[ep] += total_loss_var
                test_acc[ep] += correct_labels_var
            test_loss[ep] /= test_examples
            test_acc[ep] /= test_examples

            logger.info('BP: Epoch %d Train %f %f Test %f %f',
                        ep, train_loss[ep], train_acc[ep], test_loss[ep], test_acc[ep])
    return train_loss, train_acc, test_loss, test_acc

def train(learning_rate):
    labels = ['bp', 'cp']
    train_fn = [train_bp, train_cp]
    runs = 1
    train_loss = np.zeros((len(labels), runs, epochs))
    train_acc = np.zeros(train_loss.shape)
    test_loss = np.zeros(train_loss.shape)
    test_acc = np.zeros(train_loss.shape)
    for run in range(runs):
        for i in range(len(labels)):
            train_loss[i, run, :], train_acc[i, run, :], test_loss[i, run, :], test_acc[i, run, :] = \
                train_fn[i](learning_rate)
    fw = open('data/'+tag+'_'+str(learning_rate)+'.bin', 'wb')
    pickle.dump({'lr': learning_rate,
             'bs': batch_size,
             'stats': [train_loss, train_acc, test_loss, test_acc]}, fw)
    fw.close()

learning_rates = np.power(2.0, np.arange(-13, -5))
for lr in learning_rates:
    train(lr)
# train_cp(2.0 ** -5)