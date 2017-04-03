#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

import numpy as np
import tensorflow as tf
import pickle
import logging
from TFCrossprop import *
from TFBackprop import *
from load_mnist import *

tag = 'tf_online_MNIST_conv'
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

train_x, train_y = load_mnist('training')
test_x, test_y = load_mnist('testing')

learning_rate = 0.0001
dim_hidden = 1024
dim_out = 10

train_x = train_x.reshape([-1, 28, 28, 1])
train_y = train_y
test_x = test_x.reshape([-1, 28, 28, 1])
test_y = test_y

_, width, height, _ = train_x.shape
depth_0 = 1 # channels
filter_1 = 5
depth_1 = 64
filter_2 = 5
depth_2 = 64
dim_in = width * height * depth_2 // 16
dims = [width, height, depth_0, filter_1, depth_1, filter_2, depth_2]

train_examples = 50000
labels = ['cp', 'bp']
gate = Tanh()
initializer = tf.random_normal_initializer()

cpConvLayers = ConvLayers('cp-conv', dims, gate, initializer)
cp = CrossPropClassification(dim_in, dim_hidden, dim_out, learning_rate,
                             gate=gate, initializer=initializer,
                             bottom_layer=cpConvLayers)
bpConvLayers = ConvLayers('bp-conv', dims, gate, initializer)
bp = BackPropClissification(dim_in, dim_hidden, dim_out, learning_rate,
                            gate=gate, initializer=initializer,
                            bottom_layer=bpConvLayers)
methods = [cp, bp]
train_loss = np.zeros((len(methods), train_examples))
train_acc = np.zeros(train_loss.shape)
U_norm = np.zeros(train_loss.shape)
W_norm = np.zeros(train_loss.shape)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    initial_U = [sess.run(cp.U), sess.run(bp.U)]
    initial_W = [sess.run(cp.W), sess.run(bp.W)]
    for example_ind in range(train_examples):
        for method_ind, method in enumerate(methods):
            loss, acc = method.train(sess, train_x[example_ind: example_ind + 1, :], train_y[example_ind: example_ind + 1, :])
            train_loss[method_ind, example_ind] = loss
            train_acc[method_ind, example_ind] = acc
            current_U = sess.run(method.U)
            current_W = sess.run(method.W)
            U_norm[method_ind, example_ind] = np.mean(np.power(current_U - initial_U[method_ind], 2))
            W_norm[method_ind, example_ind] = np.mean(np.power(current_W - initial_W[method_ind], 2))
            logger.info('example %d %s loss:%f acc:%f' %
                        (example_ind, labels[method_ind],
                         np.mean(train_loss[method_ind, :example_ind + 1]),
                         np.mean(train_acc[method_ind, :example_ind + 1])))

path = 'tmp/%s_%s.bin' % (tag, str(learning_rate))
with open(path, 'wb') as f:
    pickle.dump({'loss': train_loss,
                 'acc': train_acc,
                 'U': U_norm,
                 'W': W_norm}, f)


