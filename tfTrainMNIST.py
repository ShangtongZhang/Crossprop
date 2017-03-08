#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

# import pickle
import time
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
from tqdm import tqdm
from TFCrossprop import *
from TFBackprop import *
from load_mnist import *
from CrosspropAlternate import *

train_x, train_y = load_mnist('training')
test_x, test_y = load_mnist('testing')

# TODO: add pull MNIST from tensorflow proper
# mnist = input_data.read_data_sets('/tmp/data/', one_hot=True,
                                    # fake_data=False)

epochs = 200
batch_size = 1
learning_rate = 0.0001
dim_in = 28 * 28
dim_hidden = 1024
dim_out = 10

train_x = np.matrix(train_x.reshape([-1, dim_in])) / 255.0
train_y = np.matrix(dense_to_one_hot(train_y))
test_x = np.matrix(test_x.reshape([-1, dim_in])) / 255.0
test_y = np.matrix(dense_to_one_hot(test_y))

train_examples = 1000
test_examples = 100
train_x = train_x[: train_examples, :]
train_y = train_y[: train_examples, :]
test_x = test_x[: test_examples, :]
test_y = test_y[: test_examples, :]

initialzer = tf.random_normal_initializer()

# Create the model from TFCrossprop
crossPropModel = CrossPropClassification(dim_in, dim_hidden, dim_out, learning_rate, gate=Tanh(), initializer=initialzer)

with tf.Session() as sess:

    # add a tensorboard writer
    ts = time.time()
    logdir = '/tmp/tfTrainMNIST/logs/run' + str(ts)
    print('Log dir:' + logdir)
    writer = tf.summary.FileWriter(logdir)
    writer.add_graph(sess.graph)

    # initialize all the variables
    tf.global_variables_initializer().run()
    
    # how often to report the performance
    reportEveryN = 1

    # each epoch is an entire pass through the data
    for ep in tqdm(range(epochs)):
        for start_index in tqdm(range(0, train_examples, batch_size)):
            # another training pass on the next batch of training examples
            # build the batch
            batch_xs = train_x[start_index:start_index+batch_size, :]
            batch_ys = train_y[start_index:start_index+batch_size, :]
            result = crossPropModel.train(sess, batch_xs, batch_ys)

        # add some reporting to track learning
        if ep % reportEveryN == 0:
            # report the testing accuracy on the test set
            result = crossPropModel.test(sess, test_x, test_y)
            # pull summary statistics from the training step
            summary_str = result[0]
            loss = result[1]
            acc = result[2]
            writer.add_summary(summary_str, ep)
            print('Loss at epoch %s: %s' % (ep, loss))
            print('Accuracy at epoch %s: %s' % (ep, acc))

# close THE WRITER after
writer.close()