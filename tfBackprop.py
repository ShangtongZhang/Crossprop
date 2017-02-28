import tensorflow as tf
import numpy as np
from functools import partial
from multiprocessing import Pool, Process
import pickle
from CrosspropLearner import *
from BackpropLearner import *
from GEOFF import *
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import nn_ops

def fully_connected(name, label, var_in, dim_in, dim_out,
                    initializer, transfer, reuse=False):
  """Standard fully connected layer"""
  with variable_scope.variable_scope(name, reuse=reuse):
    with variable_scope.variable_scope(label, reuse=reuse):
      if reuse:
        W = variable_scope.get_variable("W", [dim_in, dim_out])
        b = variable_scope.get_variable("b", [dim_out])
      else: # new
        W = variable_scope.get_variable("W", [dim_in, dim_out],
                                        initializer=initializer)
        b = variable_scope.get_variable("b", [dim_out],
                                        initializer=initializer)
  z_hat = math_ops.matmul(var_in, W)
  z_hat = nn_ops.bias_add(z_hat, b)
  y_hat = transfer(z_hat)
  return W, b, z_hat, y_hat

class model_f_f():
  """Define an cp_cp_f model: input -> fully connected -> output"""
  def __init__(self, name, dimensions, gate_fun, loss_fun):
      # placeholders
    dim_in, hidden, dim_out = dimensions
    self.x = tf.placeholder(tf.float32, shape=(None, dim_in)) # input
    self.y = tf.placeholder(tf.float32, shape=(None, dim_out)) # target
      # layer 1: full
    W_1, b_1, z_hat_1, y_hat_1 = fully_connected(
        name, "layer_1", self.x, dim_in, hidden,
        tf.random_normal_initializer(),
        gate_fun)
      # layer 2: full
    W_2, b_2, z_hat, y_hat = fully_connected(
        name, "layer_2", y_hat_1, hidden, dim_out,
        tf.random_normal_initializer(),
        tf.identity)
    self.train_loss = tf.reduce_sum(loss_fun(z_hat, self.y))
    self.train_vars = [W_1, b_1, W_2, b_2]
    self.name = name

runs = 1
epochs = 200

# labels = ['bp-Adam', 'bp-RMSProp']

labels = ['bp-Adam']

def train(step, hiddenUnits, nSample):
    trainErrors = np.zeros((len(labels), runs, epochs))
    testErrors = np.zeros(trainErrors.shape)

    fr = open('GEOFF'+str(nSample)+'.bin', 'rb')
    __, _, data = pickle.load(fr)
    fr.close()
    sep = nSample - 500

    for run in range(runs):
        X, Y = data[run]
        trainX = X[: sep]
        trainY = Y[: sep]
        trainX = trainX[:, : -1]
        testX = X[sep:]
        testX = testX[:, : -1]
        testY = Y[sep:]

        dimensions = [20, hiddenUnits, 1]
        gate_fun = tf.nn.relu
        loss_fun = lambda x, y: tf.scalar_mul(0.5, tf.squared_difference(x, y))

        models = []
        optimizers = []

        models.append(model_f_f(labels[0], dimensions, gate_fun, loss_fun))
        # optimizers.append(tf.train.GradientDescentOptimizer(step).minimize(models[-1].train_loss, var_list=models[-1].train_vars))
        optimizers.append(tf.train.AdamOptimizer(step).minimize(models[-1].train_loss, var_list=models[-1].train_vars))

        with tf.Session() as sess:
            for var in tf.global_variables():
                sess.run(var.initializer)
            for ind in range(len(models)):
                model = models[ind]
                optimizer = optimizers[ind]
                for ep in range(epochs):
                    indices = np.arange(trainX.shape[0])
                    np.random.shuffle(indices)
                    for i in indices:
                        feedDict = {model.x: np.matrix(trainX[i, :]), model.y: np.matrix(trainY[i])}
                        sess.run(optimizer, feed_dict=feedDict)
                        trainErrors[ind, run, ep] += sess.run(model.train_loss, feed_dict=feedDict)
                    testErrors[ind, run, ep] += sess.run(model.train_loss, feed_dict={
                        model.x: np.matrix(testX),
                        model.y: np.matrix(testY).T
                    })
                    print 'run', run, model.name, 'epoch', ep, testErrors[ind, run, ep]

    fw = open('data/tf_' + labels[0] + '_' + str(hiddenUnits) + '_' + str(step) + '_' + str(nSample) + '.bin', 'wb')
    pickle.dump({'errors': [trainErrors, testErrors],
                 'stepSize': step,
                 'hiddenUnits': hiddenUnits}, fw)
    fw.close()

hiddenUnits = [100, 500, 900]
stepSizes = np.power(2., np.arange(-16, -10))
samples = [3500, 6500, 15500, 24500]
# train(stepSizes[0], hiddenUnits[0], samples[0])
train(np.power(2.0, -10), hiddenUnits[1], samples[0])

# fr = open('data/relu_total_offline_500_1.52587890625e-05_1500.bin', 'rb')
# data = pickle.load(fr)
# fr.close()
