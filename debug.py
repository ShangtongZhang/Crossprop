from TFSMD import *
import tensorflow as tf

dims = [2, 2]
smd = TFSMD([2, 2], 0.1, tf.random_normal_initializer(seed=0), Relu, 'debug')
x = np.array([[1, 2]])
y = np.array([[0, 1]])
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    smd.train(sess, x, y)