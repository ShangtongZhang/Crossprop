import tensorflow as tf

def layer(name, dim, initializer):
    with tf.variable_scope('inner'):
        W = 0
        # W = tf.get_variable('W', [dim], initializer=initializer)
        P = tf.placeholder(dtype=tf.float32, name='P', shape=(None, 2))
    return W, P

initializer = tf.random_normal_initializer()
name = 'vs'
with tf.name_scope(name) as scope:
    # scope.reuse_variables()
    W, P = layer(name, 3, initializer)
    # print W.name
    print P.name
with tf.name_scope(name) as scope:
    # scope.reuse_variables()
    W, P2 = layer(name, 3, initializer)
    # print W.name
    print P.name
    print P2.name

# with tf.Session() as sess:
#     for var in tf.global_variables():
#         sess.run(var.initializer)
#     print sess.run([W, W2])