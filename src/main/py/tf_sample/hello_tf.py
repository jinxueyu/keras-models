import tensorflow as tf
from tensorflow.contrib import keras

__author__ = 'xueyu'


if __name__ == '__main__':

    a = tf.constant([3, 5], name='input_a')
    # b = tf.constant(5, name='input_b')

    c = tf.reduce_prod(a, name='node_c')
    d = tf.reduce_sum(a, name='node_d')

    e = tf.add(c, d, name='node_e')

    sess = tf.Session()
    output = sess.run(e)

    print output

    writer = tf.summary.FileWriter('log/hello/hello_tf', sess.graph)
    writer.close()

    sess.close()
