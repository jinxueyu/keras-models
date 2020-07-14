import tensorflow as tf
import numpy as np
from tensorflow import Variable




__author__ = 'xueyu'

if __name__ == '__main__':
    sess = tf.Session()
    # a = Variable(tf.zeros([2, 1]), name = "weigth")
    a = np.array([[1], [2], [3]], dtype=np.int32)
    b = np.array([[1, 2, 3]], dtype=np.int32)

    print a, '\n', b
    c = tf.matmul(b, a)
    print sess.run(c)

    sess.close()
