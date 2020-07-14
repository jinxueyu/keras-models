import tensorflow as tf

__author__ = 'xueyu'


if __name__ == '__main__':
    sequence = tf.placeholder(
            tf.float32,
            [None, 66, 88])
    max_length = int(sequence.get_shape()[1])
    print sequence.get_shape()
    print tf.slice(sequence, (0, 0, 0), (-1, max_length - 1, -1))

    t = tf.Tensor
