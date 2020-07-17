import tensorflow as tf
import tensorflow.keras.backend as K
from addons.layers.crf import CRF


class CRFAccuracy(object):
    def __init__(self, mask=False):
        self.mask = mask
        self.__name__ = 'crf_acc'

    def get_config(self):
        return {"name": self.__name__}

    def __call__(self, y_true, y_pred):
        mask_zero = tf.zeros_like(y_true)
        mask_zero = tf.cast(tf.logical_not(tf.equal(mask_zero, y_true)), tf.keras.backend.floatx())

        judge = tf.cast(tf.equal(y_pred, y_true), tf.keras.backend.floatx())
        if self.mask is False:
            return tf.reduce_mean(judge)
        else:
            return tf.reduce_sum(judge * mask_zero) / tf.reduce_sum(mask_zero)


crf_acc_mask = CRFAccuracy(True)
crf_acc = CRFAccuracy(False)


def _crf_accuracy(y_true, y_pred):
    print('crf_accuracy---' + str(dir(y_pred)))
    crf_layer = y_pred._keras_history[0]
    # check if last layer is CRF
    if not isinstance(crf_layer, CRF):
        raise ValueError(
            "Last layer must be CRF for use {}.".format('crf_loss')
        )

    accuracy = crf_layer.get_accuracy(y_true, y_pred)

    return accuracy


def _del_get_accuracy(y_true, y_pred, mask, sparse_target=False):
    y_pred = K.argmax(y_pred, -1)
    if sparse_target:
        y_true = K.cast(y_true[:, :, 0], K.dtype(y_pred))
    else:
        y_true = K.argmax(y_true, -1)

    judge = K.cast(K.equal(y_pred, y_true), K.floatx())
    if mask is None:
        return K.mean(judge)
    else:
        mask = K.cast(mask, K.floatx())
        return K.sum(judge * mask) / K.sum(mask)


def crf_viterbi_accuracy(y_true, y_pred):
    '''Use Viterbi algorithm to get best path, and compute its accuracy.
    `y_pred` must be an output from CRF.'''
    crf, idx = y_pred._keras_history[:2]
    X = crf._inbound_nodes[idx].input_tensors[0]
    mask = crf._inbound_nodes[idx].input_masks[0]
    y_pred = crf.viterbi_decoding(X, mask)
    return _get_accuracy(y_true, y_pred, mask, crf.sparse_target)


def crf_marginal_accuracy(y_true, y_pred):
    '''Use time-wise marginal argmax as prediction.
    `y_pred` must be an output from CRF with `learn_mode="marginal"`.'''
    crf, idx = y_pred._keras_history[:2]
    X = crf._inbound_nodes[idx].input_tensors[0]
    mask = crf._inbound_nodes[idx].input_masks[0]
    y_pred = crf.get_marginal_prob(X, mask)
    return _get_accuracy(y_true, y_pred, mask, crf.sparse_target)


def _crf_accuracy(y_true, y_pred):
    '''Ge default accuracy based on CRF `test_mode`.'''
    crf, idx = y_pred._keras_history[:2]
    if crf.test_mode == 'viterbi':
        return crf_viterbi_accuracy(y_true, y_pred)
    else:
        return crf_marginal_accuracy(y_true, y_pred)

