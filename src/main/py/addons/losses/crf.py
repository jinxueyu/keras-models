# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Implementing Conditional Random Field loss."""

from __future__ import absolute_import, division, print_function

import tensorflow as tf
from tensorflow import is_tensor, Tensor, float32
from tensorflow.keras import backend as K

from addons.layers.crf import CRF


# @tf.keras.utils.register_keras_serializable(package="Addons")
class ConditionalRandomFieldLoss(object):
    def __init__(self, name: str = "crf_loss"):
        self.name = name
        self.__name__ = name

    def get_config(self):
        return {"name": self.name}

    def __call__(self, y_true, y_pred, sample_weight=None):
        crf_layer = y_pred._keras_history[0]

        # check if last layer is CRF
        if not isinstance(crf_layer, CRF):
            raise ValueError(
                "Last layer must be CRF for use {}.".format(self.__class__.__name__)
            )

        loss_vector = crf_layer.get_loss(y_true, y_pred)

        return tf.keras.backend.mean(loss_vector)

# def crf_loss(y_true, y_pred):
#     """General CRF loss function depending on the learning mode.
#
#     # Arguments
#         y_true: tensor with true targets.
#         y_pred: tensor with predicted targets.
#
#     # Returns
#         If the CRF layer is being trained in the join mode, returns the negative
#         log-likelihood. Otherwise returns the categorical crossentropy implemented
#         by the underlying Keras backend.
#
#     # About GitHub
#         If you open an issue or a pull request about CRF, please
#         add `cc @lzfelix` to notify Luiz Felix.
#     """
#     crf, idx = y_pred._keras_history[:2]
#     if crf.learn_mode == 'join':
#         return crf_nll(y_true, y_pred)
#     else:
#         if crf.sparse_target:
#             return sparse_categorical_crossentropy(y_true, y_pred)
#         else:
#             return categorical_crossentropy(y_true, y_pred)


def crf_loss(y_true, y_pred):
    crf_layer = y_pred._keras_history[0]

    # check if last layer is CRF
    if not isinstance(crf_layer, CRF):
        raise ValueError(
            "Last layer must be CRF for use {}.".format('crf_loss')
        )

    loss_vector = crf_layer.get_loss(y_true, y_pred)

    return tf.keras.backend.mean(loss_vector)

# crf_loss = ConditionalRandomFieldLoss()
