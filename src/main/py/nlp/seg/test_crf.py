import configparser

import numpy as np
import tensorflow
from tensorflow.keras import regularizers, activations, optimizers
from tensorflow.keras.layers import Embedding
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Bidirectional, LSTM, GRU, TimeDistributed, Dropout, Activation
from tensorflow.keras import initializers

import os

from tensorflow.python.keras.utils.np_utils import to_categorical

from nlp.corpus.reader import DataProcessor
from util.object_dict import build_object_dict


def get_config():
    config = configparser.ConfigParser()
    config.read(os.path.join('', "model-args-config.ini"))

    data_path = '/Users/xueyu/Workspace/data'
    model_path = 'nlp/model/keras-models'
    corpus_path = 'nlp/corpus/icwb2-data/training/msr_training.utf8'
    glove_path = 'nlp/embidding/glove/tencent/Tencent_AILab_ChineseEmbedding_Single.txt'

    config_dict = {
        'data_path':    data_path,
        'model_path':   model_path,
        'corpus_path':  corpus_path,
        'glove_path':   glove_path
    }

    return build_object_dict(config_dict)


def build_lstm_model(input_dim, input_length, num_labels, lstm_units):
    model = Sequential()

    model.add(Embedding(input_dim, lstm_units, input_length=input_length, mask_zero=True))
    model.add(LSTM(lstm_units, return_sequences=True))
    model.add(LSTM(lstm_units, return_sequences=True))
    model.add(Dropout(0.5))
    model.add(Dense(num_labels))
    model.add(Activation('softmax'))

    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )

    print(model.summary())

    return model


def train_lstm():
    config = get_config()
    data_path = os.path.join(config.data_path, config.corpus_path)
    dataset = DataProcessor()
    input_dim = dataset.vocab_size
    input_length = 100
    num_labels = 5
    lstm_units = 128

    model = build_lstm_model(input_dim, input_length, num_labels, lstm_units)

    train_x, train_y = dataset.process_input_data(data_path, maxlen=input_length)

    train_y = to_categorical(train_y)

    history = model.fit(
        train_x, train_y,
        batch_size=512,
        epochs=20,
        validation_split=0.1
    )


if __name__ == '__main__':
    train_lstm()
