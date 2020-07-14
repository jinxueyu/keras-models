# -*- coding: utf-8 -*-
import random

from gensim.models import Word2Vec
from keras import Sequential
import keras
from keras.layers import Embedding, LSTM, Dropout, Dense, Activation, GRU
from keras.optimizers import RMSprop
from keras.utils import np_utils
import numpy as np

__author__ = 'xueyu'


def load_embedding_weights():
    word2vec_model = Word2Vec.load('/Users/xueyu/Workshop/beehive/learning/explore/src/tf_sample/data_model/word2vec.model')

    key_vector = word2vec_model.wv
    length_of_vocab = len(key_vector.vocab) + 1
    vocab_dim = 300

    n_symbols = length_of_vocab
    embedding_weights = np.zeros((n_symbols, vocab_dim))

    word_vector = {}
    word_idx = {'###': 0}
    idx_word = {0: '###'}
    idx = 1
    for word in key_vector.vocab:
        word_vector[word] = key_vector[word]
        word_idx[word] = idx
        idx_word[idx] = word
        embedding_weights[idx, :] = word_vector[word]

        idx += 1
    return n_symbols, vocab_dim, word_idx, idx_word, embedding_weights


def read_poems():
    reader = open('/Users/xueyu/Workshop/beehive/learning/explore/src/tf_sample/data_poem/poem_format.csv', 'r')
    line_sent = []
    while True:
        line = reader.readline()
        if not line:
            break
        line = line.strip()
        arr = line.split('\t')
        year = arr[3]
        if len(arr) < 5:
            continue
        type = arr[4]
        if '五' not in type:
            continue

        if year != '唐代':
            continue
        title = arr[0]
        content = arr[1]
        author = arr[2]
        if author != '李白':
            continue

        l = [s.encode('utf8') for s in content.decode('utf8')]
        line_sent.append(l)
    reader.close()
    return line_sent

length_of_window = 7


def prepare_train_data(word_idx):
    print 'prepare_train_data'
    poem_list = read_poems()
    count = 0
    dataX = []
    dataY = []
    for poem in poem_list:
        print ''.join(poem)
        length_of_poem = len(poem)
        for i in range(0, length_of_poem - length_of_window, 1):
            seq_in = poem[i: i + length_of_window]
            seq_out = poem[i + length_of_window]
            dataX.append([word_idx[x] for x in seq_in])
            dataY.append(word_idx[seq_out])

        count += 1
        if count > 1000:
            break

    train_data = np.array(dataX)
    validation_data = np_utils.to_categorical(np.array(dataY))

    print count, train_data.shape, validation_data.shape
    return train_data, validation_data


def create_lstm_model(p_n_symbols, vocab_dim, input_length, p_embedding_weights):
    print u'创建模型...',p_n_symbols, vocab_dim, input_length
    lstm_model = Sequential()
    # embedding = Embedding(input_dim=p_n_symbols,
    #                       output_dim=vocab_dim,
    #                       input_length=input_length,
    #                       mask_zero=True,
    #                       weights=[p_embedding_weights],
    #                       trainable=False,
    #                       dropout=0.2)
    #
    # lstm_model.add(embedding)
    #
    # lstm_model.add(LSTM(100, dropout_W=0.2, dropout_U=0.2))  # try using a GRU instead, for fun
    # lstm_model.add(Dense(1))
    # lstm_model.add(Activation('sigmoid'))
    # lstm_model.add(Dense(7719, activation='softmax'))
    # # try using different optimizers and different optimizer configs
    # print u'编译模型...'
    # lstm_model.compile(loss='categorical_crossentropy',
    #                    optimizer='adam',
    #                    metrics=['accuracy'])

    lstm_model.add(Embedding(p_n_symbols, 300, weights=[p_embedding_weights]))
    lstm_model.add(GRU(300))
    lstm_model.add(Dense(2667))
    lstm_model.add(Activation('softmax'))
    lstm_model.compile(loss='categorical_crossentropy', optimizer='adam')

    return lstm_model


def train_lstm_model(model, p_X_train, p_y_train, batch_size, n_epoch):
    print u"训练..."
    model.fit(p_X_train, p_y_train,
              batch_size=batch_size,
              epochs=n_epoch)


def evaluate_lstm_model(model, p_X_test, p_y_test):
    print u"评估..."
    score, acc = model.evaluate(p_X_test, p_y_test, batch_size=batch_size)
    print 'Test score:', score
    print 'Test accuracy:', acc


def save_lstm_model(model, filepath):
    model.save(filepath, overwrite=True)


def load_lstm_model(filepath):
    return keras.models.load_model(filepath)


def sample(preds, diversity=1.0):
        # sample from te given prediction
        preds = np.asarray(preds).astype('float64')
        preds = np.log(preds) / diversity
        exp_preds = np.exp(preds)
        preds = exp_preds / np.sum(exp_preds)
        probas = np.random.multinomial(1, preds, 1)

        return np.argmax(probas)


def predict_lstm_model(words, model, word_idx, idx_word):
    words = [s.encode('utf8') for s in words.decode('utf8')]
    word_idx_list = [word_idx[word] for word in words]

    for j in range(0, 400):
        #print 'input', j, ''.join([idx_word[w] for w in word_idx_list[-length_of_window:]])

        proba = model.predict(np.array([word_idx_list[-length_of_window:]]), verbose=0)
        predicted = np.argsort(proba[0])[-5:].tolist()
        predicted.reverse()
        for i in range(0, 5):
            w = predicted[i]
            if idx_word[w] in ['，', '。']:
                word_idx_list.append(w)
                break
            else:
                w = predicted[random.randint(0, 4)]
                word_idx_list.append(w)
                break
        #print idx_word[word_idx_list[-1]]

    print "".join(idx_word[i] for i in word_idx_list).replace('。', '。\n')


if __name__ == '__main__':
    model_path = '/Users/xueyu/Workshop/beehive/learning/explore/src/tf_sample/data_model/lstm.model.wv2'
    n_symbols, vocab_dim, word_idx, idx_word, embedding_weights = load_embedding_weights()
    # train_data, validation_data = prepare_train_data(word_idx)

    # lstm_model = create_lstm_model(n_symbols, vocab_dim, length_of_window, embedding_weights)
    # print lstm_model.summary()
    # batch_size = 256
    # n_epoch = 50
    # train_lstm_model(lstm_model, train_data, validation_data, batch_size, n_epoch)
    # save_lstm_model(lstm_model, model_path)

    lstm_model = load_lstm_model(model_path)
    print lstm_model.summary()

    predict_lstm_model('待', lstm_model, word_idx, idx_word)
