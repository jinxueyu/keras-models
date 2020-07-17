import numpy as np
from tensorflow.keras.layers import Embedding
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Bidirectional, LSTM, GRU, TimeDistributed
from tensorflow.python.keras.layers import Dropout

from addons.layers.crf import CRF
from addons.losses.crf import crf_loss
from nlp.corpus.reader import EmbeddingVectors, DataProcessor
from util import ObjectDict


def build_dense_layer(args):
    return Dense(args.input_shape, activation='softmax')


def build_glove_layer(args, dataset):
    vec_path = args.vec_path
    input_length = args.input_length
    embedding_vector = EmbeddingVectors(dataset, vec_path, 'glove')

    return Embedding(input_dim=embedding_vector.vocab_size,
                     output_dim=embedding_vector.vector_size,
                     # embeddings_initializer=tf.keras.initializers.Constant(glove),
                     weights=[embedding_vector.weights],
                     input_length=input_length,
                     trainable=args.trainable,
                     mask_zero=args.mask
                     )


def build_embedding_layer(args):

    return Embedding(input_dim=args.vocab_size,
                     output_dim=args.output_dim,
                     input_shape=args.input_shape,
                     # embeddings_initializer
                     # trainable=True,
                     mask_zero=True
                     )


def build_crf_layer(args):
    return CRF(args.output_dim, name='crf_layer')


def build_lstm_layer(args):
    return LSTM(args.hidden_size, return_sequences=True)


def build_gru_layer(args):
    return GRU(args.hidden_size, return_sequences=True)


def build_bilstm_layer(args):
    return Bidirectional(LSTM(args.hidden_size, return_sequences=True), merge_mode='ave')


def build_bigru_layer(args):
    # dropout ，应用于输入的第一个操作
    # recurrent_dropout ，应用于循环输入的其他操作（先前的输出和 / 或状态）
    return Bidirectional(GRU(args.hidden_size, return_sequences=True))


def build_tdd_layer(args):
    return TimeDistributed(Dense(args.num_labels, activation='softmax'))


def build_crf_model(vocab_size, num_labels):
    model = Sequential()

    embedding_layer = build_embedding_layer(vocab_size)
    model.add(embedding_layer)
    # model.add(Bidirectional(LSTM(HIDDEN_UNITS // 2, return_sequences=True)))
    # model.add(Dense(NUM_CLASS, activation='softmax'))

    # model.add(Bidirectional(LSTM(NUM_CLASS, return_sequences=True, activation="tanh"), merge_mode='sum'))
    # model.add(Bidirectional(LSTM(NUM_CLASS, return_sequences=True, activation="softmax"), merge_mode='sum'))

    crf_layer = build_crf_layer(num_labels)
    model.add(crf_layer)
    model.compile('adam', loss=crf_loss, metrics=[crf_layer.get_accuracy])
    # model.compile(loss=crf_layer.loss, optimizer='adam', metrics=[crf_layer.accuracy])

    return model


def build_glove_crf_model(vocab_size, num_labels, word_to_id_func):
    model = Sequential()

    embedding_layer = build_glove_layer(vocab_size, word_to_id_func)
    model.add(embedding_layer)

    crf_layer = build_crf_layer(num_labels)
    model.add(crf_layer)
    model.compile('adam', loss=crf_loss, metrics=[crf_layer.get_accuracy])
    return model


def build_glove_bi_lstm_crf_model(vocab_size, num_labels, word_to_id_func):
    model = Sequential()
    # 在“# 嵌入”层中, 将单词索引替换为相应的单词分布表达式
    # todo mask_zero = True不起作用
    embedding_layer = build_glove_layer(vocab_size, word_to_id_func)
    model.add(embedding_layer)
    # 添加具有200个输出尺寸的BiLSTM层
    # 通过将return_sequences = True传递给
    # LSTM类初始化程序,
    # 您可以检索所有
    # words 的预测值 如果
    # return_sequences = False, 则每个语句仅输出一个预测值
    model.add(Bidirectional(LSTM(200, return_sequences=True), merge_mode='ave'))
    # 添加一个完全连接的层, 其输出维数是标签类型的数量
    # 还将激活功能设置为softmax
    # 这是因为CRF层要求每个标签具有可预测的置信度
    model.add(Dense(num_labels, activation='softmax'))

    crf_layer = build_crf_layer(num_labels)
    model.add(crf_layer)
    model.compile('adam', loss=crf_loss, metrics=[crf_layer.get_accuracy])

    return model


def build_glove_bi_gru_crf_model(vocab_size, num_labels, word_to_id_func):
    model = Sequential()
    embedding_layer = build_glove_layer(vocab_size, word_to_id_func)
    model.add(embedding_layer)

    model.add(Bidirectional(GRU(200, return_sequences=True), merge_mode='ave'))
    model.add(Dense(num_labels, activation='softmax'))

    crf_layer = build_crf_layer(num_labels)
    model.add(crf_layer)
    model.compile('adam', loss=crf_loss, metrics=[crf_layer.get_accuracy])

    return model


def build_dropout_layer(args):
    return Dropout(args.dropout_rate)


def build_glove_bi_gru2_crf_model(args, dataset):
    DROPOUT_RATE = 0.3
    vocab_size = args.vocab_size
    num_labels = args.num_labels
    embedding_glove_path = args.glove_path

    model = Sequential()
    embedding_layer = build_glove_layer(args, dataset)
    model.add(embedding_layer)

    model.add(Bidirectional(GRU(256, return_sequences=True)))
    model.add()
    model.add(Bidirectional(GRU(256, return_sequences=True)))
    model.add(Dropout(DROPOUT_RATE))
    # model.add(TimeDistributed(Dense(num_labels, activation='softmax')))
    model.add(Dense(num_labels, activation='softmax'))

    crf_layer = build_crf_layer(args)
    model.add(crf_layer)

    model.compile('adam', loss=crf_loss, metrics=[crf_layer.get_accuracy])

    # tf.keras.utils.plot_model(model, show_shapes=True, dpi=64)
    return model


def build_laysers():
    layers = {
        'glove':   build_glove_layer,
        'bigru':   build_bigru_layer,
        'dense':   build_dense_layer,
        'dropout': build_dropout_layer,
        'crf':     build_crf_layer
    }

    return layers


def build_model(args, dataset):
    layer_dict = build_laysers()
    model = Sequential()
    for layer_args in args.layers:
        build_layer_func = layer_dict[layer_args.name]
        print('build layer:'+layer_args.name)
        if layer_args.name is 'glove':
            model_layer = build_layer_func(layer_args, dataset)
        else:
            model_layer = build_layer_func(layer_args)
        model.add(model_layer)

    model.compile(optimizer=args.optimizer, loss=args.loss, metrics=args.metrics)

    return model


