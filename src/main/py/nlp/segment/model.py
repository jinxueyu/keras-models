import numpy as np
from tensorflow.keras.layers import Embedding
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Bidirectional, LSTM, GRU, TimeDistributed
from tensorflow.python.keras.layers import Dropout

from addons.layers.crf import CRF
from addons.losses.crf import crf_loss


NUM_CLASS = 5
MAX_LENGTH = 100
EMBEDDING_OUT_DIM = 200
HIDDEN_UNITS = 200
DROPOUT_RATE = 0.3


def build_glove_layer(embedding_glove_path, vocab_size, word_to_id_func):
    # 在下面加载学习的分布式表达式
    # 这使您可以考虑来自大型语料库的单词语义信息, 而不是训练数据
    # 0th分配给填充, (词汇+ 1)分配给未知单词
    # glove = np.concatenate([np.zeros(EMBEDDING_OUT_DIM)[np.newaxis],
    #                         np.load('../glove-wiki-300-connl.npy'),
    #                         np.zeros(EMBEDDING_OUT_DIM)[np.newaxis]],
    #                        axis=0)
    embedding_weights = np.zeros((vocab_size, EMBEDDING_OUT_DIM))
    reader = open(embedding_glove_path, 'r')
    while True:
        line = reader.readline()
        if not line:
            break
        arr = line.rstrip().split(' ')
        word = arr[0]
        if len(word) > 1:
            continue
        embedding_weights[word_to_id_func(word), :] = np.array(arr[1:], dtype='float32')
    reader.close()

    return Embedding(input_dim=vocab_size,
                     output_dim=EMBEDDING_OUT_DIM,
                     # embeddings_initializer=tf.keras.initializers.Constant(glove),
                     weights=[embedding_weights],
                     input_shape=(MAX_LENGTH,),
                     trainable=True,
                     mask_zero=True
                     )


def build_embedding_layer(vocab_size):
    return Embedding(input_dim=vocab_size,
                     output_dim=EMBEDDING_OUT_DIM,
                     input_shape=(MAX_LENGTH,),
                     # embeddings_initializer
                     # trainable=True,
                     mask_zero=True
                     )


def build_crf_layer(num_labels):
    return CRF(num_labels, name='crf_layer')


def build_lstm_layer():
    return LSTM(200, return_sequences=True)


def build_gru_layer():
    return GRU(200, return_sequences=True)


def build_bilstm_layer():
    return Bidirectional(LSTM(200, return_sequences=True), merge_mode='ave')


def build_bigru_layer():
    return Bidirectional(GRU(256, return_sequences=True))


def build_tdd_layer(num_labels):
    return TimeDistributed(Dense(num_labels, activation='softmax'))


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


def build_glove_bi_gru2_crf_model(args, word_to_id_func):
    vocab_size = args.vocab_size
    num_labels = args.num_labels
    embedding_glove_path = args.glove_path

    model = Sequential()
    embedding_layer = build_glove_layer(embedding_glove_path, vocab_size, word_to_id_func)
    model.add(embedding_layer)

    model.add(Bidirectional(GRU(256, return_sequences=True)))
    model.add(Dropout(DROPOUT_RATE))
    model.add(Bidirectional(GRU(256, return_sequences=True)))
    model.add(Dropout(DROPOUT_RATE))
    # model.add(TimeDistributed(Dense(num_labels, activation='softmax')))
    model.add(Dense(num_labels, activation='softmax'))

    crf_layer = build_crf_layer(num_labels)
    model.add(crf_layer)
    model.compile('adam', loss=crf_loss, metrics=[crf_layer.get_accuracy])

    # tf.keras.utils.plot_model(model, show_shapes=True, dpi=64)
    return model
