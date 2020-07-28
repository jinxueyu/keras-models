import numpy as np
import tensorflow
from tensorflow.keras import regularizers, activations, optimizers
from tensorflow.keras.layers import Embedding
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Bidirectional, LSTM, GRU, TimeDistributed
from tensorflow.keras import initializers
from tensorflow.keras.layers import Dropout

from addons.layers.crf import CRF
from addons.losses.crf import crf_loss
from addons.metrics.crf import crf_acc_mask
from nlp.corpus.reader import EmbeddingVectors
import matplotlib.pyplot as plt


def build_kernel_initializer(args):
    if args is None:
        return None
    kernel_initializer = None
    if args.name == 'RandomUniform':
        kernel_initializer = initializers.RandomUniform(args.minval, args.maxval)

    return kernel_initializer


def build_kernel_regularizer(args):
    if args is None:
        return None
    kernel_regularizer = None
    if args.name == 'l1':
       kernel_regularizer = regularizers.l1(l=args.l)
    elif args.name == 'l2':
       kernel_regularizer = regularizers.l2(l=args.l)
    elif args.name == 'l1l2':
        kernel_regularizer = regularizers.l1_l2(l1=args.l1, l2=args.l2)

    return kernel_regularizer


def build_optimizer(args):
    optimizer = None
    if args.name == 'adam':
        optimizer = optimizers.Adam(
            learning_rate=args.learning_rate
        )
    return optimizer


def build_loss(args):
    loss = None
    if args.name == 'crf_loss':
        return crf_loss

    return loss


def build_metrics(args):
    metrics = []
    for m in args:
        if m.name == 'crf_acc_mask':
            metrics.append(crf_acc_mask)

    return metrics


def build_dense_layer(args):
    # as first layer in a sequential model:
    # model = Sequential()
    # model.add(Dense(32, input_shape=(16,)))
    # now the model will take as input arrays of shape (*, 16)
    # and output arrays of shape (*, 32)
    # after the first layer, you don't need to specify
    # the size of the input anymore:
    # model.add(Dense(32))

    # 就是常用的的全连接层。
    #
    # Dense 实现以下操作： output = activation(dot(input, kernel) + bias) 其中 activation 是按逐个元素计算的激活函数，
    # kernel 是由网络层创建的权值矩阵，以及 bias 是其创建的偏置向量 (只在 use_bias 为 True 时才有用)。

    return Dense(
        units=args.units,
        # input_shape=args.input_shape if hasattr(args, 'input_shape') else None,
        activation=args.activation if hasattr(args, 'activation') else None,  # 'softmax',
        kernel_initializer=build_kernel_initializer(args.kernel_initializer if hasattr(args, 'kernel_initializer') else None),  # 'glorot_uniform',
        kernel_regularizer=build_kernel_regularizer(args.kernel_regularizer if hasattr(args, 'kernel_regularizer') else None)
    )


def build_glove_layer(args, dataset):
    vec_path = args.vec_path
    input_length = args.input_length
    embedding_vector = EmbeddingVectors(dataset, vec_path, 'glove')
    return Embedding(
        input_dim=embedding_vector.vocab_size,
        output_dim=embedding_vector.vector_size,
        weights=[embedding_vector.weights],
        input_length=input_length,
        # embeddings_initializer=tf.keras.initializers.Constant(glove),
        trainable=args.trainable,
        mask_zero=args.mask
    )


def build_embedding_layer(args):
    # input_dim：大或等于0的整数，字典长度，即输入数据最大下标 + 1
    # output_dim：大于0的整数，代表全连接嵌入的维度
    # embeddings_initializer: 嵌入矩阵的初始化方法，为预定义初始化方法名的字符串，或用于初始化权重的初始化器。参考initializers
    # embeddings_regularizer: 嵌入矩阵的正则项，为Regularizer对象
    # embeddings_constraint:  嵌入矩阵的约束项，为Constraints对象
    # activity_regularizer
    # mask_zero：布尔值，确定是否将输入中的‘0’看作是应该被忽略的‘填充’（padding）值，该参数在使用递归层处理变长输入时有用。设置为True的话，
    #     模型中后续的层必须都支持masking，否则会抛出异常。如果该值为True，则下标0在字典中不可用，input_dim应设置为 | vocabulary | + 2。

    return Embedding(
        input_dim=args.input_dim,
        output_dim=args.output_dim,
        input_length=args.input_length,
        embeddings_initializer=build_kernel_initializer(args.embeddings_initializer),
        trainable=args.trainable,
        mask_zero=args.mask
    )


def build_crf_layer(args):
    return CRF(args.output_dim, name='crf_layer')


def build_lstm_layer(args):
    return LSTM(args.hidden_size, return_sequences=True)


def build_gru_layer(args):
    return GRU(
        args.hidden_size,
        return_sequences=True
    )


def build_bilstm_layer(args):
    return Bidirectional(LSTM(args.hidden_size, return_sequences=True), merge_mode='ave')


def build_bigru_layer(args):
    # Bidirectional
    # merge_mode = 'concat',
    # weights = None,
    # backward_layer = None,

    # dropout ，应用于输入的第一个操作
    # recurrent_dropout ，应用于循环输入的其他操作（先前的输出和 / 或状态）
    # activation = 'tanh',
    # recurrent_activation = 'sigmoid',
    # use_bias = True,
    # kernel_initializer = 'glorot_uniform',
    # recurrent_initializer = 'orthogonal',
    # bias_initializer = 'zeros',
    # kernel_regularizer = None,
    # recurrent_regularizer = None,
    # bias_regularizer = None,
    # activity_regularizer = None,
    # kernel_constraint = None,
    # recurrent_constraint = None,
    # bias_constraint = None,
    # dropout = 0.,
    # recurrent_dropout = 0.,
    # implementation = 2,
    # return_sequences = False,
    # return_state = False,
    # go_backwards = False,
    # stateful = False,
    # unroll = False,
    # time_major = False,
    # reset_after = True,

    return Bidirectional(
        merge_mode='concat',
        layer=GRU(
            args.units,
            return_sequences=True,
            kernel_initializer=initializers.RandomUniform(minval=-0.1, maxval=0.1),
            kernel_regularizer=regularizers.l2(l=1e-4)
        )
    )


def build_tdd_layer(args):
    return TimeDistributed(Dense(args.num_labels, activation='softmax'))


def build_crf_model(vocab_size, num_labels, seq_len):
    model = Sequential()

    # embedding_layer = Embedding(
    #     input_dim=vocab_size,
    #     output_dim=128,
    #     # embeddings_initializer
    #     trainable=True,
    #     mask_zero=True
    # )
    # model.add(embedding_layer)
    # model.add(Bidirectional(LSTM(num_labels, return_sequences=True, activation="tanh"), merge_mode='sum'))
    # model.add(Bidirectional(LSTM(num_labels, return_sequences=True, activation="softmax"), merge_mode='sum'))
    # crf_layer = CRF(num_labels, name='crf_layer')
    # model.add(crf_layer)

    model.add(Embedding(input_dim=vocab_size, output_dim=128, input_shape=(seq_len,)))  # mask_zero=True　は上手くいかない
    model.add(Bidirectional(LSTM(200, return_sequences=True), merge_mode='ave'))
    model.add(Dense(num_labels, activation='softmax'))
    crf_layer = CRF(num_labels, name='crf_layer')
    model.add(crf_layer)

    model.compile('adam', loss=crf_loss, metrics=[crf_layer.get_accuracy])
    print(model.summary())

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


def build_glove_bi_gru_crf_model(dataset, vec_path, seq_len):
    num_labels = dataset.num_labels

    model = Sequential()
    embedding_vector = EmbeddingVectors(dataset, vec_path, 'glove')
    embedding_layer = Embedding(input_dim=embedding_vector.vocab_size,
                     output_dim=embedding_vector.vector_size,
                     # embeddings_initializer=tf.keras.initializers.Constant(glove),
                     weights=[embedding_vector.weights],
                     input_length=seq_len,
                     trainable=True,
                     mask_zero=True
                     )

    # embedding_layer = Embedding(
    #     input_dim=dataset.vocab_size,
    #     output_dim=128,
    #     trainable=True,
    #     mask_zero=True,
    #     input_length=seq_len
    # )

    model.add(embedding_layer)

    model.add(Bidirectional(GRU(200, return_sequences=True), merge_mode='ave'))
    model.add(Dense(num_labels, activation='softmax'))

    crf_layer = CRF(num_labels, name='crf_layer')
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
    embedding_layer = build_glove_layer(args.layers[0], dataset)
    model.add(embedding_layer)

    model.add(Bidirectional(GRU(256, return_sequences=True)))
    model.add(Bidirectional(GRU(256, return_sequences=True)))
    model.add(Dropout(DROPOUT_RATE))
    # model.add(TimeDistributed(Dense(num_labels, activation='softmax')))
    model.add(Dense(num_labels, activation='softmax'))

    crf_layer = build_crf_layer(args)
    model.add(crf_layer)

    model.compile('adam', loss=crf_loss, metrics=[crf_layer.get_accuracy])

    # tf.keras.utils.plot_model(model, show_shapes=True, dpi=64)
    return model


def build_glove_bi_gru2_crf_model(args, dataset):
    DROPOUT_RATE = 0.3

    model = Sequential()

    embedd = Embedding(
        input_dim=args.vocab_size,
        output_dim=args.output_dim,
        # input_shape=args.input_shape,
        # embeddings_initializer
        trainable=True,
        mask_zero=True
    )
    model.add(embedd)
    # model.add(build_glove_layer(args.layers[0], dataset))

    model.add(Bidirectional(GRU(96 * 3, return_sequences=True)))
    model.add(Bidirectional(GRU(96 * 3, return_sequences=True)))
    # model.add(build_bigru_layer(args.layers[1]))
    # model.add(build_bigru_layer(args.layers[2]))

    # model.add(Dropout(DROPOUT_RATE))

    model.add(Dense(5, activation='softmax'))
    # model.add(build_dense_layer(args.layers[3]))

    crf_layer = CRF(5, name='crf_layer')
    model.add(crf_layer)
    # model.add(build_crf_layer(args.layers[4]))

    # layer_dict = build_layers()
    # model = Sequential()
    # for layer_args in args.layers:
    #     build_layer_func = layer_dict[layer_args.name]
    #     print('build layer:'+layer_args.name)
    #     if layer_args.name is 'glove':
    #         model_layer = build_layer_func(layer_args, dataset)
    #     else:
    #         model_layer = build_layer_func(layer_args)
    #     model.add(model_layer)

    # model.compile('adam', loss=crf_loss, metrics=[crf_layer.get_accuracy])
    model.compile(optimizer=build_optimizer(args.optimizer), loss=build_loss(args.loss),
                  metrics=build_metrics(args.metrics))

    # tf.keras.utils.plot_model(model, show_shapes=True, dpi=64)
    return model


def save_train_result(history, loss_name, acc_name, model_path):
    history_values = history.history

    loss_values = history_values[loss_name]
    val_loss_values = history_values['val_'+loss_name]

    epochs_point = range(1, len(loss_values) + 1)

    plt.plot(epochs_point, loss_values, 'bo', label='Training Loss')
    plt.plot(epochs_point, val_loss_values, 'b', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.savefig(model_path+'_loss.png')
    plt.show()

    plt.clf()

    acc_values = history_values[acc_name]
    val_acc_values = history_values['val_'+acc_name]

    plt.plot(epochs_point, acc_values, 'bo', label='Training Acc')
    plt.plot(epochs_point, val_acc_values, 'b', label='Validation Acc')
    plt.title('Training and Validation Acc')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.savefig(model_path+'_acc.png')
    plt.show()


def build_layers():
    layers = {
        'embedding': build_embedding_layer,
        'glove':   build_glove_layer,
        'bigru':   build_bigru_layer,
        'dense':   build_dense_layer,
        'dropout': build_dropout_layer,
        'crf':     build_crf_layer
    }

    return layers


def build_model(args, dataset):
    layer_dict = build_layers()
    model = Sequential()
    for layer_args in args.layers:
        build_layer_func = layer_dict[layer_args.name]
        print('build layer:'+layer_args.name)
        if layer_args.name is 'glove':
            model_layer = build_layer_func(layer_args, dataset)
        else:
            model_layer = build_layer_func(layer_args)
        model.add(model_layer)

    model.compile(optimizer=build_optimizer(args.optimizer), loss=build_loss(args.loss), metrics=build_metrics(args.metrics))

    # tensorflow.keras.utils.plot_model(model, to_file='crf_model_1.png', show_shapes=True, dpi=64)

    return model

