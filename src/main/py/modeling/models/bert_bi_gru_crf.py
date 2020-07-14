import tensorflow as tf

from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import TimeDistributed, Bidirectional, Dropout, LSTM, GRU
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Model

from addons.layers.crf import CRF
from addons.losses.crf import crf_loss
from modeling.models.bert import bert_encode, seq_bert_layer, BertInput, BertTokenizer, SeqBertLayer

import pandas as pd


VOCAB_SIZE = 2500
EMBEDDING_OUT_DIM = 128
TIME_STAMPS = 100
HIDDEN_UNITS = 200
DROPOUT_RATE = 0.3
NUM_CLASS = 5


def build_embedding_model():
    model = Sequential()
    model.add(Embedding(VOCAB_SIZE, output_dim=EMBEDDING_OUT_DIM, input_length=TIME_STAMPS))
    model.add(Bidirectional(LSTM(HIDDEN_UNITS, return_sequences=True)))
    model.add(Dropout(DROPOUT_RATE))
    model.add(Bidirectional(LSTM(HIDDEN_UNITS, return_sequences=True)))
    model.add(Dropout(DROPOUT_RATE))
    model.add(TimeDistributed(Dense(NUM_CLASS)))

    # crf_layer = CRF(NUM_CLASS)
    # model.add(crf_layer)
    # model.compile('rmsprop', loss=crf_layer.loss_function, metrics=[crf_layer.accuracy])

    # from tensorflow_addons.layers import CRF
    # from tensorflow_addons.layers import CRF
    # # from CRF import CRF
    crf = CRF(NUM_CLASS, name='crf_layer')
    model.add(crf)
    model.compile('adam', loss={'crf_layer': crf.get_loss})

    return model


def build_model(bert_path, max_len=512):
    # bert
    input = BertInput(shape=(max_len,))
    x = SeqBertLayer(bert_path)(input)

    # bi gru
    x = Bidirectional(GRU(HIDDEN_UNITS, return_sequences=True))(x)
    x = Dropout(DROPOUT_RATE)(x)
    x = Bidirectional(GRU(HIDDEN_UNITS, return_sequences=True))(x)
    x = Dropout(DROPOUT_RATE)(x)

    # crf
    print(x.shape)
    x = TimeDistributed(Dense(NUM_CLASS))(x)
    # crf_layer = CRF(NUM_CLASS)(x)

    crf_layer = CRF(NUM_CLASS, name='crf_layer')
    x = crf_layer(x)

    model = Model(input, x)
    model.compile(
        'adam',
        # loss={'crf_layer': crf_loss},
        loss=crf_loss,
        metrics=[crf_layer.get_accuracy]
    )

    # model.compile(
    #     optimizer='rmsprop',
    #     loss=keras_contrib.losses.crf_loss,
    #     metrics=[keras_contrib.metrics.crf_accuracy]
    # )

    return model


if __name__ == '__main__':
    train = pd.read_csv("/Users/xueyu/Workspace/kaggle/input/nlp-getting-started/train.csv")
    test = pd.read_csv("/Users/xueyu/Workspace/kaggle/input/nlp-getting-started/test.csv")
    submission = pd.read_csv("/Users/xueyu/Workspace/kaggle/input/nlp-getting-started/sample_submission.csv")

    bert_path = '/Users/xueyu/Workspace/embidding/tfhub-modules/tensorflow/bert_en_uncased_L-24_H-1024_A-16_2'
    max_len = 160

    model = build_model(bert_path, max_len)
    # model = build_embedding_model()
    model.summary()

    tf.keras.utils.plot_model(model, 'bert_bi_gru_crf.png', show_shapes=True, dpi=64)

    tokenizer = BertTokenizer(bert_path)

    train_input = bert_encode(train.text.values, tokenizer, max_len=160)
    test_input = bert_encode(test.text.values, tokenizer, max_len=160)
    train_labels = train.target.values

    # checkpoint = ModelCheckpoint('model.h5', monitor='val_loss', save_best_only=True)
    # train_history = model.fit(
    #     train_input, train_labels,
    #     validation_split=0.2,
    #     epochs=3,
    #     callbacks=[checkpoint],
    #     batch_size=16
    # )

    # model.load_weights('model.h5')
    # test_pred = model.predict(test_input)

    # submission['target'] = test_pred.round().astype(int)
    # submission.to_csv('submission.csv', index=False)
