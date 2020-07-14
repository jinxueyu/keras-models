

import numpy as np
import pandas as pd

from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint
import tensorflow_hub as hub
import tensorflow as tf

from official.bert.tokenization import FullTokenizer


def get_masks(tokens, max_seq_length):
    """Mask for padding"""
    if len(tokens) > max_seq_length:
        raise IndexError("Token length more than max seq length!")
    return [1] * len(tokens) + [0] * (max_seq_length - len(tokens))


def get_segments(tokens, max_seq_length):
    """Segments: 0 for the first sequence, 1 for the second"""
    if len(tokens) > max_seq_length:
        raise IndexError("Token length more than max seq length!")
    segments = []
    current_segment_id = 0
    for token in tokens:
        segments.append(current_segment_id)
        if token == "[SEP]":
            current_segment_id = 1
    return segments + [0] * (max_seq_length - len(tokens))


def get_ids(tokens, tokenizer, max_seq_length):
    """Token ids from Tokenizer vocab"""
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_ids = token_ids + [0] * (max_seq_length - len(token_ids))
    return input_ids


def bert_encode(texts, tokenizer, max_len=512):
    all_tokens = []
    all_masks = []
    all_segments = []

    for text in texts:
        text = tokenizer.tokenize(text)

        text = text[:max_len - 2]
        input_sequence = ["[CLS]"] + text + ["[SEP]"]
        pad_len = max_len - len(input_sequence)

        tokens = tokenizer.convert_tokens_to_ids(input_sequence)
        tokens += [0] * pad_len
        pad_masks = [1] * len(input_sequence) + [0] * pad_len
        segment_ids = [0] * max_len

        all_tokens.append(tokens)
        all_masks.append(pad_masks)
        all_segments.append(segment_ids)

    return np.array(all_tokens), np.array(all_masks), np.array(all_segments)


def BertInput(shape=None, **kwargs):
    input_word_ids = Input(shape=shape, dtype=tf.int32, name="input_word_ids")
    input_mask = Input(shape=shape, dtype=tf.int32, name="input_mask")
    segment_ids = Input(shape=shape, dtype=tf.int32, name="segment_ids")

    return [input_word_ids, input_mask, segment_ids]


def BertTokenizer(model_path):
    m = hub.keras_layer.load_module(model_path)
    vocab_file = m.vocab_file.asset_path.numpy()
    do_lower_case = m.do_lower_case.numpy()

    print(str(m))
    return FullTokenizer(vocab_file, do_lower_case)


class SimpleBertLayer(object):
    def __init__(self, model_path, trainable=True):
        self.__model_path = model_path
        self.__bert_layer = hub.KerasLayer(model_path, trainable=trainable)
        # self.__bert_layer._name = 'bert_layer'

    def call_bert(self, input):
        return self.__bert_layer(input)


class SeqBertLayer(SimpleBertLayer):
    def __init__(self, model_path, trainable=True):
        SimpleBertLayer.__init__(self, model_path, trainable)

    def __call__(self, input):
        return self.call_bert(input)[1]


class PooledBertLayer(SimpleBertLayer):
    def __init__(self, model_path, trainable=True):
        SimpleBertLayer.__init__(self, model_path, trainable)

    def __call__(self, input):
        return self.call_bert(input)[0]


def build_model(model_path, max_len=512):
    input = BertInput(shape=(max_len,))

    x = SeqBertLayer(model_path)(input)
    out = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=input, outputs=out)
    model.compile(Adam(lr=1e-5), loss='binary_crossentropy', metrics=['accuracy'])

    return model


if __name__ == '__main__':
    train = pd.read_csv("/Users/xueyu/Workspace/kaggle/input/nlp-getting-started/train.csv")
    test = pd.read_csv("/Users/xueyu/Workspace/kaggle/input/nlp-getting-started/test.csv")
    submission = pd.read_csv("/Users/xueyu/Workspace/kaggle/input/nlp-getting-started/sample_submission.csv")

    module_url = "https://tfhub.dev/tensorflow/bert_en_uncased_L-24_H-1024_A-16/1"
    module_url = "https://storage.googleapis.com/tfhub-modules/tensorflow/bert_en_uncased_L-24_H-1024_A-16/1.tar.gz"
    module_url = "https://hub.tensorflow.google.cn/google/tensorflow/bert_en_uncased_L-24_H-1024_A-16/1"

    model_path = '/Users/xueyu/Workspace/embidding/tfhub-modules/tensorflow/bert_en_uncased_L-24_H-1024_A-16_2'
    # model_path = '/Users/xueyu/Workspace/embidding/tfhub-modules/google/bert_uncased_L-12_H-768_A-12_1'
    model_path = '/Users/xueyu/Workspace/embidding/tfhub-modules/tensorflow/bert_zh_L-12_H-768_A-12_2'

    model = build_model(model_path, max_len=160)
    model.summary()

    tf.keras.utils.plot_model(model, show_shapes=True, dpi=64)

    checkpoint = ModelCheckpoint('model.h5', monitor='val_loss', save_best_only=True)

    tokenizer = BertTokenizer(model_path)
    train_input = bert_encode(train.text.values, tokenizer, max_len=160)
    test_input = bert_encode(test.text.values, tokenizer, max_len=160)
    train_labels = train.target.values

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



