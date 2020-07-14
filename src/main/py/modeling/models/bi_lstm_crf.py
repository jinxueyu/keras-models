
'''
https://tech.retrieva.jp/entry/2019/12/20/121726
'''


import datetime

import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Bidirectional, Dense, Embedding, LSTM
from tensorflow.keras.preprocessing.sequence import pad_sequences
from addons.layers.crf import CRF
from addons.losses.crf import ConditionalRandomFieldLoss
import tensorflow_datasets as tfds


class ConllLoader:
    def __init__(self, data_path, drop_docstart=True, max_length=None, padding='<PAD>'):
        self.max_length = max_length
        self.padding = padding
        self.sentence_lengths = []
        self.mask = []
        self.vocab = []
        self.embeddings = []

        self.sentences, self.labels = [], []
        sentence, labels_for_the_sentence = [], []

        self.char_sentences = []
        char_sentence = []

        with open(data_path, 'r') as f:
            connl_format_data = f.read().splitlines()

        for line in connl_format_data:
            if line.strip() == '':
                sentence_length = len(sentence)
                self.sentence_lengths.append(sentence_length)
                self.sentences.append(sentence)
                self.labels.append(labels_for_the_sentence)
                sentence, labels_for_the_sentence = [], []

                continue
            word, _, _, label = line.split()
            sentence.append(word)
            labels_for_the_sentence.append(label)

        if self.max_length:
            for i, word_token_list in enumerate(self.sentences):
                if len(self.sentences[i]) > self.max_length:
                    self.sentences[i] = self.sentences[i][:self.max_length]
                    self.mask.append([True] * self.max_length)
                else:
                    self.sentences[i] += [self.padding] * (self.max_length - len(self.sentences[i]))
                    self.mask.append(
                        [True] * len(self.sentences[i]) + [False] * (self.max_length - len(self.sentences[i])))

        for i, word_token_list in enumerate(self.sentences):
            self.sentences[i] = ' '.join(self.sentences[i])
            self.labels[i] = ' '.join(self.labels[i])

        if drop_docstart:
            self.sentences = self.sentences[1:]
            self.labels = self.labels[1:]

    def get_list(self):

        return self.sentences, self.labels

    def get_ndarray(self):
        global np
        if np is None:
            import numpy as np

        sentence_array = np.array(self.sentences)
        label_array = np.array(self.labels)

        return sentence_array, label_array

    def get_dataset(self):
        global tf, tfds
        if tf is None:
            import tensorflow as tf
            import tensorflow_datasets as tfds

        sentence_dataset = tf.data.Dataset.from_tensor_slices(self.sentences)
        label_dataset = tf.data.Dataset.from_tensor_slices(self.labels)

        return sentence_dataset, label_dataset

    def output_vocab(self, gensim_model, output_file_path):
        global np
        if np is None:
            import numpy as np
        self.vocab = []
        self.embeddings = []

        with open(output_file_path + '.vocab', 'w') as f:
            for sentence in self.sentences:
                words = sentence.split()
                for word in words:
                    if word not in self.vocab:
                        self.vocab.append(word)
                        f.write(word + '\n')
                        if word in gensim_model.wv:
                            self.embeddings.append(gensim_model.wv[word])
                        else:
                            self.embeddings.append(np.zeros(gensim_model.wv.vector_size))

        np.save(output_file_path + '.npy', self.embeddings)

BATCH_SIZE = 5  # 训练过程中的批处理大小 
EPOCHS = 5  # 训练过程中的历元数 
EMBEDDING_DIM = 300  # 学习单词分布表达式的维数 
MAX_LENGTH = 32  # 必须大于训练数据的最大序列长度

# CoNLL 2003从数据集中以各种格式(字符串列表, 字符串列表)读取各种数据
X_train, y_train = ConllLoader('/Users/xueyu/Workshop/NER/corpus/CoNLL-2003/eng.train').get_list()
X_dev, y_dev = ConllLoader('/Users/xueyu/Workshop/NER/corpus/CoNLL-2003/eng.testa').get_list()
X_test_original, y_test = ConllLoader('/Users/xueyu/Workshop/NER/corpus/CoNLL-2003/eng.testb').get_list()

# 在下面加载学习的分布式表达式
# 这使您可以考虑来自大型语料库的单词语义信息, 而不是训练数据
# 0th分配给填充, (词汇+ 1)分配给未知单词
glove = np.concatenate([np.zeros(EMBEDDING_DIM)[np.newaxis],
                        np.load('../glove-wiki-300-connl.npy'),
                        np.zeros(EMBEDDING_DIM)[np.newaxis]],
                       axis=0)

# 加载学习词汇分布式表达的词汇表
with open('../glove-wiki-300-connl.vocab') as f:
    vocab_list = f.readlines()

# 定义一个编码器, 该编码器将以空格分隔的单词字符串转换为单词索引字符串
word_token_encoder = tfds.features.text.TokenTextEncoder(vocab_list)

# 获取标签的所有标签列表和编号 
label_list = list(set([label for label_in_sentence in y_train for label in label_in_sentence.split()]))
n_labels = len(label_list)

# 定义一个将空格分隔的标签列转换为单词索引列的编码器
label_token_encoder = tfds.features.text.TokenTextEncoder(label_list)


def encode_and_pad_data(sequence_list, encoder):
     # 将str 的列表转换为word index的列表之后,
     # 将0填充到小于# MAX_LENGTH的句子
    enceded_sequences =[encoder.encode(sequence) for sequence in sequence_list]
    encoded_and_padded_sequences = pad_sequences(enceded_sequences, maxlen=MAX_LENGTH, dtype='int32', padding='post',
                                                 value=0)
    return encoded_and_padded_sequences


# 使用或更少的代码编码并填充
X_train = encode_and_pad_data(X_train, word_token_encoder)
X_dev = encode_and_pad_data(X_dev, word_token_encoder)
X_test = encode_and_pad_data(X_test_original, word_token_encoder)
y_train = encode_and_pad_data(y_train, label_token_encoder)
y_dev = encode_and_pad_data(y_dev, label_token_encoder)
y_test = encode_and_pad_data(y_test, label_token_encoder)


# 定义下面的模型# 使用顺序API定义模型
model = tf.keras.Sequential()
# 在“# 嵌入”层中, 将单词索引替换为相应的单词分布表达式
# todo mask_zero = True不起作用
model.add(Embedding(input_dim=word_token_encoder.vocab_size, output_dim=EMBEDDING_DIM,
                    embeddings_initializer=tf.keras.initializers.Constant(glove),
                    input_shape=(MAX_LENGTH, )))

# 添加具有200个输出尺寸的BiLSTM层
# 通过将return_sequences = True传递给# LSTM类初始化程序,
# 您可以检索所有
# words 的预测值如果# return_sequences = False, 则每个语句仅输出一个预测值
model.add(Bidirectional(LSTM(200, return_sequences=True), merge_mode='ave'))
# 添加一个完全连接的层, 其输出维数是标签类型的数量
# 还将激活功能设置为softmax
# 这是因为CRF层要求每个标签具有可预测的置信度
model.add(Dense(n_labels, activation='softmax'))
# 在最后添加CRF层
model.add(CRF(n_labels, name='crf_layer'))

# 初始化模型现在为错误函数指定ConditionalRandomFieldLoss
model.compile(optimizer='adam', loss=ConditionalRandomFieldLoss())
# 定义TensorBoard的日志存储位置
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
# 在这里, 通过传递学习数据来执行学习..
model.fit(X_train, y_train, validation_data=(X_dev, y_dev), epochs=EPOCHS,
          batch_size=BATCH_SIZE,
          callbacks=[tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)])

# 保存模型 
model.save('saved_model/model_sequential.h5')

# 获取预测标签 
y_pred = model.predict([X_test][:5 ])

# 显示预测结果
for i in range(5):
    print('原始句子:', X_test_original[I])
    print('正确的标签:', label_token_encoder.decode(y_test[I]))
    print('预测标记:', label_token_encoder.decode(y_pred[I]))
    print()
    with open('ner_results.txt', 'a') as f:
        f.write('原句:' + X_test_original[i] + '\n ')
        f.write('正确的标签:' + label_token_encoder.decode(y_test[i]) + '\n')
        f.write('预测性标签:' + label_token_encoder.decode(y_pred[i]) + '\n\n')

