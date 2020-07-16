import configparser
import os

import tensorflow as tf

from nlp.corpus.reader import DataProcessor
from nlp.segment.model import build_glove_bi_gru_crf_model, build_glove_bi_gru2_crf_model, MAX_LENGTH
from nlp.segment.seg import SegmentBase
from util import ObjectDict

build_model = build_glove_bi_gru2_crf_model


class CRFSegment(SegmentBase):
    def __init__(self, args, dataset, mode='seg'):
        self.model = build_model(args, dataset.word_to_id)
        self.model_path = args.model_path

        if mode == 'seg':
            self.model.load_weights(self.model_path)
        SegmentBase.__init__(self, self.model, dataset)

    def train(self, train_x, train_y):
        self.model.fit(
            train_x, train_y,
            batch_size=512,
            epochs=10,
            validation_split=0.1
        )
        self.model.save(self.model_path)


if __name__ == '__main__':
    # seg-glove-bi-gru2-td-crf-model-mask  correct : 1597.000000  0.400753  P: 0.849083  R: 0.820235

    config = configparser.ConfigParser()
    config.read(os.path.join('', "model-args-config.ini"))

    data_path = '/Users/xueyu/Workspace/data'
    model_path = 'nlp/model/keras-models'
    corpus_path = 'nlp/corpus/icwb2-data/training/msr_training.utf8'
    glove_path = 'nlp/embidding/glove/tencent/Tencent_AILab_ChineseEmbedding_Single.txt'

    mode = 'seg'
    mode = 'train'
    model_name = 'seg-glove-bi-gru2-crf-model-mask'

    dataset = DataProcessor()

    args = ObjectDict()
    args.vocab_size = dataset.vocab_size
    args.num_labels = dataset.num_labels

    args.model_path = os.path.join(data_path, model_path, model_name + '.h5')
    args.glove_path = os.path.join(data_path, glove_path)

    crf_seg = CRFSegment(args, dataset, mode)
    if mode == 'train':
        data_path = os.path.join(data_path, corpus_path)
        train_x, train_y = dataset.process_input_data(data_path, maxlen=MAX_LENGTH)
        crf_seg.train(train_x, train_y)

    text_list = ['我是中国人',
                 '我爱北京天安门',
                 '郭小美和王帅身穿和服走在大街上',
                 '李冰冰从马上跳下来',
                 '武汉市长江大桥发表重要讲话',
                 '人们常说生活是一部教科书，而血与火的战争更是不可多得的教科书，她确实是名副其实的‘我的大学’。',
                 '为了有效地解决“高产穷县”的矛盾，吉林省委、省政府深入实际，调查研究，确定了实施“三大一强”的农业发展战略，即经过的努力，'
                 '粮食产量要再上两个台阶，畜牧业要成为农民增收的支柱产业，农副产品加工业要成为全省工业和财政收入的一大支柱，真正成为粮食"'
                 ]

    for text in text_list:
        print(crf_seg.seg(text))
