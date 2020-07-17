import configparser
import os

from addons.losses.crf import crf_loss
from addons.metrics.crf import crf_acc_mask
from nlp.corpus.reader import DataProcessor
from nlp.segment.model import build_model
from nlp.segment.seg import SegmentBase
from util import ObjectDict

import matplotlib.pyplot as plt


class CRFSegment(SegmentBase):
    def __init__(self, args, dataset, mode='seg'):
        self.model = build_model(args, dataset)
        self.model_path = args.model_path

        if mode == 'seg':
            self.model.load_weights(self.model_path)
        SegmentBase.__init__(self, self.model, dataset)

        self.args = args
        self.dataset = dataset

    def train(self, data_path):
        train_x, train_y = self.dataset.process_input_data(data_path, maxlen=self.seq_max_len)
        history = self.model.fit(
            train_x, train_y,
            batch_size=self.args.batch_size,
            epochs=self.args.epochs,
            validation_split=self.args.validation_split
        )
        self.model.save(self.model_path)

        epochs_point = range(1, self.args.epochs + 1)
        history_values = history.history

        loss_values = history_values['loss']
        val_loss_values = history_values['val_loss']

        plt.plot(epochs_point, loss_values, 'bo', label='Training Loss')
        plt.plot(epochs_point, val_loss_values, 'b', label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')

        plt.legend()
        plt.show()

        plt.clf()

        acc_values = history_values['crf_acc']
        val_acc_values = history_values['val_crf_acc']

        plt.plot(epochs_point, acc_values, 'bo', label='Training Acc')
        plt.plot(epochs_point, val_acc_values, 'b', label='Validation Acc')
        plt.title('Training and Validation Acc')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')

        plt.legend()
        plt.show()


def get_args():
    data_path = '/Users/xueyu/Workspace/data/'
    glove_path = data_path+'nlp/embidding/glove/tencent/Tencent_AILab_ChineseEmbedding_Single.txt'

    mask = True

    args_dict = {
        'model_name': 'seg-glove-bi-gru2-crf-model-mask',
        'mask': True,
        'optimizer': 'adam',
        'loss': crf_loss,
        'metrics': [crf_acc_mask],
        'batch_size': 512,
        'epochs': 20,
        'validation_split': 0.1,
        'layers': [
            {
                'name': 'glove',
                'vec_path': glove_path,
                'input_length': 99,
                'mask': True,
                'trainable': True
            },
            {
                'name': 'bigru',
                'hidden_size': 256
            },
            {
                'name': 'dropout',
                'dropout_rate': 0.2
            },
            {
                'name': 'bigru',
                'hidden_size': 256
            },
            {
                'name': 'dense',
                'input_shape': 5
            },
            {
                'name': 'crf',
                'output_dim': 5
            }
        ]
    }

    for name, value in args_dict.items():
        if type(value) is dict:
            value = ObjectDict(value)
        if type(value) is list:
            val_list = []
            for val in value:
                if type(val) is dict:
                    val = ObjectDict(val)
                val_list.append(val)
            value = val_list

        args_dict[name] = value

    return ObjectDict(args_dict)


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

    args = get_args()

    args.vocab_size = dataset.vocab_size
    args.num_labels = dataset.num_labels

    args.model_path = os.path.join(data_path, model_path, model_name + '.h5')
    args.glove_path = os.path.join(data_path, glove_path)

    # model = build_model(args, dataset)
    # print(model.summary())
    # tf.keras.utils.plot_model(model, to_file='crf_model_1.png', show_shapes=True, dpi=64)

    text_list = ['我是中国人',
                 '我爱北京天安门',
                 '郭小美和王帅身穿和服走在大街上',
                 '李冰冰从马上跳下来',
                 '武汉市长江大桥发表重要讲话',
                 '人们常说生活是一部教科书，而血与火的战争更是不可多得的教科书，她确实是名副其实的‘我的大学’。',
                 '为了有效地解决“高产穷县”的矛盾，吉林省委、省政府深入实际，调查研究，确定了实施“三大一强”的农业发展战略，即经过的努力，'
                 '粮食产量要再上两个台阶，畜牧业要成为农民增收的支柱产业，农副产品加工业要成为全省工业和财政收入的一大支柱，真正成为粮食"'
                 ]

    crf_seg = CRFSegment(args, dataset, mode)
    if mode == 'train':
        data_path = os.path.join(data_path, corpus_path)
        crf_seg.train(data_path)

    for text in text_list:
        print(crf_seg.seg(text))



