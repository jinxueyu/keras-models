import configparser
import os
import tensorflow as tf
from nlp.corpus.reader import DataProcessor
from nlp.segment.model import build_model, save_train_result
from nlp.segment.seg import SegmentBase
from nlp.segment.utils import evaluation_seg
from util.object_dict import build_object_dict


class CRFSegment(SegmentBase):
    def __init__(self, args, dataset):
        self.model = build_model(args, dataset)
        self.model_path = args.model_path

        if args.mode == 'seg':
            self.model.load_weights(self.model_path)
        #     todo args base segment
        SegmentBase.__init__(self, self.model, dataset)

        self.args = args
        self.dataset = dataset

    def train(self, data_path):
        print(self.model.summary())
        tf.keras.utils.plot_model(model, to_file=self.model_path+'_model.png', show_shapes=True, dpi=64)
        train_x, train_y = self.dataset.process_input_data(data_path, maxlen=self.seq_max_len)
        history = self.model.fit(
            train_x, train_y,
            batch_size=self.args.batch_size,
            epochs=self.args.epochs,
            validation_split=self.args.validation_split
        )
        self.model.save(self.model_path)
        return history


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


def get_args(config):
    model_name = 'seg-glove-bi-gru2-crf-model-mask'
    glove_path = os.path.join(config.data_path, config.glove_path)

    mask = True
    args_dict = {
        'model_name': model_name,
        'mask': True,
        'optimizer': {
            'name': 'adam',
            'learning_rate': 0.001  # 0.0005
        },
        'loss': {'name': 'crf_loss'},
        'metrics': [{'name': 'crf_acc_mask'}],
        'batch_size': 512,
        'epochs': 15,
        'validation_split': 0.1,
        'layers': [
            {
                'name': 'glove',
                'vec_path': glove_path,
                'input_length': 99,
                'mask': True,
                'trainable': True
            },
            # {
            #     'name': 'embedding',
            #     'vocab_size': vocab_size,
            #     '': 96  # 128
            # },
            {
                'name': 'dense',
                'units': 96,
                'kernel_initializer': {
                    'name': 'RandomUniform',
                    'minval': -0.1,
                    'maxval': 0.1
                },
                'kernel_regularizer': {
                    'name': 'l2',
                    'l': 1e-4
                }
            },
            {
                'name': 'bigru',
                'units': 96 * 3
            },
            {
                'name': 'bigru',
                'units': 96 * 3
            },
            {
                'name': 'dense',
                'units': 5,
                'kernel_initializer': {
                    'name': 'RandomUniform',
                    'minval': -0.1,
                    'maxval': 0.1
                },
                'kernel_regularizer': {
                    'name': 'l2',
                    'l': 1e-4
                }
            },
            {
                'name': 'crf',
                'output_dim': 5
            }
        ]
    }

    args = build_object_dict(args_dict)

    args.data_path = config.data_path
    args.model_path = os.path.join(config.data_path, config.model_path, model_name + '.h5')
    args.train_data_path = os.path.join(config.data_path, config.corpus_path)

    return args


def train(args, dataset):
    args.mode = 'seg-train'

    args.vocab_size = dataset.vocab_size
    args.num_labels = dataset.num_labels

    seg = CRFSegment(args, dataset)
    history = seg.train(args.train_data_path)

    save_train_result(history, 'loss', 'crf_acc', args.model_path)


def evaluation(args, dataset):
    args.mode = 'seg'

    args.vocab_size = dataset.vocab_size
    args.num_labels = dataset.num_labels
    gold = os.path.join(args.data_path, "nlp/corpus/icwb2-data/gold/msr_test_gold.utf8")

    seg = CRFSegment(args, dataset)
    evaluation_seg(seg, gold)


if __name__ == '__main__':
    # seg-glove-bi-gru2-td-crf-model-mask  correct : 1597.000000  0.400753  P: 0.849083  R: 0.820235

    config = get_config()
    args = get_args(config)
    dataset = DataProcessor()

    train(args, dataset)

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

    # args.mode = 'seg'
    # crf_seg = CRFSegment(args, dataset)
    # for text in text_list:
    #     print(crf_seg.seg(text))

    # evaluation(args, dataset)



