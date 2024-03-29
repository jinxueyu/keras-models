import configparser
import os
from nlp.corpus.reader import DataProcessor
from nlp.seg.model import build_model, save_train_result, build_lstm_model, get_args_attr
from nlp.seg.seg import SegmentBase
from nlp.seg.utils import evaluation_seg
from util.object_dict import build_object_dict


class CRFSegment(SegmentBase):
    def __init__(self, args, dataset):
        self.model = build_model(args, dataset)
        self.model_path = args.model_path

        if args.mode == 'seg':
            self.model.load_weights(self.model_path)
        #     todo args base seg
        SegmentBase.__init__(self, self.model, dataset)

        self.args = args
        self.dataset = dataset

    def train(self, data_path):
        print(self.model.summary())

        train_x, train_y = self.dataset.process_input_data(data_path, maxlen=self.seq_max_len)
        history = self.model.fit(
            train_x, train_y,
            batch_size=self.args.batch_size,
            epochs=self.args.epochs,
            validation_split=self.args.validation_split
        )
        # self.model.save(self.model_path)
        save_train_result(self.model, args, history, 'loss', 'crf_acc')

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


def get_args(config, vocab_size=0):
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
        'epochs': 30,
        'validation_split': 0.1,
        'layers': [
            # {
            #     'name': 'glove',
            #     'vec_path': glove_path,
            #     'input_length': 100,
            #     'mask': True,
            #     'trainable': True
            # },
            {
                'name': 'embedding',
                'input_dim': vocab_size,
                'output_dim': 96,
                'input_length': 100,  # < 128
                'trainable': True,
                'mask': True,
                'embeddings_initializer': {
                    'name': 'RandomUniform',
                    'minval': -0.1,
                    'maxval': 0.1
                }
            },
            # {
            #     'name': 'dense',
            #     'units': 96,
            #     'kernel_initializer': {
            #         'name': 'RandomUniform',
            #         'minval': -0.1,
            #         'maxval': 0.1
            #     },
            #     'kernel_regularizer': {
            #         'name': 'l2',
            #         'l': 1e-4
            #     }
            # },
            # {
            #     'name': 'bigru',
            #     'units': 96 * 3
            # },
            {
                'name': 'bigru',
                'units': 96 * 3,
                'return_sequences': True,
                'merge_mode': 'sum',
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
            # {
            #     'name': 'dropout',
            #     'dropout_rate': 0.2
            # },
            {
                'name': 'bigru',
                'units': 96 * 3,
                'return_sequences': True,
                'merge_mode': 'sum',
                'kernel_initializer': {
                    'name': 'RandomUniform',
                    'minval': -0.1,
                    'maxval': 0.1
                },
                'kernel_regularizer': {
                    'name': 'l2',
                    'l': 1e-4
                },
                'dropout': 0.,
                'recurrent_dropout': 0.
            },
            {
                'name': 'dropout',
                'dropout_rate': 0.2
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
                },
                'activation': 'softmax'
            },
            {
                'name': 'crf',
                'output_dim': 5
                #  activation and something
            }
        ]
    }

    args = build_object_dict(args_dict)

    args.data_path = config.data_path
    args.model_path = os.path.join(config.data_path, config.model_path, model_name)
    args.train_data_path = os.path.join(config.data_path, config.corpus_path)

    return args


def train(args, dataset):
    args.mode = 'seg-train'

    args.vocab_size = dataset.vocab_size
    args.num_labels = dataset.num_labels

    seg = CRFSegment(args, dataset)
    history = seg.train(args.train_data_path)


def evaluation(args, dataset):
    args.mode = 'seg'

    args.vocab_size = dataset.vocab_size
    args.num_labels = dataset.num_labels
    gold = os.path.join(args.data_path, "nlp/corpus/icwb2-data/gold/msr_test_gold.utf8")

    seg = CRFSegment(args, dataset)
    evaluation_seg(seg, gold)


def train_lstm():
    data_path = os.path.join(config.data_path, config.corpus_path)
    dataset = DataProcessor()
    input_dim = dataset.vocab_size
    input_length = 128
    num_labels = 5
    lstm_units = 128

    model = build_lstm_model(input_dim, input_length, num_labels, lstm_units)

    train_x, train_y = dataset.process_input_data(data_path, maxlen=input_length)
    history = model.fit(
        train_x, train_y,
        batch_size=512,
        epochs=20,
        validation_split=0.1
    )


if __name__ == '__main__':
    # seg-glove-bi-gru2-td-crf-model-mask  correct : 1597.000000  0.400753  P: 0.849083  R: 0.820235

    dataset = DataProcessor()
    config = get_config()
    args = get_args(config, dataset.vocab_size)

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

