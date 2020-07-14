
import argparse
import logging
import __future__
import io
import os

from tensorflow.keras import preprocessing
import numpy as np

from util import ObjectDict


def load_kv_dict(dict_path,
                 reverse=False, delimiter="\t", key_func=None, value_func=None):
    """
    Load key-value dict from file
    """
    result_dict = {}
    for line in io.open(dict_path, "r", encoding='utf8'):
        terms = line.strip("\n").split(delimiter)
        if len(terms) != 2:
            continue
        if reverse:
            value, key = terms
        else:
            key, value = terms
        # if key in result_dict:
        #     raise KeyError("key duplicated with [%s]" % (key))
        if key_func:
            key = key_func(key)
        if value_func:
            value = value_func(value)
        result_dict[key] = value
    return result_dict


class Dataset(object):
    """data reader"""

    def __init__(self, args, dev_count=10):
        # read dict
        self.word2id_dict = load_kv_dict(args.word_dict_path, reverse=True, value_func=int)
        self.id2word_dict = load_kv_dict(args.word_dict_path, key_func=int)
        self.label2id_dict = load_kv_dict(args.label_dict_path, reverse=True, value_func=int)
        self.id2label_dict = load_kv_dict(args.label_dict_path, key_func=int)
        self.word_replace_dict = load_kv_dict(args.word_rep_dict_path)
        self.oov_id = self.word2id_dict['OOV']
        self.tag_type = args.tag_type

        self.args = args
        self.dev_count = dev_count

    @property
    def vocab_size(self):
        """vocabuary size"""
        return max(self.word2id_dict.values()) + 1

    @property
    def num_labels(self):
        """num_labels"""
        return max(self.label2id_dict.values()) + 1

    def get_num_examples(self, filename):
        """num of line of file"""
        return sum(1 for line in open(filename, "rb"))

    def parse_seg(self, line):
        """convert segment data to lac data format"""
        tags = []
        words = line.strip().split()

        for word in words:
            if len(word) == 1:
                tags.append('-S')
            else:
                tags += ['-B'] + ['-I'] * (len(word) - 2) + ['-E']

        return "".join(words), tags

    def parse_tag(self, line):
        """convert tagging data to lac data format"""
        tags = []
        words = []

        items = line.strip().split()
        for item in items:
            word = item[:item.rfind('/')]
            tag = item[item.rfind('/') + 1:]
            if '/' not in item or len(word) == 0 or len(tag) == 0:
                logging.warning("Data type error: %s" % line.strip())
                return [], []
            tags += [tag + '-B'] + [tag + '-I'] * (len(word) - 1)
            words.append(word)

        return "".join(words), tags

    def word_to_ids(self, words):
        """convert word to word index"""
        word_ids = []
        for word in words:
            word_id = self.word_to_id(word)
            word_ids.append(word_id)
        return word_ids

    def word_to_id(self, word):
        """convert word to word index"""
        word = self.word_replace_dict.get(word, word)
        word_id = self.word2id_dict.get(word, self.oov_id)
        return word_id

    def label_to_ids(self, labels):
        """convert label to label index"""
        label_ids = []
        for label in labels:
            if label not in self.label2id_dict:
                label = "O"
            label_id = self.label2id_dict[label]
            label_ids.append(label_id)
        return label_ids

    def file_reader(self, filename, mode="train"):
        """
        yield (word_idx, target_idx) one by one from file,
            or yield (word_idx, ) in `infer` mode
        """
        def wrapper():
            """the wrapper of data generator"""
            fread = io.open(filename, "r", encoding="utf-8")
            if mode == "infer":
                for line in fread:
                    words = line.strip()
                    word_ids = self.word_to_ids(words)
                    yield (word_ids,)
            else:
                cnt = 0
                for line in fread:
                    if (len(line.strip()) == 0):
                        continue
                    if self.tag_type == 'seg':
                        words, labels = self.parse_seg(line)
                    elif self.tag_type == 'tag':
                        words, labels = self.parse_tag(line)
                    else:
                        words, labels = line.strip("\n").split("\t")
                        words = words.split("\002")
                        labels = labels.split("\002")

                    word_ids = self.word_to_ids(words)
                    label_ids = self.label_to_ids(labels)
                    assert len(word_ids) == len(label_ids)
                    yield word_ids, label_ids
                    cnt += 1

                if mode == 'train':
                    pad_num = self.dev_count - (cnt % self.args.batch_size) % self.dev_count
                    for i in range(pad_num):
                        if self.tag_type == 'seg':
                            yield [self.oov_id], [self.label2id_dict['-S']]
                        else:
                            yield [self.oov_id], [self.label2id_dict['O']]
            fread.close()

        return wrapper


def _get_abs_path(path): return os.path.normpath(
    os.path.join(os.getcwd(), os.path.dirname(__file__), path))


class DataProcessor(Dataset):
    def __init__(self, data_type='seg'):
        conf_path = _get_abs_path(data_type + '_model/conf/')
        args = {
            'tag_type': data_type,
            'word_dict_path': conf_path + "/word.dic",
            'label_dict_path': conf_path + "/tag.dic",
            'word_rep_dict_path': conf_path + "/q2b.dic",
            'batch_size': 120
        }
        Dataset.__init__(self, ObjectDict(args), dev_count=10)

    def process_input_data(self, data_path, maxlen=256, padding='post'):
        data_generator = self.file_reader(data_path)

        train_x = []
        train_y = []
        for word_ids, label_ids in data_generator():
            train_x.append(word_ids)
            train_y.append(label_ids)

        train_x = self.pad_sequences(np.asarray(train_x), maxlen=maxlen, padding=padding)
        train_y = self.pad_sequences(np.asarray(train_y), maxlen=maxlen, padding=padding, value=0)

        return train_x, train_y

    def pad_sequences(self, seq, maxlen=256, padding='post', value=0.):
        return preprocessing.sequence.pad_sequences(np.asarray(seq), maxlen=maxlen, padding=padding, value=value)

    def text_to_ids(self, text,  maxlen=256, padding='post', value=0.):
        seq = [self.word_to_id(w) for w in text]
        return self.pad_sequences(np.asarray([seq]), maxlen=maxlen, padding=padding, value=value)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--tag_type", type=str,
                        default="seg", help="tag_type")
    parser.add_argument("--word_dict_path", type=str,
                        default="seg_model/conf/word.dic", help="word dict")
    parser.add_argument("--label_dict_path", type=str,
                        default="seg_model/conf/tag.dic", help="label dict")
    parser.add_argument("--word_rep_dict_path", type=str,
                        default="seg_model/conf/q2b.dic", help="word replace dict")
    parser.add_argument("--batch_size", type=int,
                        default="120", help="")

    args = parser.parse_args()

    print(type(args))
    dataset = Dataset(args)
    data_generator = dataset.file_reader("data/msr_training.utf8")
    for word_idx, target_idx in data_generator():
        print(word_idx, target_idx)
        print(len(word_idx), len(target_idx))
        break

    dp = DataProcessor()
    x, y = dp.process_input_data("data/msr_training.utf8")
    print(x[0], y[0])


