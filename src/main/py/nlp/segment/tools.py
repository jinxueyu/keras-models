from nlp.segment.utils import evaluation

try:
    from LAC import LAC
except ImportError:
    pass

from nlp.corpus.reader import DataProcessor
from nlp.segment.crf import CRFSegment


def evaluation_crf(gold_path, ):
    dataset = DataProcessor()

    model_path = 'seg-glove-bi-gru-crf-model-mask.h5'
    crf_seg = CRFSegment(model_path, dataset)

    def prophet(text):
        return crf_seg.seg(text)

    return evaluation(prophet, gold_path)


def evaluation_lac(gold_path):
    # correct : 883.000000  0.221581  P: 0.751281  R: 0.722980
    # correct : 2754.000000  0.691092  P: 0.813256  R: 0.812282

    # correct :893   0.22409033877038895  P: 0.9046166697425039 R:0.8714198830409356
    # correct :2754   0.6910915934755333  P: 0.9742943595604561 R:0.973127485380117

    data_path = '/Users/xueyu/Workspace/data/'
    model_path = 'nlp/model/lac/msr_seg_model/'
    model_path = data_path + model_path
    print(model_path)
    lac = LAC(model_path=model_path, mode='seg')

    # lac = LAC(mode='seg')

    def prophet(text):
        return lac.run(text)

    return evaluation(prophet, gold_path)


if __name__ == '__main__':
    org = 'msr'
    data_path = '/Users/xueyu/Workspace/data'
    gold = data_path + "/nlp/corpus/icwb2-data/gold/" + org + "_test_gold.utf8"
    evaluation_lac(gold)
