
import operator
from tqdm import tqdm

try:
    from LAC import LAC
except ImportError:
    pass

from nlp.corpus.reader import DataProcessor
from nlp.segment.crf import CRFSegment


def evaluation(prophet, gold_path):
    data_list = []
    gold_list = []

    reader = open(gold_path, 'r')
    while True:
        line = reader.readline()
        if not line:
            break
        line = line.rstrip()
        data_list.append(line.replace('  ', ''))
        gold_list.append(line)

    reader.close()

    correct = 0
    size = len(data_list)
    a = 0
    b = 0
    a_and_b = 0

    # for i in tqdm(range(10), total=10, desc="WSX", ncols=100, postfix=dict, mininterval=0.3):
    for i in tqdm(range(0, size)):
        result = prophet(data_list[i])
        if result is None:
            continue
        b += len(result)

        gold_str = gold_list[i]
        gold = gold_str.split("  ")
        a += len(gold)

        corr = operator.eq(gold, result)
        if corr:
            correct += 1
        else:
            pass
            # print(i)
            # print(' '.join(result))
            # print(' '.join(gold))

        gold = set(gold)
        for w in result:
            if w in gold:
                a_and_b += 1

        # l = list(set(result).intersection())
        # a_and_b += len(l)

    print(" correct : %f  %f  a: %f  b: %f" % (correct, a_and_b, b, a))
    print(" correct : %f  %f  P: %f  R: %f" % (correct, correct * 1.0 / size,  a_and_b * 1.0 / b,  a_and_b * 1.0 / a))
    # print(" time: " + (System.currentTimeMillis() - start))


def evaluation_crf(gold_path, ):
    dataset = DataProcessor()

    model_path = 'seg-glove-bi-gru-crf-model-mask.h5'
    crf_seg = CRFSegment(model_path, dataset)

    def prophet(text):
        return crf_seg.seg(text)

    evaluation(prophet, gold_path)


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

    evaluation(prophet, gold_path)


def evaluation_seg(segment, gold_path):
    def prophet(text):
        return segment.seg(text)

    evaluation(prophet, gold_path)


if __name__ == '__main__':
    org = 'msr'
    data_path = '/Users/xueyu/Workspace/data'
    gold = data_path + "/nlp/corpus/icwb2-data/gold/" + org + "_test_gold.utf8"
    evaluation_lac(gold)
