
import operator

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
    for i in range(0, size):
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
            print(i)
            # print(' '.join(result))
            # print(' '.join(gold))

        l = list(set(result).intersection(set(gold)))
        a_and_b += len(l)

    print(" correct : %f  %f  P: %f  R: %f" % (correct, correct * 1.0 / size ,  a_and_b * 1.0 / b ,  a_and_b * 1.0 / a))
    # print(" time: " + (System.currentTimeMillis() - start))


def evaluation_crf(gold_path, ):
    dataset = DataProcessor()

    model_path = 'seg-glove-bi-gru-crf-model-mask.h5'
    crf_seg = CRFSegment(model_path, dataset)

    def prophet(text):
        return crf_seg.seg(text)

    evaluation(prophet, gold_path)


def evaluation_lac(gold_path):
    lac = LAC()

    def prophet(text):
        return lac.run(text)[0]

    evaluation(prophet, gold_path)


def evaluation_seg(segment, gold_path):
    def prophet(text):
        return segment.seg(text)

    evaluation(prophet, gold_path)


if __name__ == '__main__':
    org = 'msr'
    gold = "/Users/xueyu/Workspace/training data/icwb2-data/gold/" + org + "_test_gold.utf8";
    evaluation_crf(gold)
