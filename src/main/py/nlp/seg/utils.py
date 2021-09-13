import operator
import os

from tqdm import tqdm


def B2Q(uchar):
    """单个字符 半角转全角"""
    inside_code = ord(uchar)
    if inside_code < 0x0020 or inside_code > 0x7e:  # 不是半角字符就返回原来的字符
        return uchar
    if inside_code == 0x0020:  # 除了空格其他的全角半角的公式为: 半角 = 全角 - 0xfee0
        inside_code = 0x3000
    else:
        inside_code += 0xfee0
    return chr(inside_code)


def Q2B(uchar):
    """单个字符 全角转半角"""
    inside_code = ord(uchar)
    if inside_code < 0x0020 or inside_code > 0x7e:  # 转完之后不是半角字符返回原来的字符
        return uchar

    if inside_code == 0x3000:
        inside_code = 0x0020
    else:
        inside_code -= 0xfee0

    return chr(inside_code)


def strQ2B(uchar):
    inside_code = ord(uchar)
    if inside_code == 12288:  # 全角空格直接转换
        inside_code = 32
    elif inside_code >= 65281 and inside_code <= 65374:  # 全角字符（除空格）根据关系转化
        inside_code -= 65248
    return chr(inside_code)


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

    r = a_and_b * 1.0 / a
    p = a_and_b * 1.0 / b
    f1 = 2 * p * r / (p + r)
    print(" correct : %f  %f  a: %f  b: %f" % (correct, a_and_b, b, a))
    print(" correct : %f  %f  P: %f  R: %f F1: %f" % (correct, correct * 1.0 / size, p, r, f1))

    return r, p, f1


def evaluation_seg(segment, gold_path):
    def prophet(text):
        return segment.seg(text)
    return evaluation(prophet, gold_path)
