# -*- coding: utf-8 -*-
from gensim import matutils

from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
from numpy.core import multiarray

__author__ = 'xueyu'

dot = multiarray.dot


def train():
    key_set = set([])

    line_sent = read_poems()

    print 'length', len(line_sent)
    model = Word2Vec(line_sent,
                     sg=0,
                     size=300,
                     window=5,
                     min_count=1,
                     workers=4,
                     iter=50)
    model.save('word2vec.model')


def read_poems():
    reader = open('poem_format.csv', 'r')
    line_sent = []
    while True:
        line = reader.readline()
        if not line:
            break
        line = line.strip()
        arr = line.split('\t')
        year = arr[3]
        if len(arr) < 5:
            continue
        type = arr[4]
        if '五' not in type:
            continue

        if year != '唐代':
            continue
        title = arr[0]
        content = arr[1]
        author = arr[2]
        if author != '李白':
            continue

        l = [s.encode('utf8') for s in content.decode('utf8')]
        line_sent.append(l)
    reader.close()
    return line_sent


if __name__ == '__main__':

    train()

    model = Word2Vec.load('word2vec.model')

    # line_sent = read_poems()
    # poem_dict = {}
    # for line in line_sent:
    #     v = None
    #     str_line = None
    #     for word in line:
    #         if word == '，' or word =='。':
    #             poem_dict[str_line] = v
    #             v = None
    #             str_line = None
    #             continue
    #
    #         w = model.wv[word]
    #
    #         if str_line == None:
    #             str_line = word
    #         else:
    #             str_line += word
    #
    #         if v is None:
    #             v = w
    #         else:
    #             v = v + w
    #
    # score_dict = {}
    #
    # test_line_v = None
    # for i in '白日依山尽'.decode('utf8'):
    #     if test_line_v is None:
    #         test_line_v = model.wv[i.encode('utf8')]
    #     else:
    #         test_line_v = test_line_v + model.wv[i.encode('utf8')]
    #
    # for k, v in poem_dict.items():
    #     if test_line_v is None or v is None:
    #         continue
    #     score = dot(matutils.unitvec(test_line_v), matutils.unitvec(v))
    #     score_dict[k] = score
    #
    # result = sorted(score_dict.items(), key=lambda x: x[1], reverse=True)
    # result = result[:10]
    # for r in result:
    #     print r[0], r[1]

    i1 = model.wv['青']
    i2 = model.wv['山']
    i3 = model.wv['水']
    i4 = model.wv['。']

    # l = model.similar_by_word(i1+i2-i4)

    # print type(i4)
    l = model.similar_by_vector(i1)

    for i in l:
        print i[0], i[1]

    t = 0
    for x in i4:
        t += x
    print t
