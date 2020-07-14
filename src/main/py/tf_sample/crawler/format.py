# -*- coding: utf-8 -*-
from tf_sample.utils import json

__author__ = 'xueyu'


if __name__ == '__main__':
    reader = open('poem_all.txt', 'r')
    writer = open('poem_format.csv', 'w')

    key_set = set([])
    count = 0
    while True:
        line = reader.readline()
        if not line:
            break
        line = line.strip()
        obj = json.read(line)

        title = obj['title']
        content = obj['content']

        tags = obj['tags']
        author = ''
        year = ''
        type = ''
        cat = ''
        if '作者' in tags:
            author = tags['作者']
        if '年代' in tags:
            year = tags['年代']
        if '体裁' in tags:
            type = tags['体裁']
        if '类别' in tags:
            cat = tags['类别']

        if count % 1000 == 0:
            print count
        count += 1

        writer.write(title+'\t'+content+'\t'+author+'\t'+year+'\t'+type+'\t'+cat+'\n')
    reader.close()
    writer.close()
