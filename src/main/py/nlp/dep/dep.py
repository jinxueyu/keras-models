import pickle

from ddparser import DDParser
from spacy.tokens.doc import Doc

from nlp.lac import LAC


class DenpdencyParser(object):
    def __init__(self, model_files_path):
        self.__ddp = DDParser(model_files_path=model_files_path)
        # self.__ddp = DDParser(encoding_model='transformer', model_files_path=model_files_path)

        print(self.__ddp.args)

    def parse(self, inputs):
        return self.__ddp.parse_seg(inputs)

# python -m spacy download en_core_web_sm
import spacy
from spacy import displacy

if __name__ == '__main__':

    # fields = None
    # with open('../corpus/ddparser/model_files/baidu/lstm/fields', "rb") as f:
    #     fields = pickle.load(f)
    #
    # if isinstance(fields.FORM, tuple):
    #     WORD, FEAT = fields.FORM
    # else:
    #     WORD, FEAT = fields.FORM, fields.CPOS
    # ARC, REL = fields.HEAD, fields.DEPREL
    #
    # print(WORD)
    # print(FEAT)

    parser = DenpdencyParser(model_files_path='../corpus/ddparser/model_files/lstm')
    docs = parser.parse([['平安银行', '董事长', '是', '谁'], ['他', '送', '了', '一本', '书']])
    doc = docs[0]
    print(doc)

    words = doc['word']
    head = doc['head']
    dep = doc['deprel']

    for i in range(0, len(words)):
        print(words[i], dep[i], 'ROOT' if head[i] == 0 else words[head[i]-1])

    nlp = spacy.load("en_core_web_sm")
    doc = nlp("Google is a tech corp.")
    print(type(doc))
    doc = Doc()
    for token in doc:
        print(token.text, token.dep_, token.head)
    displacy.serve(doc, style="dep")
