from LAC import LAC
from ddparser import DDParser


class NlpAnalyzer(object):

    def __init__(self):
        data_path = '/Users/xueyu/Workspace/data/'
        model_path = 'nlp/model/'
        lac_model_path = data_path + model_path + 'lac/models_general/lac_model'

        dep_model_path = './corpus/ddparser/model_files/lstm'

        self.__segment = LAC(model_path=lac_model_path, mode='lac')
        self.__dep_parser = DDParser(model_files_path=dep_model_path)

    def analyze(self, text):
        data = {'text': text}
        return self.dep(self.ner(self.pos(self.seg(data))))

    def seg(self, data):
        result = self.__segment.run(data['text'])
        data['word'] = result[0]
        data['tag'] = result[1]
        return data

    def pos(self, data):
        # tags = data['tag']
        return data

    def ner(self, data):
        tags = data['tag']
        data['ent'] = [None] * len(tags)
        for i in range(0, len(tags)):
            tag = tags[i]
            if len(tag) > 2:
                data['tag'][i] = 'n'
                data['ent'][i] = tag
        return data

    def dep(self, data):
        result = self.__dep_parser.parse_seg([data['word']])
        data['head'] = result[0]['head']
        data['deprel'] = result[0]['deprel']
        return data


if __name__ == '__main__':
    analyzer = NlpAnalyzer()

    text = '老板江大桥的主营业务都有哪些'
    data = analyzer.analyze(text)
    print(data)
