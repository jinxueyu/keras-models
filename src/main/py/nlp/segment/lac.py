from nlp.lac import LAC
from nlp.lac.custom import Customization


def test():
    lac = LAC(mode='seg')
    text = '步入城市电器商行，你会被西子湖畔捧出的一个个具有时代感和个性以及舒适性的东宝空调所吸引，如同置身于西湖柳堤上一样'

    reader = open('/Users/xueyu/Workspace/training data/icwb2-data/gold/msr_test_gold.utf8', 'r')
    writer = open('/Users/xueyu/Workspace/training data/msr_test_gold_lac_t.utf8', 'w')
    while True:
        line = reader.readline()
        if not line:
            break

        text = line.rstrip().replace('  ', '')
        result = lac.run(text)

        text = '  '.join(result)
        writer.write(text + '\n')

    reader.close()
    writer.close()


def train():
    # 训练和测试数据集，格式一致
    # train_file = "./data/seg_train.tsv"
    # test_file = "./data/seg_test.tsv"

    org = 'msr'
    data_path = '/Users/xueyu/Workspace/data/'
    test_file = data_path + "nlp/corpus/icwb2-data/gold/" + org + "_test_gold.utf8"
    train_file = data_path + "nlp/corpus/icwb2-data/training/msr_training.utf8"

    model_path = 'nlp/model/'

    print(train_file)

    lac = LAC(model_path=data_path+model_path+'lac/seg_model', mode='seg', pretraining=False)
    lac.train(model_save_dir=data_path+model_path+'lac/msr_seg_model', train_data=train_file, test_data=test_file)


def ac_postpress(ac_res):
    ac_res.sort()
    i = 1
    while i < len(ac_res):
        if ac_res[i - 1][0] < ac_res[i][0] and ac_res[i][0] <= ac_res[i - 1][1]:
            ac_res.pop(i)
            continue
        i += 1
    return ac_res


if __name__ == '__main__':

    # train()
    # [test]
    # P: 0.86056, R: 0.88362, F1: 0.87194
    # [test]
    # P: 0.96903, R: 0.96838, F1: 0.96870
    # correct : 1378.000000  0.345797  P: 0.921734  R: 0.918146

    data_path = '/Users/xueyu/Workspace/data/'
    model_path = 'nlp/model/'
    model_path = data_path+model_path+'lac/msr_seg_model'
    print(model_path)

    lac = LAC(model_path=model_path, mode='seg')
    text = [
        '我是中国人',
        '我爱北京天安门',
        '郭小美和王帅身穿和服走在大街上',
        '李冰冰从马上跳下来',
        '武汉市长江大桥发表重要讲话',
        '人们常说生活是一部教科书，而血与火的战争更是不可多得的教科书，她确实是名副其实的‘我的大学’。',
        '为了有效地解决“高产穷县”的矛盾，吉林省委、省政府深入实际，调查研究，确定了实施“三大一强”的农业发展战略，即经过的努力，粮食产量要再上两个台阶，畜牧业要成为农民增收的支柱产业，农副产品加工业要成为全省工业和财政收入的一大支柱，真正成为粮食"'
    ]
    r = lac.run(text)
    for x in r:
        print(x)

    # test()

    custom = Customization()
    custom.load_customization('/Users/xueyu/Workspace/training data/icwb2-data/gold/pku_training_words.utf8')
    ah = custom.ac

    query = u"共同创造美好的新世纪——二○○一年新年贺词"

    for begin, end in ah.search_all(query):
        print('all:', query[begin:end + 1])

    for begin, end in ah.search(query):
        print(str(begin)+"  , "+str(end))
        print('search:', query[begin:end + 1])

    s = ac_postpress(ah.search(query))
    for begin, end in s:
        print(str(begin) + "  , " + str(end))
        print('###:', query[begin:end + 1])

    tags = ['O'] * len(query)
    custom.parse_customization(query, tags)
    print('after parse: ', tags)
