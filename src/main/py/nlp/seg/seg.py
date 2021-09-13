

def wrap(text, label, index_word):
    # print(label)
    word_list = []
    word = ''
    for i in range(len(text)):
        if text[i] < 1:
            continue

        w = index_word[text[i]]
        # pos = np.argmax(label[i])
        pos = label[i]
        word += w

        if pos == 4 or pos == 2:
            word_list.append(word)
            word = ''
    return word_list


class ISegmentBase(object):
    def seg(self, text):
        pass


class SegmentBase(ISegmentBase):
    def __init__(self, model, dataset):
        self.__model = model  # load_model(model_path, custom_objects=custom_objects)
        self.dataset = dataset
        self.seq_max_len = self.__model.get_layer(index=0).input_length

    def set_model(self, model):
        self.__model = model

    def seg(self, text):
        data = self.dataset.text_to_ids(text, maxlen=self.seq_max_len, padding='post')
        label = self.__model.predict(data)
        return self.wrap(text, label[0])

    def wrap(self, text, label):
        tags = [self.dataset.id2label_dict[id] for id in label]

        sent_out = []
        tags_out = []
        text_size = len(text)
        for ind, tag in enumerate(tags):
            if ind == text_size:
                break

            word = text[ind]
            # for the first char
            if len(sent_out) == 0 or tag.endswith("B") or tag.endswith("S"):
                sent_out.append(word)
                tags_out.append(tag[1:])
                continue

            sent_out[-1] += word
            # 取最后一个tag作为标签
            tags_out[-1] = tag[1:]

        return sent_out