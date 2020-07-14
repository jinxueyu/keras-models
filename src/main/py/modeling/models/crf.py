from tensorflow.keras import Sequential
from tensorflow.keras.models import Model, Input
from tensorflow.keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

from addons.layers.crf import CRF

max_len = 100
n_words = 0
n_tags = 0


class SentenceGetter(object):

    def __init__(self, data):
        self.n_sent = 1
        self.data = data
        self.empty = False
        agg_func = lambda s: [(w, p, t) for w, p, t in zip(s["Word"].values.tolist(),
                                                           s["POS"].values.tolist(),
                                                           s["Tag"].values.tolist())]
        self.grouped = self.data.groupby("Sentence #").apply(agg_func)
        self.sentences = [s for s in self.grouped]

    def get_next(self):
        try:
            s = self.grouped["Sentence: {}".format(self.n_sent)]
            self.n_sent += 1
            return s
        except:
            return None

EMBED_DIM = 200
BiRNN_UNITS = 200


def build_model():
    model = Sequential()
    model.add(Embedding(n_words + 1, EMBED_DIM))  # Random embedding
    model.add(Bidirectional(LSTM(BiRNN_UNITS // 2, return_sequences=True)))

    crf = CRF(n_tags, sparse_target=True)
    model.add(crf)

    model.summary()
    model.compile('adam', loss=crf.loss_function, metrics=[crf.accuracy])

    return model


def prepare():
    import pandas as pd
    import numpy as np

    global n_words, n_tags

    data = pd.read_csv("../../ner_dataset.csv", encoding="latin1")

    data = data.fillna(method="ffill")

    words = list(set(data["Word"].values))
    words.append("ENDPAD")
    n_words = len(words)

    tags = list(set(data["Tag"].values))
    n_tags = len(tags)

    getter = SentenceGetter(data)
    sentences = getter.sentences

    max_len = 75
    word2idx = {w: i + 1 for i, w in enumerate(words)}
    tag2idx = {t: i for i, t in enumerate(tags)}

    X = [[word2idx[w[0]] for w in s] for s in sentences]
    X = pad_sequences(maxlen=max_len, sequences=X, padding="post", value=0)

    y = [[tag2idx[w[2]] for w in s] for s in sentences]
    y = pad_sequences(maxlen=max_len, sequences=y, padding="post", value=tag2idx["O"])
    y = [to_categorical(i, num_classes=n_tags) for i in y]

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.1)
    return X_tr, np.array(y_tr)


if __name__ == '__main__':

    train_x, train_y = prepare()
    print(n_words)
    print(n_tags)
    print(train_x[0])
    print(train_y[0])

    model = build_model()

    history = model.fit(
        train_x, train_y,
        batch_size=32,
        epochs=5,
        validation_split=0.1,
        verbose=1
    )

    model.predict()
