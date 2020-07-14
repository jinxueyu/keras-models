# encoding: utf-8

import sys, os
from gensim.models import Word2Vec
import tensorflow as tf
import numpy as np
from tensorflow.contrib.tensorboard.plugins import projector


def visualize(model, output_path):
    meta_file = "w2x_metadata.tsv"
    length_of_word = len(model.wv.index2word)
    placeholder = np.zeros((length_of_word, 300))
    print '#######', len(model.wv.index2word)
    with open(os.path.join(output_path, meta_file), 'w') as file_metadata:
        # file_metadata.write("{0}".format('<Empty Line>') + '\n')  # fix the bug which i can't find it
        count = 0
        for i, word in enumerate(model.wv.index2word):
            placeholder[i] = model[word]
            # temporary solution for https://github.com/tensorflow/tensorflow/issues/9094
            if word == '' or len(word.rstrip()) <= 2:
                print '#################', word
            if word == '' or len(word.rstrip()) == 0:
                print("Emply Line, should replecaed by any thing else, or will cause a bug of tensorboard")
                file_metadata.write("{0}".format('<Empty Line>') + '\n')
            else:
                file_metadata.write(word + '\n')
            count += 1
        print '######', count

    # define the model without training
    sess = tf.InteractiveSession()

    embedding = tf.Variable(placeholder, trainable=False, name='w2x_metadata')
    tf.global_variables_initializer().run()

    saver = tf.train.Saver()
    writer = tf.summary.FileWriter(output_path, sess.graph)

    # adding into projector
    config = projector.ProjectorConfig()
    embed = config.embeddings.add()
    embed.tensor_name = 'w2x_metadata'
    embed.metadata_path = meta_file

    # Specify the width and height of a single thumbnail.
    projector.visualize_embeddings(writer, config)
    saver.save(sess, os.path.join(output_path, 'w2x_metadata.ckpt'))
    print('Run `tensorboard --logdir={0}` to run visualize result on tensorboard'.format(output_path))

if __name__ == "__main__":
    """
    Just run `python w2v_visualizer.py word2vec.model visualize_result`
    """
    try:
        model_path = sys.argv[1]
        output_path = sys.argv[2]
    except:
        print("Please provice model path and output path")
    model = Word2Vec.load(model_path)
    visualize(model, output_path)
