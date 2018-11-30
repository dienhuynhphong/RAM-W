import tensorflow as tf
import numpy as np
import sys
from utils import get_data_info, read_data, load_word_embeddings
from model import RAM


FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('embedding_dim', 300, 'dimension of word embedding')
tf.app.flags.DEFINE_integer('batch_size', 32, 'number of example per batch')
tf.app.flags.DEFINE_integer('n_epoch', 50, 'number of epoch')
tf.app.flags.DEFINE_integer('n_hidden', 300, 'number of hidden unit')
tf.app.flags.DEFINE_integer('n_class', 3, 'number of distinct class')
tf.app.flags.DEFINE_integer('n_hop', 3, 'number of hop')
tf.app.flags.DEFINE_integer('pre_processed', 0, 'Whether the data is pre-processed')
tf.app.flags.DEFINE_float('learning_rate', 0.005, 'learning rate')
tf.app.flags.DEFINE_float('l2_reg', 0.001, 'l2 regularization')
tf.app.flags.DEFINE_float('dropout', 0.5, 'dropout')

tf.app.flags.DEFINE_string('embedding_fname', 'data/glove.42B.300d.txt', 'embedding file name')
#tf.app.flags.DEFINE_string('embedding_fname', 'data/glove_300_vectors.txt', 'embedding file name')
tf.app.flags.DEFINE_string('train_fname', 'data/restaurant/train.xml', 'training file name')
tf.app.flags.DEFINE_string('test_fname', 'data/restaurant/test.xml', 'testing file name')
tf.app.flags.DEFINE_string('data_info', 'data/data_info_res_w.txt', 'the file saving data information')
tf.app.flags.DEFINE_string('train_data', 'data/train_data_res_w.txt', 'the file saving training data')
tf.app.flags.DEFINE_string('test_data', 'data/test_data_res_w.txt', 'the file saving testing data')
tf.app.flags.DEFINE_string('sentiment_data', 'data/sentiment.txt', 'list words of sentiment')

#tf.app.flags.DEFINE_string('train_fname', 'data/laptop/train.xml', 'training file name')
#tf.app.flags.DEFINE_string('test_fname', 'data/laptop/test.xml', 'testing file name')
#tf.app.flags.DEFINE_string('data_info', 'data/data_info_laptop.txt', 'the file saving data information')
#tf.app.flags.DEFINE_string('train_data', 'data/train_data_laptop_w.txt', 'the file saving training data')
#tf.app.flags.DEFINE_string('test_data', 'data/test_data_laptop_w.txt', 'the file saving testing data')

#tf.app.flags.DEFINE_string('train_fname', 'data/tweet/train_m.raw', 'training file name')
#tf.app.flags.DEFINE_string('test_fname', 'data/tweet/test_m.raw', 'testing file name')
#tf.app.flags.DEFINE_string('data_info', 'data/data_info_tweet.txt', 'the file saving data information')
#tf.app.flags.DEFINE_string('train_data', 'data/train_data_tweet.txt', 'the file saving training data')
#tf.app.flags.DEFINE_string('test_data', 'data/test_data_tweet.txt', 'the file saving testing data')

#tf.app.flags.DEFINE_string('embedding_fname', 'data/vi_300_vectors.txt', 'embedding file name')
#tf.app.flags.DEFINE_string('train_fname', 'data/nhahang/Viet_ABSA_Train_SB1_sm_rm.xml', 'training file name')
#tf.app.flags.DEFINE_string('test_fname', 'data/nhahang/Viet_ABSA_Test_SB1_sm_rm.xml', 'testing file name')
#tf.app.flags.DEFINE_string('data_info', 'data/data_info_nhahang.txt', 'the file saving data information')
#tf.app.flags.DEFINE_string('train_data', 'data/train_data_nhahang.txt', 'the file saving training data')
#tf.app.flags.DEFINE_string('test_data', 'data/test_data_nhahang.txt', 'the file saving testing data')
#tf.app.flags.DEFINE_string('sentiment_data', 'data/sentiment_vi.txt', 'list words of sentiment')


def main(_):
    print('Loading data info ...')
    #FLAGS.word2id, FLAGS.max_sentence_len, FLAGS.max_aspect_len = get_data_info(FLAGS.train_fname, FLAGS.test_fname, FLAGS.data_info, FLAGS.pre_processed)
    print('Buoc 1: lay thong tin co ban cu du lieu train va test ...')
    word2id, max_sentence_len, max_aspect_len = get_data_info(FLAGS.train_fname, FLAGS.test_fname, FLAGS.data_info, FLAGS.pre_processed)
    #sys.exit()
    #sys.exit()

    #tf.app.flags.DEFINE_string('word2id', word2id, 'word2id')
    tf.app.flags.DEFINE_integer('max_sentence_len', max_sentence_len, 'max sentence len')
    tf.app.flags.DEFINE_integer('max_aspect_len', max_aspect_len, 'max aspect len')

    

    print('Buoc 2: Loading training data and testing data ...')
    print('Buoc 2.1: doac training data ...')
    train_data = read_data(FLAGS.train_fname, word2id, max_sentence_len, max_aspect_len, FLAGS.train_data, FLAGS.pre_processed, FLAGS.sentiment_data)
    #sys.exit()
    print('Buoc 2.2: doac testing data ...')
    test_data = read_data(FLAGS.test_fname, word2id, max_sentence_len,  max_aspect_len, FLAGS.test_data, FLAGS.pre_processed, FLAGS.sentiment_data)

    print('Loading pre-trained word vectors ...')
    word2vec = load_word_embeddings(FLAGS.embedding_fname, FLAGS.embedding_dim, word2id)
    

    with tf.Session() as sess:
        model = RAM(FLAGS,word2id,word2vec, sess)
	
	print('Build model ...')
        model.build_model()
	

	print('Run model ...')
        model.run(train_data, test_data)
	#sys.exit()


if __name__ == '__main__':
    tf.app.run()

