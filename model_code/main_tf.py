import os
import time
import tensorflow as tf
import numpy as np
import data_loader
from utils import data_summary_from_list2all,get_batches,prepare_batch_data,accuracy
from utils import result_analysis
from utils import result_print
import vocabulary_embeddings_extractor
import model_tf

tf.app.flags.DEFINE_integer('lstm_size',64,'LSTM hidden units')
tf.app.flags.DEFINE_float('dropout', 0.9, 'dropout rate')
tf.app.flags.DEFINE_float('learning_rate', 0.001, 'learning_rate')
tf.app.flags.DEFINE_integer('batch_size', 32, 'number of seqs in one batch')
tf.app.flags.DEFINE_integer('max_len', 100, "Max len of sentences")
tf.app.flags.DEFINE_integer('embedding_size', 300, 'size of embedding')
tf.app.flags.DEFINE_string('current_dir', './', 'path of data')
tf.app.flags.DEFINE_string('model_path', './default','path to save the model')
tf.app.flags.DEFINE_integer('num_layers', 1, 'number of layers')
tf.app.flags.DEFINE_boolean('rich_context',True, 'use rich context or not')
tf.app.flags.DEFINE_integer('nb_epoch',20,'train_epoch')
tf.app.flags.DEFINE_float('l2_strength', 0.001, 'l2_strength')
tf.app.flags.DEFINE_integer('max_grad_norm',5, 'max_grad_norm')

FLAGS = tf.app.flags.FLAGS


FLAGS = tf.app.flags.FLAGS

def __main__():

    print('Loading data...')

    embeddings_cache_file =  "./embeddings_cache_file_word2vec.pkl.bz2"

    # load pre-extracted word-to-index maps and pre-filtered Glove embeddings
    word_to_indices_map, word_index_to_embeddings_map = \
        vocabulary_embeddings_extractor.load_cached_vocabulary_and_embeddings(embeddings_cache_file)
    embeddings_pre_matrix = []
    for key in sorted(word_index_to_embeddings_map.keys()):
        value = word_index_to_embeddings_map[key]
        embeddings_pre_matrix.append(value)

    #read the data from files
    (train_instance_id_list, train_warrant0_list, train_warrant1_list, train_correct_label_w0_or_w1_list,
     train_reason_list, train_claim_list, train_debate_meta_data_list) = \
        data_loader.load_single_file(False, FLAGS.current_dir + 'data/train/train-w-swap-full.txt', word_to_indices_map)

    (dev_instance_id_list, dev_warrant0_list, dev_warrant1_list, dev_correct_label_w0_or_w1_list,
     dev_reason_list, dev_claim_list, dev_debate_meta_data_list) = \
        data_loader.load_single_file(False, FLAGS.current_dir + 'data/dev/dev-full.txt', word_to_indices_map)

    (test_instance_id_list, test_warrant0_list, test_warrant1_list,
     test_reason_list, test_claim_list, test_debate_meta_data_list) = \
        data_loader.load_single_file(True, FLAGS.current_dir + '/data/test/test-only-data.txt', word_to_indices_map)

    train_data=data_summary_from_list2all(False,train_warrant0_list, train_warrant1_list, train_reason_list, train_claim_list, train_debate_meta_data_list,train_correct_label_w0_or_w1_list,train_instance_id_list)
    # print("train_correct_label_w0_or_w1_list",train_correct_label_w0_or_w1_list)
    dev_data = data_summary_from_list2all(False, dev_warrant0_list, dev_warrant1_list, dev_reason_list, dev_claim_list, dev_debate_meta_data_list,dev_correct_label_w0_or_w1_list,dev_instance_id_list)
    test_correct_label_w0_or_w1_list = []
    test_data = data_summary_from_list2all(True, test_warrant0_list, test_warrant1_list, test_reason_list, test_claim_list, test_debate_meta_data_list,test_correct_label_w0_or_w1_list,test_instance_id_list)

    #shape is (2420,100)  2420 is sentences number,100 is D
    train_batches=get_batches(len(train_reason_list),FLAGS.batch_size)
    dev_batches = get_batches(len(dev_reason_list), FLAGS.batch_size)
    test_batches = get_batches(len(test_reason_list), FLAGS.batch_size)

    if not os.path.exists(FLAGS.model_path):
        os.makedirs(FLAGS.model_path)

    with tf.Session() as sess:
        # train_or_others  whether is train or not true:train  false:others
        #tf.set_random_seed(1007)
        model = model_tf.InitModel(FLAGS, embeddings_pre_matrix, True)
        ckpt = tf.train.get_checkpoint_state(FLAGS.model_path)
        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
            model.saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            print("Created model with fresh parameters.")
            sess.run(tf.global_variables_initializer())


        best_dev_acc = 0.0
        for epoch in range(FLAGS.nb_epoch):
            print('<Epoch {}>'.format(epoch))
            train_loss, dev_loss, test_loss = 0.0, 0.0, 0.0
            i, j, k = 0, 0, 0
            start_time = time.time()
            train_logits_list, dev_logits_list, test_logits_list = [], [], []
            dev_id_list_all = []
            for start, end in train_batches:
                i += 1
                batch_train_data = prepare_batch_data(False, train_data, start, end, FLAGS.max_len)
                batch_train_w0, batch_train_w1, batch_train_r, \
                batch_train_c, batch_train_debate,batch_train_label,batch_train_id, batch_train_len_w0, \
                batch_train_len_w1, batch_train_len_r, batch_train_len_c, batch_train_len_d = batch_train_data
                batch_train_outputs = model.batch_fit(sess,
                                                      batch_train_w0,
                                                      batch_train_w1,
                                                      batch_train_r,
                                                      batch_train_c,
                                                      batch_train_debate,
                                                      batch_train_label,
                                                      batch_train_len_w0,
                                                      batch_train_len_w1,
                                                      batch_train_len_r,
                                                      batch_train_len_c,
                                                      batch_train_len_d,
                                                      FLAGS.dropout)
                _, batch_train_loss,batch_train_preds = batch_train_outputs
                train_loss += batch_train_loss
                train_logits_list.extend(batch_train_preds)

            end_time = time.time()
            train_acc = accuracy(train_logits_list, train_correct_label_w0_or_w1_list)
            print('time on epoch {} = {},train_loss = {},train_acc = {})'.format(epoch, end_time - start_time, train_loss/i, train_acc))

            for start, end in dev_batches:
                j += 1
                batch_dev_data = prepare_batch_data(False,dev_data, start, end, FLAGS.max_len)
                batch_dev_w0, batch_dev_w1, batch_dev_r, \
                batch_dev_c, batch_dev_debate,batch_dev_label,dev_id_list, batch_dev_len_w0, \
                batch_dev_len_w1, batch_dev_len_r, batch_dev_len_c, batch_dev_len_d = batch_dev_data
                batch_dev_outputs = model.batch_dev(sess,
                                                      batch_dev_w0,
                                                      batch_dev_w1,
                                                      batch_dev_r,
                                                      batch_dev_c,
                                                      batch_dev_debate,
                                                      batch_dev_label,
                                                      batch_dev_len_w0,
                                                      batch_dev_len_w1,
                                                      batch_dev_len_r,
                                                      batch_dev_len_c,
                                                      batch_dev_len_d,
                                                      1.0)
                batch_dev_loss,batch_dev_preds = batch_dev_outputs
                dev_loss += batch_dev_loss
                dev_logits_list.extend(batch_dev_preds)
            dev_acc = accuracy(dev_logits_list, dev_correct_label_w0_or_w1_list)
            print('dev_loss = {},dev_acc = {})'.format(dev_loss / j, dev_acc))
            for start, end in test_batches:
                k += 1
                batch_test_data = prepare_batch_data(True, test_data, start, end, FLAGS.max_len)
                batch_test_w0, batch_test_w1, batch_test_r, \
                batch_test_c, batch_test_debate,batch_test_label,test_id_list, batch_test_len_w0, \
                batch_test_len_w1, batch_test_len_r, batch_test_len_c, batch_test_len_d = batch_test_data
                batch_test_outputs = model.batch_test(sess,
                                                    batch_test_w0,
                                                    batch_test_w1,
                                                    batch_test_r,
                                                    batch_test_c,
                                                    batch_test_debate,
                                                    batch_test_len_w0,
                                                    batch_test_len_w1,
                                                    batch_test_len_r,
                                                    batch_test_len_c,
                                                    batch_test_len_d,
                                                    1.0)
                batch_test_preds = batch_test_outputs
                test_logits_list.extend(batch_test_preds)
            if dev_acc >= best_dev_acc:
                #result_analysis(dev_correct_label_w0_or_w1_list,dev_logits_list,dev_instance_id_list)
                result_print(test_logits_list,test_instance_id_list)
                checkpoint_path = os.path.join(FLAGS.model_path, "best_dev_model.ckpt")
                model.saver.save(sess, checkpoint_path)
                best_dev_acc = dev_acc
                print('model saved at epoch {}'.format(epoch))

                #for i in range(len(dev_logits_list)):
                #    out_file.write(str(dev_instance_id_list[i])+'\t'+str(dev_logits_list[i])+'\n')



if __name__ == "__main__":
    __main__()
