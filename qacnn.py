# -*- coding: utf-8 -*-
# @Time    : 2019/3/19 20:19
# @Author  : Alan
# @Email   : xiezhengwen2013@163.com
# @File    : train.py
# @Software: PyCharm


import time
import logging
import numpy as np
import tensorflow as tf
import os
import tqdm
import sys
from copy import deepcopy
stdout = sys.stdout

from data_helper import *
from model import SiameseQACNN
from model_utils import *

# 创建一个logger
logger = logging.getLogger('mylogger')
logger.setLevel(logging.DEBUG)

# 创建一个handler，用于写入日志文件
timestamp = str(int(time.time()))
fh = logging.FileHandler('./log/log_' + timestamp +'.txt')
fh.setLevel(logging.DEBUG)

# 定义handler的输出格式
formatter = logging.Formatter('[%(asctime)s][%(levelname)s] ## %(message)s')
fh.setFormatter(formatter)
# ch.setFormatter(formatter)

# 给logger添加handler
logger.addHandler(fh)
# logger.addHandler(ch)


class NNConfig(object):
    def __init__(self, embeddings=None):
        # 输入问题(句子)长度
        self.ques_length = 25
        # 输入答案长度
        self.ans_length = 90
        # 循环数
        self.num_epochs = 100
        # batch大小
        self.batch_size = 128
        # 不同类型的filter，对应不同的尺寸
        self.window_sizes = [1, 2, 3, 5, 7, 9]
        # 隐层大小
        self.hidden_size = 128
        self.output_size = 128
        self.keep_prob = 0.5
        # 每种filter的数量
        self.n_filters = 128
        # margin大小
        self.margin = 0.5
        # 词向量大小
        self.embeddings = np.array(embeddings).astype(np.float32)
        # 学习率
        self.learning_rate = 0.001
        # contrasive loss 中的 positive loss部分的权重
        self.pos_weight = 0.25
        # 优化器
        self.optimizer = 'adam'
        self.clip_value = 5
        self.l2_lambda = 0.0001
        # 评测
        self.eval_batch = 100

        # self.cf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        # self.cf.gpu_options.per_process_gpu_memory_fraction = 0.2


def evaluate(sess, model, corpus, config):
    iterator = Iterator(corpus)

    count = 0
    total_qids = []
    total_aids = []
    total_pred = []
    total_labels = []
    total_loss = 0.
    Acc = []
    for batch_x in iterator.next(config.batch_size, shuffle=False):
        batch_qids, batch_q, batch_aids, batch_a, batch_qmask, batch_amask, labels = zip(*batch_x)
        batch_q = np.asarray(batch_q)
        batch_a = np.asarray(batch_a)
        q_ap_cosine, loss, acc = sess.run([model.q_a_cosine, model.total_loss, model.accu],
                                     feed_dict={model._ques: batch_q,
                                                model._ans: batch_a,
                                                model._ans_neg: batch_a,
                                                model.dropout_keep_prob: 1.0})
        total_loss += loss
        Acc.append(acc)
        count += 1
        total_qids.append(batch_qids)
        total_aids.append(batch_aids)
        total_pred.append(q_ap_cosine)
        total_labels.append(labels)

        # print(batch_qids[0], [id2word[_] for _ in batch_q[0]],
        #     batch_aids[0], [id2word[_] for _ in batch_ap[0]])
    total_qids = np.concatenate(total_qids, axis=0)
    total_aids = np.concatenate(total_aids, axis=0)
    total_pred = np.concatenate(total_pred, axis=0)
    total_labels = np.concatenate(total_labels, axis=0)
    MAP, MRR = eval_map_mrr(total_qids, total_aids, total_pred, total_labels)
    acc_ = np.sum(Acc)/count
    ave_loss = total_loss/count
    # print('Eval loss:{}'.format(total_loss / count))
    return MAP, MRR, ave_loss, acc_


def test(corpus, config):
    with tf.Session() as sess:
        model = SiameseQACNN(config)
        saver = tf.train.Saver()
        saver.restore(sess, tf.train.latest_checkpoint(best_path))
        test_MAP, test_MRR, _, acc = evaluate(sess, model, corpus, config)
        print('start test...............')
        print("-- test MAP %.5f -- test MRR %.5f" % (test_MAP, test_MRR))


def train(train_corpus, val_corpus, test_corpus, config, eval_train_corpus=None):
    iterator = Iterator(train_corpus)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if not os.path.exists(best_path):
        os.makedirs(best_path)

    with tf.Session() as sess:
        # training
        print('Start training and evaluating ...')
        start_time = time.time()

        model = SiameseQACNN(config)
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=10)
        best_saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)
        ckpt = tf.train.get_checkpoint_state(save_path)
        print('Configuring TensorBoard and Saver ...')
        summary_writer = tf.summary.FileWriter(save_path, graph=sess.graph)
        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            print('Reloading model parameters..')
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            print('Created new model parameters..')
            sess.run(tf.global_variables_initializer())

        # count trainable parameters
        total_parameters = count_parameters()
        print('Total trainable parameters : {}'.format(total_parameters))

        current_step = 0
        best_map_val = 0.0
        best_mrr_val = 0.0
        last_dev_map = 0.0
        last_dev_mrr = 0.0
        for epoch in range(config.num_epochs):
            print("----- Epoch {}/{} -----".format(epoch + 1, config.num_epochs))
            count = 0
            for batch_x in iterator.next(config.batch_size, shuffle=True):

                batch_q, batch_a_pos, batch_a_neg, batch_qmask, batch_a_pos_mask, batch_a_neg_mask = zip(*batch_x)
                batch_q = np.asarray(batch_q)
                batch_a_pos = np.asarray(batch_a_pos)
                batch_a_neg = np.asarray(batch_a_neg)
                _, loss, summary, train_acc = sess.run([model.train_op, model.total_loss, model.summary_op, model.accu],
                                           feed_dict={model._ques: batch_q,
                                                      model._ans: batch_a_pos,
                                                      model._ans_neg: batch_a_neg,
                                                      model.dropout_keep_prob: config.keep_prob})
                count += 1
                current_step += 1
                if count % 100 == 0:
                    print('[epoch {}, batch {}]Loss:{}'.format(epoch, count, loss))
                summary_writer.add_summary(summary, current_step)
            if eval_train_corpus is not None:
                train_MAP, train_MRR, train_Loss, train_acc_ = evaluate(sess, model, eval_train_corpus, config)
                print("--- epoch %d  -- train Loss %.5f -- train Acc %.5f -- train MAP %.5f -- train MRR %.5f" % (
                        epoch+1, train_Loss, train_acc_, train_MAP, train_MRR))
            if val_corpus is not None:
                dev_MAP, dev_MRR, dev_Loss, dev_acc = evaluate(sess, model, val_corpus, config)
                print("--- epoch %d  -- dev Loss %.5f -- dev Acc %.5f --dev MAP %.5f -- dev MRR %.5f" % (
                    epoch + 1, dev_Loss, dev_acc, dev_MAP, dev_MRR))
                logger.info("\nEvaluation:")
                logger.info("--- epoch %d  -- dev Loss %.5f -- dev Acc %.5f --dev MAP %.5f -- dev MRR %.5f" % (
                    epoch + 1, dev_Loss, dev_acc, dev_MAP, dev_MRR))

                test_MAP, test_MRR, test_Loss, test_acc= evaluate(sess, model, test_corpus, config)
                print("--- epoch %d  -- test Loss %.5f -- test Acc %.5f --test MAP %.5f -- test MRR %.5f" % (
                    epoch + 1, test_Loss, test_acc, test_MAP, test_MRR))
                logger.info("\nTest:")
                logger.info("--- epoch %d  -- test Loss %.5f -- dev Acc %.5f --test MAP %.5f -- test MRR %.5f" % (
                    epoch + 1, test_Loss, test_acc, test_MAP, test_MRR))

                checkpoint_path = os.path.join(save_path, 'map{:.5f}_{}.ckpt'.format(test_MAP, current_step))
                bestcheck_path = os.path.join(best_path, 'map{:.5f}_{}.ckpt'.format(test_MAP, current_step))
                saver.save(sess, checkpoint_path, global_step=epoch)
                if test_MAP > best_map_val or test_MRR > best_mrr_val:
                    best_map_val = test_MAP
                    best_mrr_val = test_MRR
                    best_saver.save(sess, bestcheck_path, global_step=epoch)
                last_dev_map = test_MAP
                last_dev_mrr = test_MRR
        logger.info("\nBest and Last:")
        logger.info('--- best_MAP %.4f -- best_MRR %.4f -- last_MAP %.4f -- last_MRR %.4f'% (
            best_map_val, best_mrr_val, last_dev_map, last_dev_mrr))
        print('--- best_MAP %.4f -- best_MRR %.4f -- last_MAP %.4f -- last_MRR %.4f' % (
            best_map_val, best_mrr_val, last_dev_map, last_dev_mrr))


def main(args):
    max_q_length = 25
    max_a_length = 90
    processed_data_path_pairwise = '../data/WikiQA/processed/pairwise'
    train_file = os.path.join(processed_data_path_pairwise, 'WikiQA-train-triplets.tsv')
    dev_file = os.path.join(processed_data_path_pairwise, 'WikiQA-dev.tsv')
    test_file = os.path.join(processed_data_path_pairwise, 'WikiQA-test.tsv')
    vocab = os.path.join(processed_data_path_pairwise, 'wiki_clean_vocab.txt')
    embeddings_file = os.path.join(processed_data_path_pairwise, 'wiki_embedding.pkl')
    _embeddings = load_embedding(embeddings_file)
    train_transform = transform_train(train_file, vocab)
    dev_transform = transform(dev_file, vocab)
    test_transform = transform(test_file, vocab)
    train_corpus = load_train_data(train_transform, max_q_length, max_a_length)
    dev_corpus = load_data(dev_transform, max_q_length, max_a_length, keep_ids=True)
    test_corpus = load_data(test_transform, max_q_length, max_a_length, keep_ids=True)

    config = NNConfig(embeddings=_embeddings)
    config.ques_length = max_q_length
    config.ans_length = max_a_length
    if args.train:
        train(deepcopy(train_corpus), dev_corpus, test_corpus, config)
    elif args.test:
        test(test_corpus, config)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", help="whether to train", action='store_true')
    parser.add_argument("--test", help="whether to test", action='store_true')
    args = parser.parse_args()

    save_path = "./model/checkpoint"
    best_path = "./model/bestval"
    main(args)
