# -*- coding: utf-8 -*-
# @Time    : 2019/3/19 16:10
# @Author  : Alan
# @Email   : xiezhengwen2013@163.com
# @File    : data_helper.py
# @Software: PyCharm

import sys
import numpy as np
import random
from collections import namedtuple
import pickle

random.seed(1337)
np.random.seed(1337)

def load_embedding(dstPath):
    with open(dstPath, 'rb') as fin:
        _embeddings = pickle.load(fin)
    print("load embedding finish! embedding shape:{}".format(np.shape(_embeddings)))
    return _embeddings


class Batch:
    # batch类，里面包含了encoder输入，decoder输入，decoder标签，decoder样本长度mask
    def __init__(self):
        self.quest_id = []
        self.ans_id = []
        self.quest = []
        self.ans = []
        self.quest_mask = []
        self.ans_mask = []
        self.label = []


def transform(fin_path, vocab, unk_id=1):
    word2id = {}
    transformed_corpus = []
    with open(vocab, 'r', encoding='utf-8') as f1:
        for line in f1:
            word = line.strip().split('\t')[1].lower()
            id = int(line.strip().split('\t')[0])
            word2id[word] = id
    with open(fin_path, 'r', encoding='utf-8') as fin:
        fin.readline()
        for line in fin:
            qid, q, aid, a, label = line.strip().split('\t')
            q = [word2id.get(w.lower(), unk_id) for w in q.split()]
            a = [word2id.get(w.lower(), unk_id) for w in a.split()]
            transformed_corpus.append([qid, q, aid, a, int(label)])
    return transformed_corpus


def transform_train(fin_path, vocab, unk_id=1):
    word2id = {}
    transformed_corpus = []
    with open(vocab, 'r', encoding='utf-8') as f1:
        for line in f1:
            word = line.strip().split('\t')[1].lower()
            id = int(line.strip().split('\t')[0])
            word2id[word] = id
    with open(fin_path, 'r', encoding='utf-8') as fin:
        fin.readline()
        for line in fin:
            q, a_pos, a_neg = line.strip().split('\t')
            q = [word2id.get(w.lower(), unk_id) for w in q.split()]
            a_pos = [word2id.get(w.lower(), unk_id) for w in a_pos.split()]
            a_neg = [word2id.get(w.lower(), unk_id) for w in a_neg.split()]
            transformed_corpus.append([q, a_pos, a_neg])
    return transformed_corpus


def padding(sent, sequence_len):
    """
     convert sentence to index array
    """
    if len(sent) > sequence_len:
        sent = sent[:sequence_len]
    padding = sequence_len - len(sent)
    sent2idx = sent + [0]*padding
    return sent2idx, len(sent)


def load_train_data(transformed_corpus, ques_len, ans_len):
    """
        load train data
        """
    pairwise_corpus = []
    for sample in transformed_corpus:
        q, a_pos, a_neg = sample
        q_pad, q_len = padding(q, ques_len)
        a_pos_pad, a_pos_len = padding(a_pos, ans_len)
        a_neg_pad, a_neg_len = padding(a_neg, ans_len)
        pairwise_corpus.append((q_pad, a_pos_pad, a_neg_pad, q_len, a_pos_len, a_neg_len))
    return pairwise_corpus


def load_data(transformed_corpus, ques_len, ans_len, keep_ids=False):
    """
    load test data
    """
    pairwise_corpus = []
    for sample in transformed_corpus:
        qid, q, aid, a, label = sample
        q_pad, q_len = padding(q, ques_len)
        a_pad, a_len = padding(a, ans_len)
        if keep_ids:
            pairwise_corpus.append((qid, q_pad, aid, a_pad, q_len, a_len, label))
        else:
            pairwise_corpus.append((q_pad, a_pad, q_len, a_len, label))
    return pairwise_corpus


class Iterator(object):
    """
    数据迭代器
    """
    def __init__(self, x):
        self.x = x
        self.sample_num = len(self.x)

    def next_batch(self, batch_size, shuffle=True):
        # produce X, Y_out, Y_in, X_len, Y_in_len, Y_out_len
        if shuffle:
            np.random.shuffle(self.x)
        l = np.random.randint(0, self.sample_num - batch_size + 1)
        r = l + batch_size
        x_part = self.x[l:r]
        return x_part

    def next(self, batch_size, shuffle=False):
        if shuffle:
            np.random.shuffle(self.x)
        l = 0
        while l < self.sample_num:
            r = min(l + batch_size, self.sample_num)
            batch_size = r - l
            x_part = self.x[l:r]
            l += batch_size
            yield x_part


if __name__ == '__main__':
    train = '../data/WikiQA/processed/pointwise/WikiQA-train.tsv'
    vocab = '../data/WikiQA/processed/pointwise/wiki_clean_vocab.txt'
    transform(train, vocab)
