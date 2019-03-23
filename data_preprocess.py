# -*- coding: utf-8 -*-
# @Time    : 2019/3/19 15:09
# @Author  : Alan
# @Email   : xiezhengwen2013@163.com
# @File    : data_preprocess.py
# @Software: PyCharm

import nltk
import codecs
import logging
import numpy as np
import re
from collections import defaultdict
import pickle
import os
from collections import Counter
import copy
import random
#nltk.download('wordnet')

raw_data_path = '../data/WikiQA/raw'
lemmatized_data_path = '../data/WikiQA/lemmatized'
processed_data_path = '../data/WikiQA/processed'
glove_path = '../glove//glove.6B.300d.txt'

processed_data_path_pointwise = '../data/WikiQA/processed/pointwise'
processed_data_path_pairwise = '../data/WikiQA/processed/pairwise'

if not os.path.exists(lemmatized_data_path):
    os.mkdir(lemmatized_data_path)

if not os.path.exists(processed_data_path):
    os.mkdir(processed_data_path)

if not os.path.exists(processed_data_path_pointwise):
    os.mkdir(processed_data_path_pointwise)

if not os.path.exists(processed_data_path_pairwise):
    os.mkdir(processed_data_path_pairwise)

class QaSample(object):
    def __init__(self, q_id, question, a_id, answer, label=None, score=0):
        self.q_id = q_id
        self.question = question
        self.a_id = a_id
        self.answer = answer
        self.label = int(label)
        self.score = float(score)


def load_qa_data(fname):
    with open(fname, 'r', encoding='utf-8') as fin:
        for line in fin:
            try:
                q_id, question, a_id, answer, label = line.strip().split('\t')
            except ValueError:
                q_id, question, a_id, answer = line.strip().split('\t')
                label = 0
            yield QaSample(q_id, question, a_id, answer, label)


def lemmatize():
    wn_lemmatizer = nltk.stem.WordNetLemmatizer()
    data_sets = ['WikiQA-train.tsv', 'WikiQA-dev.tsv', 'WikiQA-test.tsv']
    for set_name in data_sets:
        fin_path = os.path.join(raw_data_path, set_name)
        fout_path = os.path.join(lemmatized_data_path, set_name)
        with open(fin_path, 'r', encoding='utf-8') as fin, open(fout_path, 'w', encoding='utf-8') as fout:
            fin.readline()
            for line in fin:
                line_info = line.strip().split('\t')
                q_id = line_info[0]
                question = line_info[1]
                a_id = line_info[4]
                answer = line_info[5]
                question = ' '.join(map(lambda x: wn_lemmatizer.lemmatize(x), nltk.word_tokenize(question)))
                answer = ' '.join(map(lambda x: wn_lemmatizer.lemmatize(x), nltk.word_tokenize(answer)))
                if set_name != 'test':
                    label = line_info[6]
                    fout.write('\t'.join([q_id, question, a_id, answer, label]) + '\n')
                else:
                    fout.write('\t'.join([q_id, question, a_id, answer]) + '\n')


def gen_train_triplets(same_q_sample_group):
    question = same_q_sample_group[0].question
    pos_answers = [sample.answer for sample in same_q_sample_group if sample.label == 1]
    neg_answers = [sample.answer for sample in same_q_sample_group if sample.label == 0]
    if len(pos_answers) != 0:
        for pos_answer in pos_answers:
            for neg_answer in neg_answers:
                yield question, pos_answer, neg_answer


# 获取clean的dev和test数据，写入文件
def gen_clean_test(filename):
    f_in = os.path.join(lemmatized_data_path, filename)
    f_out = os.path.join(processed_data_path_pointwise, filename)
    qa_samples = load_qa_data(f_in)
    dic = {}
    dic2 = {}
    for qasa in qa_samples:
        if qasa.q_id not in dic:
            dic[qasa.q_id] = [qasa.label]
            dic2[qasa.q_id] = [qasa]
        else:
            dic[qasa.q_id].append(qasa.label)
            dic2[qasa.q_id].append(qasa)
    q = []
    for k, v in dic.items():
        if sum(v) != 0:
            q.append(k)
    print('所有label有效（不全为0）的数据为：{}'.format(len(q)))
    with open(f_out, 'w', encoding='utf-8') as fout:
        for t in q:
            same_q_samples = dic2[t]
            for r in same_q_samples:
                fout.write('{}\t{}\t{}\t{}\t{}\n'.format(r.q_id, r.question, r.a_id, r.answer, r.label))


# 获得train、dev、test中所有的词,目前采用lemmatized的，并不是clean后的，但是没有什么影响
def gen_vocab():
    words = []
    data_sets = ['WikiQA-train.tsv', 'WikiQA-dev.tsv', 'WikiQA-test.tsv']
    for set_name in data_sets:
        fin_path = os.path.join(lemmatized_data_path, set_name)
        with open(fin_path, 'r', encoding='utf-8') as fin:
            fin.readline()
            for line in fin:
                line_in = line.strip().split('\t')
                question = line_in[1].split(' ')
                answer = line_in[3].split(' ')
                for r1 in question:
                    if r1 not in words:
                        words.append(r1)
                for r2 in answer:
                    if r2 not in words:
                        words.append(r2)
    fout_path = os.path.join(processed_data_path_pointwise, 'wiki_vocab.txt')
    with open(fout_path, 'w', encoding='utf-8') as f:
        for i, j in enumerate(words):
            f.write('{}\t{}\n'.format(i, j))


# 根据词表生成对应的embedding
def data_transform(embedding_size):
    file_in = os.path.join(processed_data_path_pointwise, 'wiki_vocab.txt')
    clean_vocab_out = os.path.join(processed_data_path_pointwise, 'wiki_clean_vocab.txt')
    embedding_out = os.path.join(processed_data_path_pointwise, 'wiki_embedding.pkl')
    words = []
    with open(file_in, 'r', encoding='utf-8') as f1:
        for line in f1:
            word = line.strip().split('\t')[1].lower()
            words.append(word)
    print('wiki_vocab.txt总共有{}个词'.format(len(words)))

    embedding_dic = {}
    rng = np.random.RandomState(None)
    pad_embedding = rng.uniform(-0.25, 0.25, size=(1, embedding_size))
    unk_embedding = rng.uniform(-0.25, 0.25, size=(1, embedding_size))
    embeddings = []
    clean_words = ['<PAD>', '<UNK>']
    embeddings.append(pad_embedding.reshape(-1).tolist())
    embeddings.append(unk_embedding.reshape(-1).tolist())
    print('uniform_init...')
    with open(glove_path, 'r', encoding='utf-8') as fin:
        for line in fin:
            try:
                line_info = line.strip().split()
                word = line_info[0]
                embedding = [float(val) for val in line_info[1:]]
                embedding_dic[word] = embedding
                if word in words:
                    clean_words.append(word)
                    embeddings.append(embedding)
            except:
                print('Error while loading line: {}'.format(line.strip()))
    print('目前词表总共有{}个词'.format(len(clean_words)))
    print('embeddings总共有{}个词'.format(len(embeddings)))
    print('embeddings的shape为： {}'.format(np.shape(embeddings)))
    with open(clean_vocab_out, 'w', encoding='utf-8') as f:
        for i, j in enumerate(clean_words):
            f.write('{}\t{}\n'.format(i, j))
    with open(embedding_out, 'wb') as f2:
        pickle.dump(embeddings, f2)


if __name__ == '__main__':
    # 1.nltk分词
    # 2.获取clean的train, dev和test数据，写入文件
    # 3.获取词表
    # 4.生成相应的embedding

    type = 'pointwise'
    # 1.nltk分词
    # lemmatize()

    # 2.获取clean的train, dev和test数据，写入文件
    train_file = 'WikiQA-train.tsv'
    dev_file = 'WikiQA-dev.tsv'
    test_file = 'WikiQA-test.tsv'
    gen_clean_test(train_file)
    gen_clean_test(dev_file)
    gen_clean_test(test_file)

    # 3.获取对应的词表
    gen_vocab()

    # 4.生成相应的embedding
    data_transform(300)