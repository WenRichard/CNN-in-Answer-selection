# -*- coding: utf-8 -*-
# @Time    : 2019/3/19 22:11
# @Author  : Alan
# @Email   : xiezhengwen2013@163.com
# @File    : model_utils.py
# @Software: PyCharm

import tensorflow as tf
import numpy as np


def eval_map_mrr(qids, aids, preds, labels):
    # 衡量map指标和mrr指标
    dic = dict()
    pre_dic = dict()
    for qid, aid, pred, label in zip(qids, aids, preds, labels):
        pre_dic.setdefault(qid, [])
        pre_dic[qid].append([aid, pred, label])
    for qid in pre_dic:
        dic[qid] = sorted(pre_dic[qid], key=lambda k: k[1], reverse=True)
        aid2rank = {aid: [label, rank] for (rank, (aid, pred, label)) in enumerate(dic[qid])}
        dic[qid] = aid2rank
    # correct = 0
    # total = 0
    # for qid in dic:
    #     cur_correct = 0
    #     for aid in dic[qid]:
    #         if dic[qid][aid][0] == 1:
    #             cur_correct += 1
    #     if cur_correct > 0:
    #         correct += 1
    #     total += 1
    # print(correct * 1. / total)

    MAP = 0.0
    MRR = 0.0
    useful_q_len = 0
    for q_id in dic:
        sort_rank = sorted(dic[q_id].items(), key=lambda k: k[1][1], reverse=False)
        correct = 0
        total = 0
        AP = 0.0
        mrr_mark = False
        for i in range(len(sort_rank)):
            if sort_rank[i][1][0] == 1:
                correct += 1
        if correct == 0:
            continue
        useful_q_len += 1
        correct = 0
        for i in range(len(sort_rank)):
            # compute MRR
            if sort_rank[i][1][0] == 1 and mrr_mark == False:
                MRR += 1.0 / float(i + 1)
                mrr_mark = True
            # compute MAP
            total += 1
            if sort_rank[i][1][0] == 1:
                correct += 1
                AP += float(correct) / float(total)

        AP /= float(correct)
        MAP += AP

    MAP /= useful_q_len
    MRR /= useful_q_len
    return MAP, MRR


# print tensor shape
def print_shape(varname, var):
    """
    :param varname: tensor name
    :param var: tensor variable
    """
    try:
        print('{0} : {1}'.format(varname, var.get_shape()))
    except:
        print('{0} : {1}'.format(varname, np.shape(var)))


# count the number of trainable parameters in model
def count_parameters():
    totalParams = 0
    for variable in tf.trainable_variables():
        shape = variable.get_shape()
        variableParams = 1
        for dim in shape:
            variableParams *= dim.value
        totalParams += variableParams
    return totalParams


# 余弦相似度计算
def feature2cos_sim(feat_q, feat_a):
    # feat_q: 2D:(bz, hz)
    norm_q = tf.sqrt(tf.reduce_sum(tf.multiply(feat_q, feat_q), 1))
    norm_a = tf.sqrt(tf.reduce_sum(tf.multiply(feat_a, feat_a), 1))
    mul_q_a = tf.reduce_sum(tf.multiply(feat_q, feat_a), 1)
    cos_sim_q_a = tf.div(mul_q_a, tf.multiply(norm_q, norm_a))
    return tf.clip_by_value(cos_sim_q_a, 1e-5, 0.99999)

