# coding: UTF-8
import os
import yaml
import time
import h5py
import pickle
from datetime import timedelta
import random
import string


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def read_config(path):
    return AttrDict(yaml.safe_load(open(path, 'r', encoding='utf-8')))


def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))


def load_data(cache_file_h5py, cache_file_pickle):
    """
    load data from h5py and pickle cache files
    :param cache_file_h5py:
    :param cache_file_pickle:
    :return:
    """
    if not os.path.exists(cache_file_h5py) or not os.path.exists(cache_file_pickle):

        raise RuntimeError("############################ERROR##############################\n. "
                           "请先准备数据集")
    f_data = h5py.File(cache_file_h5py, 'r')
    # return narray
    # https://stackoverflow.com/questions/46733052/read-hdf5-file-into-numpy-array
    input_ids = f_data['input_ids'][()]
    attention_mask = f_data['attention_mask'][()]
    label_list = f_data['label_list'][()]

    label2index = None
    with open(cache_file_pickle, 'rb') as data_f_pickle:
        label2index = pickle.load(data_f_pickle)
    return label2index,input_ids,attention_mask,label_list

def load_labels(cache_file_h5py):
    """
    load data from h5py and pickle cache files
    :param cache_file_h5py:
    :param cache_file_pickle:
    :return:
    """
    if not os.path.exists(cache_file_h5py):

        raise RuntimeError("############################ERROR##############################\n. "
                           "请先准备数据集")
    f_data = h5py.File(cache_file_h5py, 'r')
    # return narray
    # https://stackoverflow.com/questions/46733052/read-hdf5-file-into-numpy-array
    input_ids = f_data['input_ids'][()]
    attention_mask = f_data['attention_mask'][()]

    return input_ids,attention_mask

def load_formula(cache_file_h5py, cache_file_pickle):
    """
    load data from h5py and pickle cache files
    :param cache_file_h5py:
    :param cache_file_pickle:
    :return:
    """
    if not os.path.exists(cache_file_h5py) or not os.path.exists(cache_file_pickle):

        raise RuntimeError("############################ERROR##############################\n. "
                           "请先准备数据集")
    f_data = h5py.File(cache_file_h5py, 'r')
    # return narray
    # https://stackoverflow.com/questions/46733052/read-hdf5-file-into-numpy-array
    formula = f_data['X_mathml'][()]

    word2index, label2index = None, None
    with open(cache_file_pickle, 'rb') as data_f_pickle:
        word2index, label2index = pickle.load(data_f_pickle)
    return word2index, label2index, formula


def load_embed_data(embedding_pickle):
    '''加载预训练向量 narray'''
    embeddings = None
    with open(embedding_pickle, 'rb') as data_f_pickle:
        embeddings = pickle.load(data_f_pickle)
    return embeddings


def randomword(length):
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(length))
