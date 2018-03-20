# coding:utf-8
"""
-------------------------------------------------
   File Name：     rnn_model
   Description :
   Author :       deep
   date：          18-1-11
-------------------------------------------------
   Change Activity:
                   18-1-11:

   __author__ = 'deep'
-------------------------------------------------
"""

import sys
import os
import time
import codecs
import argparse
import re

parentdir = os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))
sys.path.insert(0, parentdir)

grandfatherdir = os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))

sys.path.insert(0, grandfatherdir)
sys.path.append(grandfatherdir)

import numpy as np
import pickle
import tensorflow as tf
from tqdm import tqdm

import rnn.data_helper as data_helper
from rnn.config import RNNConfig as Config
from rnn.model import RNN_Model
import utils.embedding.vector_helper as vector_helper


def prepare_data(data_config):
    metadata, data = data_helper.load_raw_data(data_config)

    with open(data_config["metadata_path"], 'wb') as f:
        pickle.dump(metadata, f)
    with open(data_config["data_path"], 'wb') as f:
        pickle.dump(data, f)


def load_data(data_config):
    """Loads metadata and data"""
    with open(data_config["metadata_path"], 'rb') as f:
        metadata = pickle.load(f)

    with open(data_config["data_path"], 'rb') as f:
        data = pickle.load(f)
    train = data['train']
    valid = data['valid']

    # embedding
    print('pre embedding')
    sentences_embedding = data_helper.sentence_embedding(metadata)
    print('pre embedding done')
    # metadata['max_q_len'] = max_len
    # metadata['max_sen_len'] = max_len
    metadata['sentences_embedding'] = sentences_embedding

    return train, valid, metadata

def train(config):
    pass


def parse_args(args):
    parser = argparse.ArgumentParser(description='LSTM')
    parser.add_argument("-r", "--restore", action='store_true',
                        help="restore previously trained weights")
    parser.add_argument("-s", "--strong_supervision",
                        help="use labelled supporting facts (default=false)")
    parser.add_argument("-l", "--l2_loss", type=float,
                        default=0.001, help="specify l2 loss constant")
    parser.add_argument("-n", "--num_runs", type=int,
                        help="specify the number of model runs")
    parser.add_argument(
        "-p", "--infer", action='store_true', help="predict")
    parser.add_argument("-t", "--train", action='store_true', help="train")
    parser.add_argument("-d", "--prep_data",
                        action='store_true', help="prepare data")

    args = vars(parser.parse_args(args))
    return args


def inference(config):
    pass

def main(args):
    args = parse_args(args)
    config = Config()
    args['train'] = True
    if args['prep_data']:
        data_config = config.data_config
        print('\n>> Preparing Data\n')
        begin = time.clock()
        prepare_data(data_config)
        end = time.clock()
        print('>> Preparing Data Time:{}'.format(end - begin))
        sys.exit()
    elif args['train']:
        train(config)
    elif args['inference']:
        inference(config)
    else:
        print('ERROR:Unknow Mode')


if __name__ == '__main__':
    main(sys.argv[1:])
