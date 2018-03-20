"""
description: this file helps to load raw file and gennerate batch x,y
author:luchi
date:22/11/2016
"""

import os


# import cPickle as pkl
# import _pickle as pkl
import sys
sys.path.append("..")
import random
import json
import traceback

import jieba
import numpy as np
import pandas as pd
from collections import OrderedDict

import utils.embedding.vector_helper as vector_helper
from utils.embedding.vector_helper import BenebotVector as VectorHelper
import utils.query_util as query_util

# file path
# dataset_path = '/data/PycharmProjects/question_matching_framework/work_space/example/dataset/aaa'

def load_cn_data_from_files(classify_files):
    count = len(classify_files)
    x_text = []
    y = []
    for index in range(count):
        classify_file = classify_files[index]
        lines = list(open(classify_file, "r").readlines())
        label = [0] * count
        label[index] = 1
        labels = [label for _ in lines]
        if index == 0:
            x_text = lines
            y = labels
        else:
            x_text = x_text + lines
            y = np.concatenate([y, labels])
    x_text = [clean_str_cn(sent) for sent in x_text]
    return [x_text, y]

def load_cn_data_from_files2(classify_files,candidates_file):
    pass


def clean_str_cn(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    return string.strip().lower()

def load_data(classify_files,config, sort_by_len=True, enhance = False, reverse=False):
    x_text, y = load_cn_data_from_files(classify_files)
    # print(x_text[:10],y)

    new_text = []
    if reverse == True:
        for text in x_text:
            text_list = text.strip().split(' ')
            text_list.reverse()
            reversed_text = ' '.join(text_list)
            new_text.append(reversed_text)
        x_text = new_text
    else:
        pass

    y = list(y)

    original_dataset = list(zip(x_text, y))

    if enhance == True:
        num_sample = len(original_dataset)

        # shuffle
        for i in range(num_sample):
            text_list = original_dataset[i][0].split(' ')
            random.shuffle(text_list)
            text_shuffled = ' '.join(text_list)
            label_shuffled = original_dataset[i][1]
            x_text.append(text_shuffled)
            y.append(label_shuffled)

    else:
        pass

    # Randomly shuffle data
    shuffle_indices = list(range(len(y)))
    random.shuffle(shuffle_indices)
    # print(shuffle_indices)
    x_shuffled = []
    y_shuffled_tmp = []
    for shuffle_indice in shuffle_indices:
        x_shuffled.append(x_text[shuffle_indice])
        y_shuffled_tmp.append(y[shuffle_indice])
    y_shuffled = np.array(y_shuffled_tmp)

    # train_set length
    n_samples = len(x_shuffled)
    # shuffle and generate train and valid data set
    sidx = np.random.permutation(n_samples)
    n_train = int(np.round(n_samples * (1. - config.valid_portion)))
    print("Train/Test split: {:d}/{:d}".format(n_train, (n_samples - n_train)))
    valid_set_x = [x_shuffled[s] for s in sidx[n_train:]]
    valid_set_y = [y_shuffled[s] for s in sidx[n_train:]]
    train_set_x = [x_shuffled[s] for s in sidx[:n_train]]
    train_set_y = [y_shuffled[s] for s in sidx[:n_train]]

    train_set = (train_set_x, train_set_y)
    valid_set = (valid_set_x, valid_set_y)
    # test_set = (x_test, y_test)

    # test_set_x, test_set_y = test_set
    valid_set_x, valid_set_y = valid_set
    train_set_x, train_set_y = train_set

    def len_argsort(seq):
        return sorted(range(len(seq)), key=lambda x: len(seq[x]))

    if sort_by_len:

        sorted_index = len_argsort(valid_set_x)
        valid_set_x = [valid_set_x[i] for i in sorted_index]
        valid_set_y = [valid_set_y[i] for i in sorted_index]

        sorted_index = len_argsort(train_set_x)
        train_set_x = [train_set_x[i] for i in sorted_index]
        train_set_y = [train_set_y[i] for i in sorted_index]

    train_set=(train_set_x,train_set_y)
    valid_set=(valid_set_x,valid_set_y)

    max_len = config.num_step

    def generate_mask(data_set):
        set_x = data_set[0]
        mask_x = np.zeros([max_len, len(set_x)])
        for i,x in enumerate(set_x):
            x_list = x.split(' ')
            if len(x_list) < max_len:
                mask_x[0:len(x_list), i] = 1
            else:
                mask_x[:, i] = 1
        new_set = (set_x, data_set[1], mask_x)
        return new_set

    train_set = generate_mask(train_set)
    valid_set = generate_mask(valid_set)

    train_data = (train_set[0], train_set[1], train_set[2])
    valid_data = (valid_set[0], valid_set[1], valid_set[2])

    return train_data, valid_data

# return batch data set
def batch_iter(data,batch_size, shuffle = True):
    # get data set and label
    x, y, mask_x = data

    # mask_x = np.array(mask_x)
    mask_x = np.asarray(mask_x).T.tolist()

    data_size = len(x)
    if shuffle:
        shuffle_indices = list(range(data_size))
        random.shuffle(shuffle_indices)
        shuffled_x = []
        shuffled_y = []
        shuffled_mask_x = []

        for shuffle_indice in shuffle_indices:
            shuffled_x.append(x[shuffle_indice])
            shuffled_y.append(y[shuffle_indice])
            shuffled_mask_x.append(mask_x[shuffle_indice])
    else:
        shuffled_x = x
        shuffled_y = y
        shuffled_mask_x = mask_x

    shuffled_mask_x = np.asarray(shuffled_mask_x).T  # .tolist()

    shuffled_x = np.array(shuffled_x)
    shuffled_y = np.array(shuffled_y)
    shuffled_mask_x = np.array(shuffled_mask_x)

    # num_batches_per_epoch=int((data_size-1)/batch_size) + 1
    num_batches_per_epoch = data_size // batch_size

    for batch_index in range(num_batches_per_epoch):
        start_index=batch_index*batch_size
        end_index=min((batch_index+1)*batch_size,data_size)
        return_x = shuffled_x[start_index:end_index]
        return_y = shuffled_y[start_index:end_index]
        return_mask_x = shuffled_mask_x[:,start_index:end_index]

        yield (return_x,return_y,return_mask_x)

def convert(data_path):
    prefix=os.path.join(os.path.split(data_path)[0],'train')
    os.mkdir(prefix)
    with open(data_path,'r') as f:
        data=f.readlines()
    files=dict()
    for line in data:
        line=line.strip()
        query=line.split('#')[0]
        label=line.split('#')[1]
        if label not in files:
            path=os.path.join(prefix,label)
            os.mknod(path)
            f=open(path,'w')
            files[label]=f
        else:
            f=files[label]
        tokens=list(jieba.cut(query))
        tokens=' '.join(tokens)
        f.write(tokens+'\n')
    for _,f in files.items():
        f.close()
    for i in range(100):
        path=os.path.join(prefix, 'reserved_' + str(i))
        os.mknod(path)
        with open(path,'w') as f:
            f.write('reserved_' + str(i))

def load_candidates():
    candidates = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
    candid2idx = {"toxic":0, "severe_toxic":1, "obscene":2, "threat":3, "insult":4, "identity_hate":5}
    idx2candid = {0:"toxic", 1:"severe_toxic", 2:"obscene", 3:"threat", 4:"insult", 5:"identity_hate"}
    # with open(candidates_f) as f:
    #     for i, line in enumerate(f):
    #         candid2idx[line.strip()] = i
    #         idx2candid[i] = line.strip()
    #         candidates.append(line.strip())

    return candidates, candid2idx, idx2candid


def load_dialog(sentences, data_dir, partition, partitions):
    train_file = os.path.join(data_dir, 'train.csv')
    # test_file = os.path.join(data_dir, 'test.txt')
    # val_file = os.path.join(data_dir, 'val.txt')

    all_data = get_dialogs(sentences, partition, partitions)
    train_data = all_data[0: int(0.8 * len(all_data))]
    # test_data = all_data[int(0.8 * len(all_data)):]
    val_data = all_data[int(0.8 * len(all_data)):]
    return train_data, val_data


def get_dialogs(sentences, partition, partitions):
    """

    Given a file name, read the file, retrieve the dialogs, and then convert the sentences into a single dialog.
    If max_length is supplied, any stories longer than max_length tokens will be discarded.

    :param sentences:
    :param f:
    :param candid_dic:
    :param char:
    :return:
    """

    return parse_dialogs_per_response(sentences, partition, partitions)


def parse_dialogs_per_response(sentences, partition, partitions):
    """
        Parse dialogs provided in the babi tasks format
    :param sentences:
    :param lines:
    :param candid_dic:
    :param char:
    :param plain:
    :return:
    """
    data = []
    context = []

    df = pd.read_csv('/opt/luis/toxic/data/train/automata/train.csv',
                     names=["id","comment_text","toxic", "severe_toxic",
                            "obscene", "threat",
                            "insult", "identity_hate"], skiprows=1)
    from tqdm import tqdm
    rows = df.shape[0]
    fr = float(partition) / partitions * rows
    to = float(partition + 1) / partitions * rows
    fr = int(fr)
    to = int(to)
    df = pd.read_csv('/opt/luis/toxic/data/train/automata/train.csv',
                     names=["id", "comment_text", "toxic", "severe_toxic",
                            "obscene", "threat",
                            "insult", "identity_hate"], skiprows=1 + fr, nrows=to - fr)
    # print(candid_dic)
    for index, row in df.iterrows():

        # if r not in candid_dic:
        #     print('warning candidate is not listed..', r)
        #     continue
        a = [row['toxic'], row['severe_toxic'], row['obscene'], row['threat'], row['insult'], row['identity_hate']]
        # print(a)
        u = row['comment_text']
        _id_ = row['id']
        u = query_util.tokenize(u, char=8)
        # r = query_util.tokenize(r, char=char)
        sentences.append(u)
        # sentences.add(vector_helper.SEPERATOR.join(r))

        # print(u)
        # temporal encoding, and utterance/response encoding
        # data.append((context[:],u[:],candid_dic[' '.join(r)]))
        data.append((context[:], u[:], a))

    # print(data)
    sentences.append(["<PAD>"])
    # sentences.add(vector_helper.PAD)
    random.shuffle(data)
    return data


def get_sentence_lens(inputs):
    # print(inputs[0])
    lens = np.zeros((len(inputs)), dtype=int)
    sen_lens = []
    max_sen_lens = []
    for i, t in enumerate(inputs):
        sentence_lens = np.zeros((len(t)), dtype=int)
        for j, s in enumerate(t):
            sentence_lens[j] = len(s)
        lens[i] = len(t)

        sen_lens.append(sentence_lens)
        max_sen_lens.append(np.max(sentence_lens))

    return lens, sen_lens, max(max_sen_lens) if len(max_sen_lens) else 0


def get_lens(inputs, split_sentences=False):
    lens = np.zeros((len(inputs)), dtype=int)
    for i, t in enumerate(inputs):
        lens[i] = len(t)
    return lens


def load_raw_data(config, partition, partitions):
    print('Load raw data.....partition {} of {}'.format(partition, partitions))
    candidates, candid2idx, idx2candid = load_candidates()
    candidate_size = len(candidates)

    sentences = list()
    train_data, val_data = load_dialog(sentences,
                                                  data_dir=config["data_dir"],
                                                  partition=partition,
                                                  partitions=partitions)
    total_data = train_data
    total_data += val_data

    # inputs = []
    questions = []
    answers = []
    # relevant_labels = []
    # input_masks = []
    for data in total_data:
        inp, question, answer = data
        # inp = inp[-config.max_memory_size:]
        # if len(inp) == 0:
        #     inp = [[config.EMPTY]]
        # inputs.append(inp)
        if len(question) == 0:
            question = [vector_helper.EMPTY]
        questions.append(question)
        answers.append(answer)
        # relevant_labels.append([0])
    # if not config.split_sentences:
    #     if input_mask_mode == 'word':
    #         input_masks.append(
    #             np.array([index for index, w in enumerate(inp)], dtype=np.int32))
    #     elif input_mask_mode == 'sentence':
    #         input_masks.append(
    #             np.array([index for index, w in enumerate(inp) if w == '.'], dtype=np.int32))
    #     else:
    #         raise Exception("invalid input_mask_mode")

    q_lens = get_lens(questions)
    max_q_len = np.max(q_lens) if len(q_lens) > 0 else 0

    answers = np.stack(answers)

    num_train = int(len(questions) * 0.8)
    train = questions[:num_train], q_lens[:num_train], answers[:num_train]
    valid = questions[num_train:], q_lens[num_train:], answers[num_train:]

    w2idx = None
    idx2w = None
    vocab_size = -1

    metadata = dict()
    data = dict()
    data['train'] = train
    data['valid'] = valid
    # data['sentences'] = sentences

    # metadata['sentences_embedding'] = sentences_embedding
    metadata['sentences'] = sentences
    metadata['max_q_len'] = max_q_len
    metadata['max_sen_len'] = max_q_len
    metadata['candidate_size'] = candidate_size
    metadata['candid2idx'] = candid2idx
    metadata['idx2candid'] = idx2candid
    metadata['w2idx'] = w2idx
    metadata['idx2w'] = idx2w
    metadata['vocab_size'] = vocab_size

    return metadata, data


def pad_inputs(inputs, max_len):
    # print(max_len, max([len(inputs[i]) for i in range(len(inputs))]))
    padded = [np.pad(inp, (0, max_len - len(inputs[i])),
                     'constant', constant_values=0) for i, inp in enumerate(inputs)]
    # print(padded)
    return padded


def vectorize_data(data, metadata):
    questions, q_lens, answers = data

    sentences_embedding = metadata['sentences_embedding']
    # q_embedding = sentences_embedding['questions_embedding']
    # a_embedding = sentences_embedding['answers_embedding']

    questions_embeddings = []
    for question in questions:
        question = ",".join(question)
        questions_embeddings.append(sentences_embedding[question])
    questions_embeddings = np.asarray(
        questions_embeddings, dtype=np.float32)

    data = questions_embeddings
    return data


def sentence_embedding_core(metadata):
    vector_helper = VectorHelper()

    sentences = list(metadata["sentences"])
    split_sentences = [sen for sen in metadata["sentences"]]
    max_q_len = metadata['max_q_len']
    # print(max_q_len)
    pad_sentences = pad_inputs(split_sentences, max_q_len)
    sentences_embedding = OrderedDict()
    lens = len(pad_sentences)
    from tqdm import tqdm
    for index in tqdm(range(lens)):
        sen = pad_sentences[index]
        sen = sen.tolist()
        join_sen = sentences[index]
        sen = list(map(lambda x: [x, "<PAD>"][x == ""], sen))
        # print(sen)
            # sen_embedding = [ff_embedding_local(word) for word in sen]
        sen_embedding = [vector_helper.getVector(word) for word in sen]
        sentences_embedding[",".join(join_sen)] = sen_embedding

    #     inp_empty_embedding = [vector_helper.getVector(vector_helper.PAD) for _ in range(max_len)]
    #
    # sentences_embedding[config.EMPTY] = inp_empty_embedding
    # sentences_embedding[config.PAD] = [vector_helper.getVector(vector_helper.PAD)] * max_len
    return sentences_embedding


def sentence_embedding(metadata):
    sentences_embedding = sentence_embedding_core(metadata)
    return sentences_embedding


if __name__ == '__main__':
    convert('../work_space/xhsd_wj/automata/train.txt')