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

from sklearn.metrics import roc_auc_score

import rnn.data_helper as data_helper
from rnn.config import RNNConfig as Config
from rnn.model import RNN_Model
import utils.embedding.vector_helper as vector_helper


def run_epoch(model, config, session, data, metadata,
              train_op=None, verbose=2, display=False):
    dp = config.keep_prob
    if train_op is None:
        train_op = tf.no_op()
        dp = 1

    total_steps = len(data[0]) // config.batch_size

    total_loss = []
    accuracy = []
    error = []

    _roc_auc_scores_ = []

    # shuffle data
    # p = np.random.permutation(len(data[0]))
    questions, q_lens, answers = data

    if not display:
        for step in tqdm(range(total_steps)):
            # index = range(step * config.batch_size,
            #               (step + 1) * config.batch_size)
            b = step * config.batch_size
            e = (step + 1) * config.batch_size
            batch_size_data = questions[b:e], q_lens[b:e], answers[b:e]
            question_input = data_helper.vectorize_data(
                batch_size_data, metadata)
            question_lens_input = np.array(q_lens[b:e])
            answer_input = np.array(answers[b:e])
            # print(qp[:2])
            # print(il_vc)
            feed = {model.question_placeholder: question_input,
                    model.question_len_placeholder: question_lens_input,
                    model.target_placeholder: answer_input,
                    model.dropout_placeholder: dp}

            # _ = session.run([train_op], feed_dict=feed)
            loss, pred, _ = session.run(
                [model.loss, model.pred, train_op], feed_dict=feed)

            # print loss
            # total_loss.append(loss)
            # if verbose and step % verbose == 0:
            #     sys.stdout.write('\r{} / {} : loss = {}'.format(
            #         step, total_steps, np.mean(total_loss)))
            #     sys.stdout.flush()

        return display

    for step in range(total_steps):
        # index = range(step * config.batch_size,
        #               (step + 1) * config.batch_size)
        b = step * config.batch_size
        e = (step + 1) * config.batch_size
        batch_size_data = questions[b:e], q_lens[b:e], answers[b:e]
        question_input = data_helper.vectorize_data(batch_size_data, metadata)
        question_lens_input = np.array(q_lens[b:e])
        answer_input = np.array(answers[b:e])

        feed = {model.question_placeholder: question_input,
                model.question_len_placeholder: question_lens_input,
                model.target_placeholder: answer_input,
                model.dropout_placeholder: dp}

        loss, pred, _ = session.run(
            [model.loss, model.predict_proba_op, train_op], feed_dict=feed)

        # if train_writer is not None:
        #     train_writer.add_summary(
        #         summary, num_epoch * total_steps + step)

        batch_answers = batch_size_data[2]
        batch_questions = batch_size_data[0]

        """
        AUC score
        http://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html#sklearn.metrics.roc_auc_score

        >>> import numpy as np
        >>> from sklearn.metrics import roc_auc_score
        >>> y_true = np.array([0, 0, 1, 1])
        >>> y_scores = np.array([0.1, 0.4, 0.35, 0.8])
        >>> roc_auc_score(y_true, y_scores)
        0.75
        """
        for i in range(len(batch_answers)):
            try:
                _roc_auc_score_ = roc_auc_score(batch_answers[i], pred[i])
                _roc_auc_scores_.append(_roc_auc_score_)
            except ValueError:
                pass
        # accuracy.append(np.sum(pred == batch_answers) /
        #                 float(len(batch_answers)))
        # idx2candid = metadata['idx2candid']
        # for Q, A, P in zip(batch_questions, batch_answers, pred):
        #     # print(A, P)
        #     if A != P:
        #         # Q = ''.join([idx2w.get(idx, '')
        #         #              for idx in Q.astype(np.int32).tolist()])
        #         Q = ''.join(Q)
        #         Q = Q.replace(vector_helper.UNK, '')
        #         A = idx2candid[A]
        #         P = idx2candid[P]
        #         error.append((Q, A, P))

        total_loss.append(loss)
        if verbose and step % verbose == 0:
            sys.stdout.write('\r{} / {} : loss = {} --- roc_auc = {}'.format(
                step, total_steps, np.mean(total_loss), np.mean(_roc_auc_scores_)))
            sys.stdout.flush()

    if verbose:
        sys.stdout.write('\r')

    return np.mean(total_loss), np.mean(_roc_auc_scores_), error


def train(config, restore=False):
    # load info
    with open(config.data_config["metadata_path"] + 'info.pkl', 'rb') as f:
        info = pickle.load(f)
    partitions = info['partitions']
    # sentences_embedding = metadata['sentences_embedding']

    model = RNN_Model(config, info)

    print('==> Training RNN start\n')
    # context.train_info.append("Training RNN start")
    print('==> initializing variables')
    # context.train_info.append("initializing variables")
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    with tf.Session() as session:
        session.run(init)

        best_train_epoch = 0
        best_train_loss = float('inf')
        best_train_accuracy = 0
        prev_epoch_loss = float('inf')

        if restore:
            print('==> restoring weights')
            # context.train_info.append("restoring weights")
            _check_restore_parameters(sess=session,
                                      saver=saver, model_path=config.data_config["ckpt_path"])
            # saver.restore(
            #     session, config.data_config["ckpt_path"] + 'rnn.weights')

        print('==> commence training')
        # context.train_info.append("commence training")

        for epoch in range(config.max_epochs):
            partition = epoch % partitions
            train_data, valid_data, metadata = load_data(config.data_config, partition=partition)
            if not (epoch % config.interval_epochs == 0 and epoch > 1):
                # context.train_info.append("Epoch"+str(epoch))
                print('Epoch {}, using partion {}'.format(epoch, partition))
                _ = run_epoch(model=model,
                              config=config,
                              session=session,
                              data=train_data,
                              metadata=metadata,
                              train_op=model.train_op,
                              verbose=2,
                              display=False)
            else:
                # context.train_info.append("Epoch" + str(epoch))
                print('Epoch {}, using partion {}'.format(epoch, partition))
                start = time.time()
                train_loss, train_accuracy, train_error = run_epoch(model=model,
                                                                    config=config,
                                                                    session=session,
                                                                    data=train_data,
                                                                    metadata=metadata,
                                                                    train_op=model.train_op,
                                                                    verbose=2,
                                                                    display=True)
                valid_loss, valid_accuracy, valid_error = run_epoch(model=model,
                                                                    config=config,
                                                                    session=session,
                                                                    data=train_data,
                                                                    metadata=metadata,
                                                                    train_op=model.train_op,
                                                                    verbose=2,
                                                                    display=True)
                if train_accuracy > 0.99:
                    for e in train_error:
                        print(e)
                # context.train_info.append("train_loss:" + str(train_loss) + "&nbsp&nbsptrain_accuracy:" + str(train_accuracy) + "&nbsp&nbsptrain_error:" + str(train_error))
                # context.train_info.append("valid_loss:" + str(valid_loss) + "&nbsp&nbspvalid_accuracy:" + str(valid_accuracy) + "&nbsp&nbspvalid_error:" + str(valid_error))
                print('Training loss: {}'.format(train_loss))
                print('Validation loss: {}'.format(valid_loss))
                print('Training accuracy: {}'.format(train_accuracy))
                print('Vaildation accuracy: {}'.format(valid_accuracy))

                if train_loss < best_train_loss:
                    print('Saving weights and updating best_train_loss:{} -> {},\
                            best_train_accuracy:{} -> {}'.format(best_train_loss, train_loss,
                                                                 best_train_accuracy, train_accuracy))
                    best_train_accuracy = train_accuracy
                    saver.save(
                        session, config.data_config["ckpt_path"] + 'rnn.weights')
                    best_train_loss = train_loss
                    best_train_epoch = epoch
                print('best_train_loss: {}'.format(best_train_loss))
                print('best_train_epoch: {}'.format(best_train_epoch))
                print('best_train_accuracy: {}'.format(best_train_accuracy))
                # context.train_info.append("best_train_loss:" + str(best_train_loss))
                # context.train_info.append("best_train_epoch:" + str(best_train_epoch))
                # context.train_info.append("best_train_accuracy:" + str(best_train_accuracy))
                # anneal
                if train_loss > prev_epoch_loss * config.anneal_threshold:
                    config.lr /= config.anneal_by
                    print('annealed lr to %f' % config.lr)

                prev_epoch_loss = train_loss

                # if epoch - best_val_epoch > config.early_stopping:
                #     break
                print('Total time: {}'.format(time.time() - start))
        print('==> training stop')
        # context.train_info.append("training stop")


def replace_model_path(checkpoint_path):
    pat = re.compile(r'\"(/.+/)model-\d+\"$')
    model = ''.join(pat.findall(checkpoint_path))
    text = re.sub(model, '', checkpoint_path)
    return text


def prepare_data(data_config):
    train_file_path = os.path.join(data_config['data_dir'], 'train.csv')
    num_train_lines = sum(1 for _ in open(train_file_path))
    max_scan = data_config['max_scan_lines']
    partitions = int(num_train_lines / max_scan) + 1
    print('num of train lines is {}, {} partitions will be produced..'.format(num_train_lines, partitions))
    info = dict()
    for i in range(partitions):
        metadata, data = data_helper.load_raw_data(data_config, i, partitions)

        with open(data_config["metadata_path"] + 'metadata_partition_' + str(i) + ".pkl", 'wb') as f:
            pickle.dump(metadata, f)
        with open(data_config["data_path"] + 'data_partition_' + str(i) + ".pkl", 'wb') as f:
            pickle.dump(data, f)

        info["candidate_size"] = metadata['candidate_size']
        info['candid2idx'] = metadata['candid2idx']
        info['candid2idx'] = metadata['idx2candid']
    info["partitions"] = partitions
    with open(data_config["metadata_path"] + 'info.pkl', 'wb') as f:
        pickle.dump(info, f)


def load_data(data_config, partition):
    """Loads metadata and data"""
    with open(data_config["metadata_path"] + 'metadata_partition_' + str(partition) + ".pkl", 'rb') as f:
        metadata = pickle.load(f)

    with open(data_config["data_path"] + 'data_partition_' + str(partition) + ".pkl", 'rb') as f:
        data = pickle.load(f)
    train = data['train']
    valid = data['valid']

    # embedding
    print('pre embedding')
    # context.train_info.append('pre embedding')
    sentences_embedding = data_helper.sentence_embedding(metadata)
    print('pre embedding done')
    # context.train_info.append('pre embedding done')
    # metadata['max_q_len'] = max_len
    # metadata['max_sen_len'] = max_len
    metadata['sentences_embedding'] = sentences_embedding

    return train, valid, metadata


def _check_restore_parameters(sess, saver, model_path):
    """ Restore the previously trained parameters if there are any. """
    print("--checking directory:", model_path)
    ckpt = tf.train.get_checkpoint_state(model_path)
    if ckpt and ckpt.model_checkpoint_path:
        print("Loading parameters for the model")
        saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        print("Initializing fresh parameters for the model")


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
        train(config, args['restore'])
    elif args['inference']:
        inference(config)
    else:
        print('ERROR:Unknow Mode')


if __name__ == '__main__':
    main(sys.argv[1:])
