"""
-------------------------------------------------
   File Name：     config
   Description :
   Author :       deep
   date：          18-1-30
-------------------------------------------------
   Change Activity:
                   18-1-30:
                   
   __author__ = 'deep'
-------------------------------------------------
"""

import sys
import os

parentdir = os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))
sys.path.insert(0, parentdir)
sys.path.append(parentdir)

grandfatherdir = os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))

sys.path.insert(0, grandfatherdir)
sys.path.append(grandfatherdir)

# from py.singleton import Singleton

class RNNConfig(object):

    def __init__(self, config_dict=None):
        if not config_dict:
            self.hidden_neural_size = 300
            self.embed_dim = 300
            self.hidden_layer_num = 2
            self.keep_prob = 1
            self.lr_decay = 1
            self.batch_size = 64
            self.max_epochs = 3456
            self.interval_epochs = 5
            self.dropout = 1
            self.lr = 0.001
            self.l2 = 0.001
            self.top_k=6

            self.cap_grads = True
            self.max_grad_val = 10
            self.noisy_grads = True

            # NOTE not currently used hence non-sensical anneal_threshold
            self.anneal_threshold = 1000
            self.anneal_by = 1

            prefix = grandfatherdir = os.path.dirname(os.path.dirname(
                os.path.dirname(os.path.abspath(__file__))))

            self.data_config = dict()
            self.data_config["data_dir"] = '/opt/luis/kaggle/toxic/data/'
            self.data_config["candid_path"] = os.path.join(
                prefix, 'data/rnn/train/automata/candidates.txt')

            self.data_config["metadata_path"] = '/opt/luis/kaggle/toxic/model/rnn/rnn_preprocessed/metadata.pkl'
            self.data_config["data_path"] = '/opt/luis/kaggle/toxic/model/rnn/rnn_preprocessed/data.pkl'
            self.data_config["ckpt_path"] = '/opt/luis/kaggle/toxic/model/rnn/ckpt/'
        else:
            self.hidden_neural_size = config_dict["hidden_neural_size"]
            self.embed_dim = config_dict["embed_dim"]
            self.hidden_layer_num = config_dict["hidden_layer_num"]
            self.keep_prob = config_dict["keep_prob"]
            self.lr_decay = config_dict["lr_decay"]
            self.batch_size = config_dict["batch_size"]
            self.max_epochs = config_dict["max_epochs"]
            self.interval_epochs = config_dict["interval_epochs"]
            self.dropout = config_dict["dropout"]
            self.lr = config_dict["lr"]
            self.l2 = config_dict["l2"]
            self.top_k = int(config_dict["top_k"])

            self.cap_grads = config_dict["cap_grads"]
            self.max_grad_val = config_dict["max_grad_val"]
            self.noisy_grads = config_dict["noisy_grads"]

            # NOTE not currently used hence non-sensical anneal_threshold
            self.anneal_threshold = config_dict["anneal_threshold"]
            self.anneal_by = config_dict["anneal_by"]

            # prefix = grandfatherdir = os.path.dirname(os.path.dirname(
            #     os.path.dirname(os.path.abspath(__file__))))

            self.data_config = dict()
            # self.data_config["data_dir"] = config_dict["data_dir"]
            # self.data_config["candid_path"] = config_dict["candid_path"]

            self.data_config["metadata_path"] = config_dict["data_metadata_path"]
            # self.data_config["data_path"] = config_dict["data_path"]
            self.data_config["ckpt_path"] = config_dict["data_ckpt_path"]
