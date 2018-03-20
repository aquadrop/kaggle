import sys
import os
import pickle
import tensorflow as tf

parentdir = os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))
sys.path.insert(0, parentdir)

grandfatherdir = os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))))

from rnn.config import RNNConfig as Config
from rnn.model import RNN_Model
from py.singleton import Singleton
import utils.embedding.vector_helper as vector_helper
from utils.embedding.vector_helper import BenebotVector as VectorHelper
from utils.query_util import tokenize


class RnnSession(metaclass=Singleton):
    def __init__(self, session, model, config, metadata, char=2):
        self.session = session
        self.model = model
        self.config = config
        # self.max_q_len = metadata['max_q_len']
        # self.idx2candid = metadata['idx2candid']
        self.char = char
        self.vh = VectorHelper()

    def reply(self, query):
        q = tokenize(query, self.char)
        q_len = len(q)
        # q = q[:self.max_q_len]
        # q_vector = q + \
        #            [vector_helper.PAD for _ in range(
        #                    self.max_q_len - len(q))]
        q_vector = [self.vh.getVector(word) for word in q]

        pred, top_prob = self.model.predict(self.session, [q_vector], [q_len], self.config.dropout)

        # reply_msg = [self.idx2candid[ind] for ind in pred]

        return '', top_prob


class RnnInfer(metaclass=Singleton):
    def __init__(self, rnn_config=None):
        self.config = Config(rnn_config)
        with open(self.config.data_config['metadata_path'] + 'info.pkl', 'rb') as f:
            self.info = pickle.load(f)
        self.model = self._load_model()

    def _load_model(self):
        self.session = tf.Session()
        model = RNN_Model(self.config, self.info, is_training=False)
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()

        self.session.run(init)

        # restore checkpoint
        ckpt_path = os.path.join(self.config.data_config['ckpt_path'], 'rnn.weights')
        # print(ckpt_path)
        # ckpt = tf.train.get_checkpoint_state(ckpt_path)
        # if ckpt and ckpt.model_checkpoint_path:
        # print('\n>> restoring checkpoint from',
        #       ckpt.model_checkpoint_path)
        saver.restore(self.session, ckpt_path)
        return model

    def get_session(self):
        char = 8
        isess = RnnSession(self.session, self.model,
                           self.config, self.info, char)
        return isess


def main():
    ri = RnnInfer()
    sess = ri.get_session()

    query = ''
    while query != 'exit':
        query = input('>> ')
        print(sess.reply(query))


if __name__ == '__main__':
    main()

Hey... what is it..@ | talk .What is it... an exclusive group of some WP TALIBANS...who are good at destroying, self-appointed purist who GANG UP any one who asks them questions abt their ANTI-SOCIAL and DESTRUCTIVE (non)-contribution at WP?Ask Sityush to clean up his behavior than issue me nonsensical warnings...
