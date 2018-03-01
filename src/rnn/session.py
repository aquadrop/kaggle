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
from utils.query_util import tokenize


class RnnSession(metaclass=Singleton):
    def __init__(self, session, model, config, metadata, char=2):
        self.session = session
        self.model = model
        self.config = config
        self.max_q_len = metadata['max_q_len']
        self.idx2candid = metadata['idx2candid']
        self.char = char

    def reply(self, query):
        q = tokenize(query, self.char)
        q_len = len(q)
        q = q[:self.max_q_len]
        q_vector = q + \
                   [vector_helper.PAD for _ in range(
                           self.max_q_len - len(q))]
        q_vector = [vector_helper.getVector(word) for word in q_vector]

        pred, top_prob = self.model.predict(self.session, [q_vector], [q_len], self.config.dropout)

        reply_msg = [self.idx2candid[ind] for ind in pred]

        return reply_msg[0], top_prob[0][0][0]


class RnnInfer(metaclass=Singleton):
    def __init__(self, rnn_config=None):
        self.config = Config(rnn_config)
        with open(self.config.data_config['metadata_path'], 'rb') as f:
            self.metadata = pickle.load(f)
        self.model = self._load_model()

    def _load_model(self):
        self.session = tf.Session()
        model = RNN_Model(self.config, self.metadata, is_training=False)
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
        char = 2
        isess = RnnSession(self.session, self.model,
                           self.config, self.metadata, char)
        return isess


class RnnSession2(object):
    def __init__(self, rnn_config=None, char=2):
        self.config = Config(rnn_config)
        self.dropout=self.config.dropout
        with open(self.config.data_config['metadata_path'], 'rb') as f:
            metadata = pickle.load(f)
        self.max_q_len = metadata['max_q_len']
        self.idx2candid = metadata['idx2candid']
        self.char = char
        self._init_session()

    def _init_session(self):
        graph = tf.Graph()
        with graph.as_default():
            self.session = tf.Session()
            ckpt_dir=self.config.data_config["ckpt_path"]
            ckpt_path=os.path.join(ckpt_dir,'rnn.weights')
            ckpt_meta_path=os.path.join(ckpt_dir,'rnn.weights.meta')
            saver = tf.train.import_meta_graph(ckpt_meta_path)
            saver.restore(self.session, ckpt_path)

            self.questions=graph.get_operation_by_name("questions").outputs[0]
            self.questions_len=graph.get_operation_by_name("question_lens").outputs[0]
            self.dropout_prop=graph.get_operation_by_name("dropout").outputs[0]

            self.top_prob=graph.get_operation_by_name("softmax_layer_and_output/top_predict_proba_op").outputs[0]
            self.pred=graph.get_operation_by_name("softmax_layer_and_output/pred").outputs[0]

            # self.prediction = graph.get_operation_by_name("accuracy/prediction").outputs[0]
            # self.scores = graph.get_operation_by_name("Softmax_layer_and_output/scores").outputs[0]
            # self.probability = graph.get_operation_by_name("accuracy/probability").outputs[0]

    def reply(self, query):
        q = tokenize(query, self.char)
        q_len = len(q)
        q = q[:self.max_q_len]
        # print('q:',q)
        q_vector = q + \
                   [vector_helper.PAD for _ in range(
                           self.max_q_len - len(q))]
        q_vector = [vector_helper.getVector(word) for word in q_vector]
        # print('q_vec:',q_vector[0])


        feed_dict=dict()
        feed_dict[self.questions]=[q_vector]
        feed_dict[self.questions_len]=[q_len]
        feed_dict[self.dropout_prop]=self.dropout

        fetch=[self.pred,self.top_prob]

        [pred,top_prob]=self.session.run(fetch,feed_dict)
        reply_msg = [self.idx2candid[ind] for ind in pred]

        return reply_msg, top_prob


def main():
    # ri = RnnInfer()
    sess = RnnSession2()

    query = ''
    while query != 'exit':
        query = input('>> ')
        print(sess.reply(query))


if __name__ == '__main__':
    main()
