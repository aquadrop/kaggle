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
import os

import inspect
import tensorflow as tf

import numpy as np
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

def _add_gradient_noise(t, stddev=1e-3, name=None):
    """Adds gradient noise as described in http://arxiv.org/abs/1511.06807
    The input Tensor `t` should be a gradient.
    The output will be `t` + gaussian noise.
    0.001 was said to be a good fixed value for memory networks."""
    # with tf.op_scope([t, stddev], name, "add_gradient_noise") as name:
    with tf.name_scope(name, "add_gradient_noise", [t, stddev]) as name:
        t = tf.convert_to_tensor(t, name="t")
        gn = tf.random_normal(tf.shape(t), stddev=stddev)
        return tf.add(t, gn, name=name)

class RNN_Model(object):

    def __init__(self, config, metadata, is_training=True):

        self.keep_prob = config.keep_prob
        self.batch_size = config.batch_size

        self.max_sequence_len = metadata["max_q_len"]
        self.embed_dim = config.embed_dim


        self.hidden_neural_size=config.hidden_neural_size
        self.hidden_layer_num=config.hidden_layer_num

        self.num_classes = metadata["candidate_size"]
        self.is_training = is_training
        self.max_grad_norm = 10
        self.config = config
        self.initializer=tf.random_normal_initializer(stddev=0.1)
        # self.max_grad_norm = config.max_grad_norm

        if is_training:
            self._create_placeholder()
            self._create_inference()
            self._create_loss()
            self._create_train_op()
            self._create_measure()
        else:
            self._create_placeholder()
            self._create_inference()


    def _create_placeholder(self):
        # self.embedding_placeholder = tf.placeholder(tf.float32,
        #                                             [None, self.max_sequence_len, self.embed_dim],
        #                                             name="embedding")
        # self.target_placeholder = tf.placeholder(tf.int64,
        #                                          (None,), name='target')
        self.target_placeholder = tf.placeholder(tf.float32, [None, self.num_classes], name="input_y")
        # self.mask_placeholder = tf.placeholder(tf.float32, [self.max_sequence_len, None], name="mask")

        self.question_placeholder = tf.placeholder(
            tf.float32, shape=(None, self.max_sequence_len, self.embed_dim), name='questions')
        # self.question_placeholder = tf.placeholder(
        #         tf.float32, shape=(None, None, None), name='questions')
        self.question_len_placeholder = tf.placeholder(
            tf.int32, shape=(None,), name='question_lens')

        self.dropout_placeholder = tf.placeholder(tf.float32, name='dropout')

    # build LSTM network
    def lstm_cell(self):
        if 'reuse' in inspect.signature(tf.contrib.rnn.BasicLSTMCell.__init__).parameters:
            return tf.contrib.rnn.BasicLSTMCell(self.hidden_neural_size, forget_bias=0.0,
                                                state_is_tuple=True,
                                                reuse=tf.get_variable_scope().reuse)
        else:
            return tf.contrib.rnn.BasicLSTMCell(
                self.hidden_neural_size, forget_bias=0.0, state_is_tuple=True)

    def _create_inference(self):
        # attn_cell = self.lstm_cell()

        # cell = tf.contrib.rnn.MultiRNNCell(
        #     [attn_cell() for _ in range(self.hidden_layer_num)], state_is_tuple=True)
        cell = tf.contrib.rnn.GRUCell(self.hidden_neural_size)

        reuse = False #if hop_index > 0 else False

        with tf.variable_scope('gru', reuse=reuse):
            outputs, q_vec = tf.nn.dynamic_rnn(cell,
                                         self.question_placeholder,
                                         dtype=np.float32,
                                         sequence_length=self.question_len_placeholder
                                         )
        with tf.name_scope("softmax_layer_and_output"):
            self.softmax_w = tf.get_variable("softmax_w", [self.hidden_neural_size, self.num_classes], dtype=tf.float32, initializer=self.initializer)
            self.softmax_b = tf.get_variable("softmax_b", [self.num_classes], dtype=tf.float32)
            self.output_rnn_last = tf.reduce_mean(outputs, axis=1)
            # self.logits = tf.matmul(self.output_rnn_last,
            #                    self.softmax_w) + self.softmax_b
            # self.logits = tf.matmul(out_put,softmax_w)
            # self.logits = tf.add(self.logits, softmax_b, name='scores')
            self.logits = tf.nn.xw_plus_b(self.output_rnn_last, self.softmax_w, self.softmax_b, name="scores")
            # rnn_output = tf.nn.dropout(q_vec, self.dropout_placeholder)
            #
            # o = tf.layers.dense(rnn_output,
            #                          self.num_classes,
            #                          activation=None)
            # self.logits = o

            self.predict_proba_op = tf.nn.softmax(self.logits,name='predict_proba_op')
            self.predict_proba_top_op = tf.nn.top_k(
                    self.predict_proba_op, k=self.config.top_k, name='top_predict_proba_op')
            self.pred = tf.argmax(self.predict_proba_op, 1, name='pred')


    def _create_loss(self):
        gate_loss = 0
        with tf.name_scope("loss"):
            # self.loss = tf.nn.softmax_cross_entropy_with_logits(
            #     labels=self.target_placeholder, logits=self.logits + 1e-10)
            # self.loss = tf.reduce_mean(self.loss)
            self.loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.target_placeholder, logits=self.logits)
            self.loss = tf.reduce_sum(self.loss) + gate_loss

    def _create_train_op(self):

        """Calculate and apply gradients"""
        opt = tf.train.AdamOptimizer(
            learning_rate=self.config.lr, epsilon=1e-8)
        gvs = opt.compute_gradients(self.loss)

        # optionally cap and noise gradients to regularize
        if self.config.cap_grads:
            gvs = [(tf.clip_by_norm(grad, self.config.max_grad_val), var)
                   for grad, var in gvs]
        if self.config.noisy_grads:
            gvs = [(_add_gradient_noise(grad), var) for grad, var in gvs]

        self.train_op = opt.apply_gradients(gvs)

    def _create_measure(self):
        with tf.name_scope("accuracy"):
            self.pred = tf.argmax(tf.nn.softmax(self.logits), 1, name="prediction")
            correct_prediction = tf.equal(self.pred, tf.argmax(self.target_placeholder, 1))
            self.correct_num = tf.reduce_sum(tf.cast(correct_prediction, tf.float32))
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name="accuracy")
            self.probability = tf.nn.softmax(self.logits, name="probability")

    def predict(self,sess,questions,q_lens,dropout):
        feed = {
            self.question_placeholder    : questions,
            self.question_len_placeholder: q_lens,
            self.dropout_placeholder     : dropout
        }
        pred, prob_top = sess.run(
                [self.pred, self.predict_proba_top_op], feed_dict=feed)
        return pred, prob_top

        # probs, state = sess.run([self.probs, self.final_state], feed_dict=feed)

        # results = np.argmax(probs, 1)
        # id2labels = dict(zip(labels.values(), labels.keys()))
        # labels = map(id2labels.get, results)
        # return labels