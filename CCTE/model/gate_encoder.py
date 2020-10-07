import tensorflow as tf
from .attention import Attention
from .layer import FFN
from ..utils.model_utils import LayerNormalization


class Encoder:
    """ Encoder class """
    def __init__(self,
                 num_layers=6,
                 num_heads=8,
                 linear_key_dim=512,
                 linear_value_dim=512,
                 model_dim=512,
                 ffn_dim=2048,
                 dropout=0.1,
                 activation="gelu",
                 isTrain=True):

        self.num_layers = num_layers
        self.num_heads = num_heads
        self.linear_key_dim = linear_key_dim
        self.linear_value_dim = linear_value_dim
        self.model_dim = model_dim
        self.ffn_dim = ffn_dim
        self.dropout = dropout
        self.activation = activation
        self.isTrain = isTrain
        self.layer_norm = LayerNormalization(self.model_dim)

    def build(self, encoder_inputs, char_cnn_output, padding_bias):
        """
        :param encoder_inputs: (batch_size, word_len, hidden_dim)
        :param char_cnn_output: (batch_size, word_len, hidden_dim)
        :param padding_bias: padding mask (batch_size, 1, 1, word_len)
        :return:
        """
        o1 = tf.identity(encoder_inputs)  ## char-cnn output with positional info : (batch_size, enc_len, dim)
        c1 = tf.identity(char_cnn_output) ## char-cnn output without positional info : (batch_size, enc_len, dim)

        self.hidden_states = [encoder_inputs]
        self.char_states = [char_cnn_output]
        self.alpha_states = []

        for i in range(1, self.num_layers + 1):
            with tf.variable_scope("layer-{}".format(i)):
                ## char info ##
                char_info = self._add_and_norm(c1, self._char_encoder(c1), num=0)
                c1 = tf.identity(char_info)
                ## context info ##
                o2 = self._add_and_norm(o1, self._self_attention(q=o1,
                                                                 k=o1,
                                                                 v=o1,
                                                                 bias=padding_bias), num=1)
                context_info = self._add_and_norm(o2, self._positional_feed_forward(o2), num=2)
                ## Weighted sum of context, char info
                o3, alpha = self._context_char(context_info, char_info)
                o1 = tf.identity(o3)

                self.hidden_states.append(o1)
                self.char_states.append(c1)
                self.alpha_states.append(alpha)

        return o3

    def _self_attention(self, q, k, v, bias):
        """
        :param q: query (batch_size, word_len, hidden_dim)
        :param k: key (batch_size, word_len, hidden_dim)
        :param v: value (batch_size, word_len, hidden_dim)
        :param bias: padding mask (batch_size, 1, 1, word_len)
        :return: (batch_size, word_len, hidden_dim)
        """
        with tf.variable_scope("self-attention"):
            attention = Attention(num_heads=self.num_heads,
                                    linear_key_dim=self.linear_key_dim,
                                    linear_value_dim=self.linear_value_dim,
                                    model_dim=self.model_dim,
                                    dropout=self.dropout)
            return attention.multi_head(q, k, v, bias, self.isTrain)

    def _add_and_norm(self, x, sub_layer_x, num=0):
        """
        :param x: (batch_size, word_len, hidden_dim)
        :param sub_layer_x: (batch_size, word_len, hidden_dim)
        :param num: integer
        :return: (batch_size, word_len, hidden_dim)
        """
        with tf.variable_scope("add-and-norm-{}".format(num)):
            return self.layer_norm(tf.add(x, sub_layer_x)) # with Residual connection

    def _positional_feed_forward(self, output):
        """
        :param output: (batch_size, word_len, hidden_dim)
        :param padding: (batch_size, )
        :return: (batch_size, word_len, hidden_dim)
        """
        with tf.variable_scope("feed-forward"):
            ffn = FFN(w1_dim=self.ffn_dim,
                      w2_dim=self.model_dim,
                      dropout=self.dropout,
                      activation=self.activation)
            return ffn.dense_layer(output, self.isTrain)

    def _char_encoder(self, char_info):
        """
        :param char_info: (batch_size, word_len, hidden_dim)
        :return: (batch_size, word_len, hidden_dim)
        """
        with tf.variable_scope("char-encoder"):
            ffn = FFN(w1_dim=self.ffn_dim,
                      w2_dim=self.model_dim,
                      dropout=self.dropout,
                      activation=self.activation)
        return ffn.dense_layer(char_info, self.isTrain)

    def _context_char(self, context_info, char_info):
        """
        :param context_info: (batch_size, word_len, hidden_dim)
        :param char_info: (batch_size, word_len, hidden_dim)
        :return: (batch_size, word_len, hidden_dim)
        """
        with tf.variable_scope("context-char"):
            total_info = tf.concat([context_info,char_info], axis=2) # [batch_size, max_sent_len, model_dim*2]
            alpha = tf.layers.dense(total_info,                      # [batch_size, max_sent_len, model_dim]
                                    units=self.model_dim,
                                    activation=tf.nn.sigmoid)
            o3 = alpha*char_info + (1-alpha)*context_info
        return o3, alpha