from ..utils import model_utils
from .gate_encoder import Encoder
from .char_cnn import char_cnn
import tensorflow as tf

class Discriminator(object):
    """ Discriminator class """
    def __init__(self, hyp_args):
        self.num_layers = hyp_args['D_num_layers']
        self.num_heads = hyp_args['D_num_heads']
        self.hidden_dim = hyp_args['D_hidden_dim']
        self.linear_key_dim = hyp_args['D_linear_key_dim']
        self.linear_value_dim = hyp_args['D_linear_value_dim']
        self.ffn_dim = hyp_args['D_ffn_dim']
        self.dropout = hyp_args['D_dropout']
        self.activation = hyp_args['D_activation']
        self.layer_norm = model_utils.LayerNormalization(self.hidden_dim)
        self.max_word_len = hyp_args["max_word_len"]
        self.max_char_len = hyp_args["max_char_len"]
        self.char_vocab_size = hyp_args['char_vocab_size']
        self.char_emb_dim = hyp_args['char_emb_dim']
        self.kernel_width = hyp_args['kernel_width']
        self.kernel_depth = hyp_args['kernel_depth']
        self.highway_layers = hyp_args['highway_layers']
        if self.activation == "gelu":
            self.activation = model_utils.gelu

    def build_embed(self, inputs, word_seq_len, isTrain):
        """
        :param inputs: (batch_size, max_len)
        :param isTrain: boolean (True/False)
        :return: (batch_size, max_len, emb_dim)
        """
        max_seq_length = tf.shape(inputs)[1]
        # Positional Encoding
        with tf.variable_scope("Electra/Positional-encoding", reuse=tf.AUTO_REUSE):
            position_emb = model_utils.get_position_encoding(max_seq_length, self.hidden_dim)
        # Word Embedding
        with tf.variable_scope("Electra/Char-Embeddings", reuse=tf.AUTO_REUSE):
            pad_table = tf.zeros(shape=(1, self.char_emb_dim))
            char_emb_table = tf.get_variable('Char_Weights', [self.char_vocab_size - 1, self.char_emb_dim],
                                                   initializer=tf.random_uniform_initializer(-1.0, 1.0))
            self.char_emb_table = tf.concat([pad_table, char_emb_table], axis=0)
            mask = tf.sequence_mask(lengths=word_seq_len, maxlen=self.max_word_len, dtype=tf.float32)
            char_emb = tf.nn.embedding_lookup(self.char_emb_table, inputs)  ## batch_size, word_len, char_len, dim
            word_emb = char_cnn(char_emb, self.kernel_width, self.kernel_depth, self.hidden_dim, self.highway_layers,
                                self.max_word_len, self.max_char_len, self.char_emb_dim)
            word_emb *= tf.expand_dims(mask, -1)  ## zeros out masked positions
            word_emb *= self.hidden_dim ** 0.5  ## Scale embedding by the sqrt of the hidden size
        ## Add Word emb & Positional emb
        encoded_inputs = tf.add(word_emb, position_emb)
        if isTrain:
            encoded_inputs = tf.nn.dropout(encoded_inputs, 1.0 - self.dropout) ## embedding with position info
            word_emb = tf.nn.dropout(word_emb, 1.0 - self.dropout)  ## embedding without position info
        return encoded_inputs, word_emb

    def build_encoder(self, enc_input_char_idx, word_seq_len, isTrain):
        ## enc_input_char_idx : (batch_size, word_len, char_len)
        """
        :param enc_input_idx: (batch_size, enc_len)
        :param isTrain: boolean (True/False)
        :return: (batch_size, enc_len, hidden_dim)
        """
        padding_bias = model_utils.get_padding_bias_with_seqlen(word_seq_len, self.max_word_len)
        encoder_emb_inp, char_cnn_output = self.build_embed(enc_input_char_idx, word_seq_len, isTrain) # batch_size, word_len, dim
        with tf.variable_scope("Discriminator", reuse=tf.AUTO_REUSE):
            encoder = Encoder(num_layers=self.num_layers,
                              num_heads=self.num_heads,
                              linear_key_dim=self.linear_key_dim,
                              linear_value_dim=self.linear_value_dim,
                              model_dim=self.hidden_dim,
                              ffn_dim=self.ffn_dim,
                              dropout=self.dropout,
                              activation=self.activation,
                              isTrain=isTrain)

            encoder_outputs = encoder.build(encoder_emb_inp, char_cnn_output, padding_bias) # batch_size, word_len, dim
            self.hidden_states = encoder.hidden_states
            self.char_states = encoder.char_states
            self.alpha_states = encoder.alpha_states
            return encoder_outputs

    def build_logits(self, encoder_outputs):
        with tf.variable_scope("Discriminator/Transform_layer", reuse=tf.AUTO_REUSE):
            transformed_output = tf.layers.dense(encoder_outputs, self.hidden_dim, activation=self.activation,
                                                kernel_initializer=tf.random_normal_initializer(0., self.hidden_dim ** -0.5))
            transformed_output = self.layer_norm(transformed_output)
        with tf.variable_scope("Discriminator/Output_layer", reuse=tf.AUTO_REUSE):
            logits = tf.squeeze(tf.layers.dense(transformed_output, 1), -1)
        return logits

    def build_graph(self, input_char_idx, word_seq_len):
        ## Encoder
        encoder_outputs = self.build_encoder(input_char_idx, word_seq_len, isTrain=True)
        ## Logits
        logits = self.build_logits(encoder_outputs)
        return logits

    def build_loss(self, logits, labels, seq_len):
        """
        :param logits : (batch_size, max_len)
        :param labels : (batch_size, max_len)
        :param seq_len : (batch_size,)
        """
        labels = tf.cast(labels, tf.float32)
        loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels)
        weight_label = tf.cast(tf.sequence_mask(seq_len, maxlen=tf.shape(labels)[1]), tf.float32)
        # sequence mask for padding
        loss = tf.reduce_sum(loss * weight_label) / (tf.reduce_sum(weight_label) + 1e-10)
        return loss

    def build_accuracy(self, infer, labels, seq_len):
        """
        :param infer : (batch_size, max_len)
        :param labels : (batch_size, max_len)
        :param seq_len : (batch_size,)
        """
        D_equal = tf.cast(tf.equal(infer, labels), tf.float32)
        weight_label = tf.cast(tf.sequence_mask(seq_len, maxlen=tf.shape(labels)[1]), tf.float32)
        # sequence mask for padding
        accuracy = tf.reduce_sum(D_equal * weight_label) / (tf.reduce_sum(weight_label) + 1e-10)
        return accuracy
