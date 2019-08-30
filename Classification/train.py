# -*- coding: utf-8 -*-
"""
Created on Fri May  3 14:27:21 2019

@author: jbk48
"""

import model
import tensorflow as tf

if __name__ == '__main__':

    flags = tf.app.flags
    FLAGS = flags.FLAGS
    
    ## Model parameter
    flags.DEFINE_integer('char_dim', 16, 'dimension of character vector')
    flags.DEFINE_integer('max_word_len', 100, 'max length of words of sentences')
    flags.DEFINE_integer('max_char_len', 50, 'max length of characters of words')
    flags.DEFINE_float('learning_rate', 0.0001, 'initial learning rate')
    flags.DEFINE_integer('batch_size', 64, 'number of batch size')
    flags.DEFINE_integer('training_epochs', 12, 'number of training epochs')
    ## Transformer-Encoder parameter
    flags.DEFINE_integer('num_layers', 6, 'number of layers of transformer encoders')
    flags.DEFINE_integer('num_heads', 8, 'number of heads of transformer encoders')
    flags.DEFINE_integer('linear_key_dim', 512, 'dimension of')
    flags.DEFINE_integer('linear_value_dim', 512, 'dimension of')
    flags.DEFINE_integer('model_dim', 512, 'output dimension of transformer encoder')
    flags.DEFINE_integer('ffn_dim', 2048, 'dimension of feed forward network')
    flags.DEFINE_integer('n_class', 4, 'number of output class')
    flags.DEFINE_string('char_mode', 'char_cnn', 'mode of character embedding')
    
    print('========================')
    for key in FLAGS.__flags.keys():
        print('{} : {}'.format(key, getattr(FLAGS, key)))
    print('========================')
    ## Build model
    t_model = model.Model(FLAGS.char_dim, FLAGS.max_word_len, FLAGS.max_char_len, FLAGS.learning_rate)
    t_model.build_parameter(FLAGS.num_layers, FLAGS.num_heads, FLAGS.linear_key_dim, FLAGS.linear_value_dim,
                            FLAGS.model_dim, FLAGS.ffn_dim, FLAGS.n_class)
    
    ## Train model
    t_model.train(FLAGS.batch_size, FLAGS.training_epochs, FLAGS.char_mode)
    
