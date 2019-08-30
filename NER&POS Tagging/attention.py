# -*- coding: utf-8 -*-
"""
Created on Sun Jan 20 18:53:17 2019

@author: jbk48
"""


import numpy as np
import tensorflow as tf
from keras.layers.pooling import GlobalAveragePooling2D, GlobalMaxPooling2D


__all__ = [
    "positional_encoding", "Attention"
]


def positional_encoding(dim, sentence_length, dtype=tf.float32):
    pos_enc = np.array([[pos / np.power(10000., 2. * (i // 2) / dim) for i in range(dim)] for pos in range(sentence_length)]) # [seq_len, d_model]    
    # Apply the cosine to even columns and sin to odds.
    pos_enc[:, 0::2] = np.sin(pos_enc[:, 0::2])  # dim 2i
    pos_enc[:, 1::2] = np.cos(pos_enc[:, 1::2])  # dim 2i+1
    return tf.convert_to_tensor(pos_enc, dtype=dtype)


class Attention:

    def __init__(self,
                 num_heads=1,
                 masked=False,
                 linear_key_dim=50,
                 linear_value_dim=50,
                 model_dim=100,
                 dropout=0.2,
                 batch_size=128):

        assert linear_key_dim % num_heads == 0
        assert linear_value_dim % num_heads == 0

        self.num_heads = num_heads
        self.masked = masked
        self.linear_key_dim = linear_key_dim
        self.linear_value_dim = linear_value_dim
        self.model_dim = model_dim
        self.dropout = dropout
        self.batch_size = batch_size

    def multi_head(self, q, k, v, seq_len):
        q, k, v = self._linear_projection(q, k, v)
        qs, ks, vs = self._split_heads(q, k, v)
        outputs = self._scaled_dot_product(qs, ks, vs, seq_len)
        output = self._concat_heads(outputs)
        output = tf.layers.dense(output, self.model_dim)

        return tf.nn.dropout(output, 1.0 - self.dropout)

    def classifier_head(self, q, k, v, seq_len, n_class):
        q, k, v = self._linear_projection(q, k, v)
        qs, ks, vs = self._split_heads(q, k, v)
        outputs = self._scaled_dot_product(qs, ks, vs, seq_len)
        output = self._SelfAttention_pooling(outputs, n_class)
        
        return output
    
    def _SelfAttention_pooling(self, outputs, n_class): # batch_size, num_heads, max_seq_len, dim
        batch_size, num_heads, max_seq_len, dim = outputs.get_shape().as_list()
        W = tf.layers.dense(outputs, 1, use_bias=False) # batch_size, num_heads, max_seq_len, 1
        softmax_W = tf.nn.softmax(W, axis=2)  # batch_size, num_heads, max_seq_len, 1
        outputs = tf.multiply(softmax_W, outputs) # batch_size, num_heads, max_seq_len, dim
        outputs = tf.reduce_sum(outputs, axis=2) # batch_size, num_heads, dim
        outputs = tf.reshape(outputs, shape = [-1, num_heads*dim]) # batch_size, num_heads*dim
        outputs = tf.layers.dense(outputs, n_class) # batch_size, n_class
        return outputs

    def _seq_len_pooling(self, outputs, seq_len, n_class): # batch_size, max_seq_len, dim
        row_vector = tf.range(0,outputs.shape[1],1)  ## [, max_seq_len]
        matrix = tf.cast(tf.expand_dims(seq_len,-1), tf.int32) ## [batch_size, 1]        
        t = tf.cast(row_vector < matrix, tf.float32) ##  [batch_size, max_seq_len]
        mask = tf.tile(tf.expand_dims(t, axis=-1), [1, 1, int(outputs.shape[2])]) # batch_size, max_seq_len, dim
        outputs = tf.multiply(outputs, mask) # batch_size, max_seq_len, dim
        outputs = tf.reduce_sum(outputs, axis=1) # batch_size, dim
        div = tf.expand_dims(tf.reduce_sum(t, axis=-1), axis=-1) ##  batch_size, 1
        outputs = tf.divide(outputs, div) # batch_size, dim
        outputs = tf.layers.dense(outputs, n_class) # batch_size, n_class
        return outputs
    
    def _GlobalAverage_heads(self, outputs):
        outputs = tf.transpose(outputs, [0, 3, 2, 1]) # [batch_size, dim, max_seq_len, num_heads]
        outputs = GlobalAveragePooling2D()(outputs)
        return outputs

    def _GlobalMax_heads(self, outputs):
        outputs = tf.transpose(outputs, [0, 3, 2, 1]) # [batch_size, dim, max_seq_len, num_heads]
        outputs = GlobalMaxPooling2D()(outputs)
        return outputs

    def _linear_projection(self, q, k, v):
        q = tf.layers.dense(q, self.linear_key_dim, use_bias=False)
        k = tf.layers.dense(k, self.linear_key_dim, use_bias=False)
        v = tf.layers.dense(v, self.linear_value_dim, use_bias=False)
        return q, k, v

    def _split_heads(self, q, k, v):

        def split_last_dimension_then_transpose(tensor, num_heads, dim): ## dim = num_head * project_dim
            t_shape = tensor.get_shape().as_list()
            tensor = tf.reshape(tensor, [-1] + t_shape[1:-1] + [num_heads, dim // num_heads])
            return tf.transpose(tensor, [0, 2, 1, 3]) # [batch_size, num_heads, max_seq_len, dim]

        qs = split_last_dimension_then_transpose(q, self.num_heads, self.linear_key_dim)
        ks = split_last_dimension_then_transpose(k, self.num_heads, self.linear_key_dim)
        vs = split_last_dimension_then_transpose(v, self.num_heads, self.linear_value_dim)

        return qs, ks, vs

    def _scaled_dot_product(self, qs, ks, vs, seq_len):
        ## qs, ks, vs : [batch_size, num_heads, max_seq_len, dim]
        key_dim_per_head = self.linear_key_dim // self.num_heads
        o1 = tf.matmul(qs, ks, transpose_b=True)
        o2 = o1 / (key_dim_per_head**0.5) ## [batch_size, num_heads, max_seq_len, max_seq_len]
        
        if self.masked: ## mask score matrix to max_seq_len
            row_vector = tf.range(0,o2.shape[2],1)  ## [, max_seq_len]
            matrix = tf.cast(tf.expand_dims(seq_len,-1), tf.int32) ## [batch_size, 1]
            
            t = tf.cast(row_vector < matrix, tf.float32) ##  [batch_size, max_seq_len]
            t = tf.expand_dims(t, -1) ##  [batch_size, max_seq_len, 1]
            masks = t * tf.transpose(t, [0,2,1]) ##  [batch_size, max_seq_len, max_seq_len]
            masks = tf.linalg.set_diag(masks, tf.zeros_like(masks[:,:,0])) ## setting diag value to 0
            masks = tf.tile(tf.expand_dims(masks, 1), [1, int(o2.shape[1]), 1, 1]) ##  [batch_size, num_heads, max_seq_len, max_seq_len]            
            paddings = tf.ones_like(masks) * -1e9        
            o2 = tf.where(tf.equal(masks, 0), paddings, o2)
            o2 = tf.nn.softmax(o2) 
            weights = tf.where(tf.equal(masks, 0), tf.zeros_like(masks), o2)
            weights = tf.nn.dropout(weights, 1.0 - self.dropout)
            attention_output = tf.matmul(weights, vs)            
        return attention_output

    def _concat_heads(self, outputs):

        def transpose_then_concat_last_two_dimenstion(tensor):
            tensor = tf.transpose(tensor, [0, 2, 1, 3]) # [batch_size, max_seq_len, num_heads, dim]
            t_shape = tensor.get_shape().as_list()
            num_heads, dim = t_shape[-2:]
            return tf.reshape(tensor, [-1] + t_shape[1:-2] + [num_heads * dim]) # [batch_size, max_seq_len, num_heads*dim]

        return transpose_then_concat_last_two_dimenstion(outputs)