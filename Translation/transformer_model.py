# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 13:57:59 2019

@author: jbk48
"""

import os
import datetime
import tensorflow as tf
import numpy as np
import random
import pandas as pd
import model_utils

from wmt_loader import Data
from encoder import Encoder
from decoder import Decoder
from nltk.translate.bleu_score import corpus_bleu

os.environ['CUDA_VISIBLE_DEVICES'] = "3"
config = tf.ConfigProto()
config.gpu_options.allow_growth = True


class Model(object):
    def __init__(self, hidden_dim=512, num_layers=6, num_heads=8,
                 linear_key_dim=512, linear_value_dim=512, ffn_dim=2048, char_dim=16,
                 max_enc_len=100, max_dec_len=100, max_char_len=50, batch_size=128, warmup_steps=4000):
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.linear_key_dim = linear_key_dim
        self.linear_value_dim = linear_value_dim
        self.ffn_dim = ffn_dim
        self.char_dim = char_dim

        self.max_enc_len = max_enc_len
        self.max_dec_len = max_dec_len
        self.max_char_len = max_char_len

        self.batch_size = batch_size
        self.warmup_steps = warmup_steps
        
        # Placeholder for Encoder
        self.x_in = tf.placeholder(dtype=tf.int32, shape=(None, max_enc_len, max_char_len))
        self.x_len = tf.placeholder(dtype=tf.int32, shape=(None, ))
        # Placeholder for Decoder
        self.y_out = tf.placeholder(dtype=tf.int32, shape=(None, max_dec_len))
        self.y_len = tf.placeholder(dtype=tf.int32, shape=(None, ))
        self.dropout = tf.placeholder(dtype=tf.float32, shape=())
        
        self.data = Data(path='./vi2en', max_enc_len=max_enc_len, max_dec_len=max_dec_len)
        self.bos_idx = self.data.bos_idx ## beginning of sentence
        self.eos_idx = self.data.eos_idx ## end of sentence
        self.vocab = self.data.vocab
        self.char_vocab_size = len(self.data.idx2c)
        
        # Train
        self.train_enc_in, self.train_dec_out, self.train_enc_len, self.train_dec_len = self.data.read_file("train")
        # Val
        self.val_enc_in, self.val_dec_out, self.val_enc_len, self.val_dec_len = self.data.read_file("tst2012")
        # Test
        self.test_enc_in, self.test_dec_out, self.test_enc_len, self.test_dec_len = self.data.read_file("tst2013")
        print(' *---- Dataset Intialized ----\n')
        
        self.train_size = len(self.train_enc_in)
        self.test_size = len(self.test_enc_in)
        
        train_dataset = tf.data.Dataset.from_tensor_slices((self.x_in, self.x_len, self.y_out, self.y_len))
        train_dataset = train_dataset.shuffle(self.train_size)
        train_dataset = train_dataset.batch(self.batch_size)
        train_dataset = train_dataset.repeat()

        test_dataset = tf.data.Dataset.from_tensor_slices((self.x_in, self.x_len, self.y_out, self.y_len))
        test_dataset = test_dataset.batch(self.batch_size)
        test_dataset = test_dataset.repeat()
        
        iters = tf.data.Iterator.from_structure(train_dataset.output_types, train_dataset.output_shapes)
        self.iter_x_in, self.iter_x_len, self.iter_y_out, self.iter_y_len = iters.get_next()

        # create the initialisation operations
        self.train_init_op = iters.make_initializer(train_dataset)
        self.test_init_op = iters.make_initializer(test_dataset)
        print("building model")
        ## Output
        encoder_outputs, decoder_inputs, train_prob = self.decoder_train(self.iter_x_in, self.iter_x_len, self.iter_y_out)
        self.build_loss(train_prob)
        self.global_step = tf.train.get_or_create_global_step()
        self.build_opt(self.global_step)
        
        self.pred_token = self.decoder_infer(self.iter_x_in, self.iter_x_len)

        print("done")
        
    def train(self, training_epochs):
        
        num_train_batch = int(self.train_size / self.batch_size) + 1
        num_test_batch = int(self.test_size / self.batch_size) + 1
        
        train_feed_dict = {self.x_in: self.train_enc_in, self.x_len: self.train_enc_len,
                           self.y_out: self.train_dec_out, self.y_len: self.train_dec_len}     
        
        test_feed_dict = {self.x_in: self.test_enc_in, self.x_len: self.test_enc_len,
                          self.y_out: self.test_dec_out, self.y_len: self.test_dec_len}       
        
        print("vocab size : {}".format(self.vocab))
        print("start training")
        modelpath = "./model/"
        modelName = "transformer_de2en.ckpt"
        saver = tf.train.Saver()  
        
        with tf.Session(config=config) as sess:
            
            sess.run(tf.global_variables_initializer())

            if(not os.path.exists(modelpath)):
                os.mkdir(modelpath)
            ckpt = tf.train.get_checkpoint_state(modelpath)
            
            if(ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path)):
                self.load_char_embedding(modelpath + "char_embedding.npy")
                saver.restore(sess, modelpath + modelName)
                print("Model loaded!")

            sess.run(self.train_init_op, feed_dict = train_feed_dict) 
            gs = sess.run(self.global_step)
                        
            start_time = datetime.datetime.now()
            
            train_loss_list = []
            test_loss_list = []
            test_bleu_list = []
            
            for epoch in range(training_epochs):
                
                train_loss = 0
                
                for step in range(num_train_batch):
                    char_embedding_matrix = sess.run(self.clear_char_embedding_padding)                      
                    loss, _ = sess.run([self.loss, self.optimizer],
                                       feed_dict={self.dropout: 0.1})               
                    train_loss += loss/num_train_batch
                    print("epoch {:02d} step {:04d} loss {:.6f}".format(epoch+1, step+1, loss))
                                        
                print("Now for test data\nCould take few minutes")
                sess.run(self.test_init_op, feed_dict = test_feed_dict)
                test_loss = 0
                pred_list, true_list = [], []
                
                for step in range(num_test_batch):                  
                    loss = sess.run(self.loss, feed_dict={self.dropout: 0.0})
                    pred_, true_ = self.sample_test(self.data, sess)
                    pred_list += pred_
                    true_list += true_
                    test_loss += loss/num_test_batch
                
                bleu_score = corpus_bleu(true_list, pred_list)*100
                
                print("epoch {:02d} train loss {:.6f}".format(epoch+1, train_loss))   
                print("epoch {:02d} test loss {:.6f}".format(epoch+1, test_loss))
                print("epoch {:02d} bleu_score {:.6f}".format(epoch+1, bleu_score))
                train_loss_list.append(train_loss)
                test_loss_list.append(test_loss)
                test_bleu_list.append(bleu_score)
                sess.run(self.train_init_op, feed_dict = train_feed_dict) 
                save_path = saver.save(sess, modelpath + modelName)
                np.save(modelpath + "char_embedding.npy", char_embedding_matrix)
                print ('save_path',save_path)
                
            result = pd.DataFrame({"train_loss":train_loss_list,
                                   "test_loss":test_loss_list,
                                   "test_bleu":test_bleu_list})
            
            result.to_csv("./loss.csv", sep =",", index=False)
            elapsed_time = datetime.datetime.now() - start_time
            print("{}".format(elapsed_time))
            

    def char_cnn(self, input_, kernels, kernel_features, scope='char_cnn'):
        '''
        :input:           input float tensor of shape  [batch_size, max_word_len, max_word_len, char_embed_size]
        :kernel_features: array of kernel feature sizes (parallel to kernels)
        '''
        assert len(kernels) == len(kernel_features), 'Kernel and Features must have the same size'
        
        max_word_len = input_.get_shape()[1]
        max_char_len = input_.get_shape()[2]
        char_embed_size = input_.get_shape()[3]
        
        input_ = tf.reshape(input_, [-1, max_char_len, char_embed_size])
    
        input_ = tf.expand_dims(input_, 1) # input_: [batch_size*max_word_len, 1, max_char_len, char_embed_size]
        
        layers = []
        with tf.variable_scope(scope):
            for kernel_size, kernel_feature_size in zip(kernels, kernel_features):
                reduced_length = max_char_len - kernel_size + 1
    
                # [batch_size*max_word_len, 1, reduced_length, kernel_feature_size]
                conv = self.conv2d(input_, kernel_feature_size, 1, kernel_size, name="kernel_%d" % kernel_size)
    
                # [batch_size*max_word_len, 1, 1, kernel_feature_size]
                pool = tf.nn.max_pool(tf.tanh(conv), [1, 1, reduced_length, 1], [1, 1, 1, 1], 'VALID')
    
                layers.append(tf.squeeze(pool, [1, 2]))
    
            if len(kernels) > 1:
                output = tf.concat(layers, 1) # [batch_size*max_word_len, sum(kernel_features)]
            else:
                output = layers[0]
            
            # [batch_size*max_word_len, sum(kernel_features)]
            output = tf.layers.dense(output, self.hidden_dim, activation = None) ## projection layer
            output = self.highway(output, output.get_shape()[-1], num_layers = 2)
            output = tf.reshape(output, (-1, max_word_len, self.hidden_dim)) 
            
        return output
    
    def conv2d(self, input_, output_dim, k_h, k_w, name="conv2d"):
        with tf.variable_scope(name):
            w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim])
            b = tf.get_variable('b', [output_dim])   
        return tf.nn.conv2d(input_, w, strides=[1, 1, 1, 1], padding='VALID') + b
    
    
    def highway(self, input_, size, num_layers=1, scope='Highway'):
        """Highway Network (cf. http://arxiv.org/abs/1505.00387).
        t = sigmoid(Wy + b)
        z = t * g(Wy + b) + (1 - t) * y
        where g is nonlinearity, t is transform gate, and (1 - t) is carry gate.
        """   
        with tf.variable_scope(scope):
            for idx in range(num_layers):
                g = tf.nn.relu(tf.layers.dense(input_, size, name='highway_lin_%d' % idx))
    
                t = tf.sigmoid(tf.layers.dense(input_, size, name='highway_gate_%d' % idx))
    
                output = t * g + (1. - t) * input_
                input_ = output
    
        return output


    def build_embed(self, inputs, seq_len=None, encoder=True, reuse=False):
        ## inputs : batch_size, max_word_len, max_char_len
        with tf.variable_scope("Embeddings", reuse=reuse, initializer=tf.contrib.layers.xavier_initializer()):
            self.shared_weights = tf.get_variable('shared_weights', [self.vocab, self.hidden_dim], dtype = tf.float32)     
            
            kernels = [1, 2, 3, 4, 5, 6, 7]
            kernel_features = [32, 32, 64, 128, 256, 512, 1024] ## sum up to 2048 
            
            if encoder:
                max_seq_length = self.max_enc_len
                self.char_embedding = tf.get_variable('char_embedding', [self.char_vocab_size, self.char_dim],
                                                      initializer=tf.random_uniform_initializer(-1.0, 1.0), trainable = True)   
                self.clear_char_embedding_padding = tf.scatter_update(self.char_embedding, [0], tf.constant(0.0, shape=[1, self.char_dim]))
                char_emb = tf.nn.embedding_lookup(self.char_embedding, inputs)
                word_emb = self.char_cnn(char_emb, kernels, kernel_features, scope='char_cnn') ## batch_size, max_word_len, dim
                mask = tf.sequence_mask(lengths=seq_len, maxlen=max_seq_length, dtype=tf.float32) ## batch_size, max_word_len
            else:
                max_seq_length = self.max_dec_len
                word_emb = tf.nn.embedding_lookup(self.shared_weights, inputs)  ## batch_size, length, dim
                mask = tf.to_float(tf.not_equal(inputs, 0))

            # Positional Encoding
            with tf.variable_scope("positional-encoding"):
                positional_encoded = model_utils.get_position_encoding(max_seq_length,
                                                                       self.hidden_dim)
            batch_size = tf.shape(inputs)[0]

            ## Add
            word_emb *= tf.expand_dims(mask, -1) ## zeros out masked positions
            word_emb *= self.hidden_dim ** 0.5 ## Scale embedding by the sqrt of the hidden size
            position_inputs = tf.tile(tf.range(0, max_seq_length), [batch_size])
            position_inputs = tf.reshape(position_inputs, [batch_size, max_seq_length])
            position_emb = tf.nn.embedding_lookup(positional_encoded, position_inputs)                       
            encoded_inputs = tf.add(word_emb, position_emb)
            
            if encoder:
                encoded_inputs = tf.nn.dropout(encoded_inputs, 1.0 - self.dropout) ## with position
                word_emb =  tf.nn.dropout(word_emb, 1.0 - self.dropout) ## with out position
                return encoded_inputs, word_emb
            else:
                return tf.nn.dropout(encoded_inputs, 1.0 - self.dropout)

    def build_encoder(self, encoder_emb_inp, char_inp, attention_bias, reuse=False):
        ## x: (batch_size, enc_len)
        padding_bias = attention_bias
        with tf.variable_scope("Encoder", reuse=reuse, initializer=tf.contrib.layers.xavier_initializer()):
            encoder = Encoder(num_layers=self.num_layers,
                              num_heads=self.num_heads,
                              linear_key_dim=self.linear_key_dim,
                              linear_value_dim=self.linear_value_dim,
                              model_dim=self.hidden_dim,
                              ffn_dim=self.ffn_dim)
            mask = tf.sequence_mask(lengths=self.iter_x_len, maxlen=self.max_enc_len, dtype=tf.float32)
            padding = 1 - mask
            return encoder.build(encoder_emb_inp, char_inp, padding_bias, padding=padding)

    def build_decoder(self, decoder_emb_inp, encoder_outputs, dec_bias, attention_bias, reuse=False):
        enc_dec_bias = attention_bias
        with tf.variable_scope("Decoder", reuse=reuse, initializer=tf.contrib.layers.xavier_initializer()):
            decoder = Decoder(num_layers=self.num_layers,
                              num_heads=self.num_heads,
                              linear_key_dim=self.linear_key_dim,
                              linear_value_dim=self.linear_value_dim,
                              model_dim=self.hidden_dim,
                              ffn_dim=self.ffn_dim)
            return decoder.build(decoder_emb_inp, encoder_outputs, dec_bias, enc_dec_bias)

    def build_output(self, decoder_outputs, reuse=False):
        with tf.variable_scope("Output", reuse=reuse):
            t_shape = decoder_outputs.get_shape().as_list() ## batch_size, seq_length, dim
            seq_length = t_shape[1]
            decoder_outputs = tf.reshape(decoder_outputs, [-1,self.hidden_dim])
            logits = tf.matmul(decoder_outputs, self.shared_weights, transpose_b=True)
            logits = tf.reshape(logits, [-1, seq_length, self.vocab])
        return logits
    
    def decoder_train(self, x_in, x_len, y):
        ## x_in: (batch_size, enc_len, char_len) , y_in: (batch_size, dec_len, char_len)
        dec_bias = model_utils.get_decoder_self_attention_bias(self.max_dec_len)
        attention_bias = model_utils.get_padding_bias(x_len, self.max_enc_len)
        # Encoder
        encoder_emb_inp, char_inp = self.build_embed(x_in, x_len, encoder=True, reuse=False)
        encoder_outputs = self.build_encoder(encoder_emb_inp, char_inp, attention_bias, reuse=False)
        # Decoder
        batch_size = tf.shape(x_in)[0]
        start_tokens = tf.fill([batch_size, 1], self.bos_idx) # 2: <s> ID
        target_slice_last_1 = tf.slice(y, [0, 0], [batch_size, self.max_dec_len-1])
        decoder_inputs = tf.concat([start_tokens, target_slice_last_1], axis=1) ## shift to right
        decoder_emb_inp = self.build_embed(decoder_inputs, encoder=False, reuse=True)
        decoder_outputs = self.build_decoder(decoder_emb_inp, encoder_outputs, dec_bias, attention_bias, reuse=False)
        train_prob = self.build_output(decoder_outputs, reuse=False)
        return encoder_outputs, decoder_inputs, train_prob

    def decoder_infer(self, x_in, x_len):   
        dec_bias = model_utils.get_decoder_self_attention_bias(self.max_dec_len)
        attention_bias = model_utils.get_padding_bias(x_len, self.max_enc_len)
        # Encoder
        encoder_emb_inp, char_inp = self.build_embed(x_in, x_len, encoder=True, reuse=True)
        encoder_outputs = self.build_encoder(encoder_emb_inp, char_inp, attention_bias, reuse=True)
        # Decoder
        batch_size = tf.shape(x_in)[0]
        start_tokens = tf.fill([batch_size, 1], self.bos_idx) # 2: <s> ID
        next_decoder_inputs = tf.concat([start_tokens, tf.zeros([batch_size, self.max_dec_len-1], dtype=tf.int32)], axis=1) ## batch_size, dec_len   
        # predict output with loop. [encoder_outputs, decoder_inputs (filled next token)]
        for i in range(1, self.max_dec_len):
            decoder_emb_inp = self.build_embed(next_decoder_inputs, encoder=False, reuse=True)
            decoder_outputs = self.build_decoder(decoder_emb_inp, encoder_outputs, dec_bias, attention_bias, reuse=True)
            logits = self.build_output(decoder_outputs, reuse=True)
            next_decoder_inputs = self._filled_next_token(next_decoder_inputs, logits, i)

        # slice start_token
        decoder_input_start_1 = tf.slice(next_decoder_inputs, [0, 1], [batch_size, self.max_dec_len-1])
        output_token = tf.concat([decoder_input_start_1, tf.zeros([batch_size, 1], dtype=tf.int32)], axis=1)
        return output_token

    def _filled_next_token(self, inputs, logits, decoder_index):
        batch_size = tf.shape(inputs)[0]
        next_token = tf.slice(
                tf.argmax(logits, axis=2, output_type=tf.int32),
                [0, decoder_index - 1],
                [batch_size, 1])
        left_zero_pads = tf.zeros([batch_size, decoder_index], dtype=tf.int32)
        right_zero_pads = tf.zeros([batch_size, (self.max_dec_len-decoder_index-1)], dtype=tf.int32)
        next_token = tf.concat((left_zero_pads, next_token, right_zero_pads), axis=1)
        return inputs + next_token

    def build_loss(self, train_prob):
        # sequence mask for different size
        self.masks = tf.sequence_mask(lengths=self.iter_y_len, maxlen=self.max_dec_len, dtype=tf.float32)
        y_ = self.label_smoothing(tf.one_hot(self.iter_y_out, depth=self.vocab))
        self.cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_, logits=train_prob)
        self.loss = tf.reduce_sum(self.cross_entropy * self.masks) / (tf.reduce_sum(self.masks) + 1e-10)

    def label_smoothing(self, inputs, epsilon=0.1):
        '''Applies label smoothing. See 5.4 and https://arxiv.org/abs/1512.00567.
        inputs: 3d tensor. [N, T, V], where V is the number of vocabulary.
        epsilon: Smoothing rate.
        '''
        V = inputs.get_shape().as_list()[-1] # number of channels
        return ((1-epsilon) * inputs) + (epsilon / V)
        
    def build_opt(self, global_step):
        # define optimizer
        learning_rate = self.noam_scheme(self.linear_key_dim , global_step, self.warmup_steps)
        self.optimizer = tf.contrib.opt.LazyAdamOptimizer(learning_rate=learning_rate, 
                                                          beta1=0.9, beta2=0.98, epsilon=1e-9).minimize(self.loss, global_step=global_step)
    def noam_scheme(self, d_model, global_step, warmup_steps=4000):
        '''Noam scheme learning rate decay
        init_lr: initial learning rate. scalar.
        global_step: scalar.
        warmup_steps: scalar. During warmup_steps, learning rate increases
            until it reaches init_lr.
        '''
        step = tf.cast(global_step + 1, dtype=tf.float32)
        return d_model ** (-0.5) * tf.minimum(step * warmup_steps ** -1.5, step ** -0.5)

    def restore(self, sess):
        print(' - Restoring variables...')
        var_list = [var for var in tf.all_variables()]
        saver = tf.train.Saver(var_list)
        saver.restore(sess, "models/model")
        print(' * model restored ')

    def load_char_embedding(self, filename):
        print("Char embedding loaded!")
        self.char_embedding = np.load(filename)
       
    def sample_test(self, data, sess):        
        pred_token, dec = sess.run([self.pred_token, self.iter_y_out], feed_dict={self.dropout: 0.0})
        
        sample_idx = random.randint(0,len(pred_token)-1)
        pred_list = []
        true_list = []
        
        for i in range(len(pred_token)):            
            pred_line = " ".join([data.idx2w[o] for o in pred_token[i]]).split("</s>")[0].strip()
            true_line = " ".join([data.idx2w[o] for o in dec[i]]).split("</s>")[0].strip()           
            pred_list.append(pred_line.split(" "))
            true_list.append([true_line.split(" ")])
            
            if(i == sample_idx):
                sample_pred = pred_line
                sample_true = true_line

        print("Decoder True ===> {}".format(sample_true).encode('utf-8'))
        print("Decoder Pred ===> {}".format(sample_pred).encode('utf-8'))
        print("="*90)
        print()
        return pred_list, true_list
