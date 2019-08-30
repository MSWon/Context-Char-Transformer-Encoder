# -*- coding: utf-8 -*-
"""
Created on Fri Aug 16 11:52:48 2019

@author: jbk48
"""

import tensorflow as tf
import pandas as pd
import datetime
import os
import numpy as np
import transformer
from attention import positional_encoding
from seqeval.metrics import f1_score, recall_score, precision_score
from ner_preprocess import Preprocess

os.environ['CUDA_VISIBLE_DEVICES'] = "0"
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

class Model():
    
    def __init__(self, *args):

        self.num_layers=args[0]
        self.num_heads=args[1]
        self.linear_key_dim=args[2]
        self.linear_value_dim=args[3]
        self.model_dim=args[4]
        self.ffn_dim=args[5]        
        self.char_dim = args[6]
        self.max_word_len = args[7]
        self.max_char_len = args[8]
        self.batch_size = args[9]
        self.learning_rate = args[10]
        
        ## Placeholders
        self.char_input = tf.placeholder(tf.int32, shape = [None, self.max_word_len, self.max_char_len], name = 'ner-char_input')
        self.seq_len = tf.placeholder(tf.int32, shape = [None], name = 'ner-seq_len')
        self.label = tf.placeholder(tf.int32, shape = [None, self.max_word_len], name = 'ner-label')
        self.dropout = tf.placeholder(tf.float32, shape = (), name = 'ner-dropout')
        
        self.prepro = Preprocess(max_word_len=self.max_word_len, max_char_len=self.max_char_len)
        self.char_vocab_size = len(self.prepro.char2idx)
        self.num_class = len(self.prepro.idx2label)
        self.train_char_idx, self.train_len, self.train_Y = self.prepro.load_data("./ner_data/train.txt")
        self.test_char_idx, self.test_len, self.test_Y = self.prepro.load_data("./ner_data/test.txt")       
        self.train_size, self.test_size = len(self.train_Y), len(self.test_Y)
        num_train_steps = int(self.train_size / self.batch_size) + 1
        
        train_dataset = tf.data.Dataset.from_tensor_slices((self.char_input, self.seq_len, self.label))
        train_dataset = train_dataset.shuffle(self.train_size)
        train_dataset = train_dataset.batch(self.batch_size)
        train_dataset = train_dataset.repeat()

        test_dataset = tf.data.Dataset.from_tensor_slices((self.char_input, self.seq_len, self.label))
        test_dataset = test_dataset.batch(self.batch_size)
        test_dataset = test_dataset.repeat()

        iters = tf.data.Iterator.from_structure(train_dataset.output_types, train_dataset.output_shapes)        
        self.iter_char_input, self.iter_seq_len, self.iter_label = iters.get_next()
        # create the initialisation operations
        self.train_init_op = iters.make_initializer(train_dataset)
        self.test_init_op = iters.make_initializer(test_dataset)        

        ## Build graph
        self.build_model()
        self.build_optimizer(num_train_steps)
        
    def train(self, training_epochs):
        num_train_batch = int(self.train_size / self.batch_size) + 1
        num_test_batch = int(self.test_size / self.batch_size) + 1
        
        train_feed_dict = {self.char_input: self.train_char_idx, self.seq_len: self.train_len, self.label: self.train_Y}        
        test_feed_dict = {self.char_input: self.test_char_idx, self.seq_len: self.test_len, self.label: self.test_Y}
        
        modelpath = "./pos-model/"
        modelName = "pos.ckpt"
        saver = tf.train.Saver()  
        best_f1_score = 0.
        
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
            start_time = datetime.datetime.now()
            
            train_loss_list, test_loss_list = [], []
            test_f1_list, test_recall_list, test_precision_list = [], [], []
            
            print("start training")
            
            for epoch in range(training_epochs):
                
                train_loss = 0.           
                for step in range(num_train_batch):
                    char_embedding_matrix = sess.run(self.clear_char_embedding_padding)
                    loss, _ = sess.run([self.loss, self.optimizer],
                                        feed_dict={self.dropout: 0.2})               
                    train_loss += loss/num_train_batch
                    print("epoch {:02d} step {:04d} loss {:.6f}".format(epoch+1, step+1, loss))
                                        
                print("Now for test data\nCould take few minutes")
                sess.run(self.test_init_op, feed_dict = test_feed_dict)
                y_true_list, y_pred_list, test_loss = [], [], 0.              
                for step in range(num_test_batch):                  
                    loss = sess.run(self.loss, feed_dict={self.dropout: 0.0})
                    y_true, y_pred = self.predict(sess)
                    y_true_list += y_true
                    y_pred_list += y_pred
                    test_loss += loss/num_test_batch
                
                test_f1_score = f1_score(y_true_list, y_pred_list)
                test_recall_score = recall_score(y_true_list, y_pred_list)
                test_precision_score = precision_score(y_true_list, y_pred_list)
                
                print("epoch {:02d} [train] loss {:.6f}".format(epoch+1, train_loss))   
                print("epoch {:02d} [test] loss {:.6f} f1 {:.4f} recall {:.4f} precision {:.4f}".format(epoch+1, test_loss, test_f1_score, test_recall_score, test_precision_score))
                train_loss_list.append(train_loss)
                test_loss_list.append(test_loss)
                test_f1_list.append(test_f1_score)
                test_recall_list.append(test_recall_score)
                test_precision_list.append(test_precision_score)
                sess.run(self.train_init_op, feed_dict = train_feed_dict)
                
                if(best_f1_score <= test_f1_score):
                    best_f1_score = test_f1_score
                    saver.save(sess, modelpath + modelName)
                    np.save(modelpath + "char_embedding.npy", char_embedding_matrix) 
                    
            result = pd.DataFrame({"train_loss":train_loss_list,
                                   "test_loss":test_loss_list,
                                   "test_f1":test_f1_list,
                                   "test_recall":test_recall_list ,
                                   "test_precision": test_precision_list})
            
            result.to_csv("./loss_pos.csv", sep =",", index=False)
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
            output = self.highway(output, output.get_shape()[-1], num_layers = 2)
            output = tf.layers.dense(output, self.model_dim, activation = None) ## projection layer
            output = tf.reshape(output, (-1, max_word_len, self.model_dim)) 
            
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
               
    def build_embed(self, char_inputs, seq_len):        
        with tf.variable_scope("positional-encoding"):
            positional_encoded = positional_encoding(self.model_dim,
                                                     self.max_word_len)
        batch_size = tf.shape(char_inputs)[0]
        position_inputs = tf.tile(tf.range(0, self.max_word_len), [batch_size])
        position_inputs = tf.reshape(position_inputs, [batch_size, self.max_word_len]) # batch_size x [0, 1, 2, ..., n]     

        self.char_embedding = tf.get_variable('char_embedding', [self.char_vocab_size, self.char_dim],
                                              initializer=tf.random_uniform_initializer(-1.0, 1.0), trainable = True)   
        self.clear_char_embedding_padding = tf.scatter_update(self.char_embedding, [0], tf.constant(0.0, shape=[1, self.char_dim]))
        char_emb = tf.nn.embedding_lookup(self.char_embedding, char_inputs)
        kernels = [1, 2, 3, 4, 5, 6, 7]
        kernel_features = [32, 32, 64, 128, 256, 512, 1024] ## sum up to 2048 
        char_cnn_output = self.char_cnn(char_emb, kernels, kernel_features, scope='char_cnn')
        mask = tf.sequence_mask(lengths=seq_len, maxlen=self.max_word_len, dtype=tf.float32) ## batch_size, max_word_len
        char_cnn_output *= tf.expand_dims(mask, -1) ## zeros out masked positions
        ## char_cnn_output *= self.model_dim ** 0.5 ## Scale embedding by the sqrt of the hidden size
        encoded_inputs = tf.add(char_cnn_output, tf.nn.embedding_lookup(positional_encoded, position_inputs))
        return encoded_inputs, char_cnn_output


    def build_model(self):       
        print("Building model!") 
            
        with tf.variable_scope("Transformer", initializer=tf.contrib.layers.xavier_initializer()):                       
            self.W = tf.get_variable(name="W", shape=[self.model_dim, self.num_class],
                                dtype=tf.float32, initializer = tf.contrib.layers.xavier_initializer())
            self.b = tf.get_variable(name="b", shape=[self.num_class], dtype=tf.float32,
                                initializer=tf.zeros_initializer())
                 
            self.W_CRF = tf.get_variable(name="W_CRF", shape=[self.num_class, self.num_class],
                                dtype=tf.float32, initializer = tf.contrib.layers.xavier_initializer())
            self.b_CRF = tf.get_variable(name="b_CRF", shape=[self.num_class], dtype=tf.float32,
                                initializer=tf.zeros_initializer())
            
            encoder = transformer.Encoder(num_layers=self.num_layers,
                                  num_heads=self.num_heads,
                                  linear_key_dim=self.linear_key_dim,
                                  linear_value_dim=self.linear_value_dim,
                                  model_dim=self.model_dim,
                                  ffn_dim=self.ffn_dim,
                                  dropout=self.dropout,
                                  batch_size=self.batch_size)
            encoder_emb, char_cnn_output = self.build_embed(self.iter_char_input, self.iter_seq_len)
            mask = tf.sequence_mask(lengths=self.iter_seq_len, maxlen=self.max_word_len, dtype=tf.float32)
            padding = 1 - mask
            encoder_output = encoder.build(encoder_emb, char_cnn_output, self.iter_seq_len, padding)  ## batch_size, word_len, dim      

            self.layer_hidden_states = encoder.layer_hidden_states ## list of each layer hidden states
            self.layer_char_states = encoder.layer_char_states ## list of each layer char states
            self.layer_alpha_states = encoder.layer_alpha_states ## list of each layer alpha states
 
            encoder_output = tf.reshape(encoder_output, (-1, self.model_dim)) ## batch_size*word_len, dim 
            logits = tf.matmul(encoder_output, self.W) + self.b  ## batch_size*word_len, num_class
            matricized_unary_scores = tf.matmul(logits, self.W_CRF) + self.b_CRF ## batch_size*word_len, num_class          
            self.matricized_unary_scores = tf.reshape(matricized_unary_scores, (-1, self.max_word_len, self.num_class)) ## batch_size, word_len, num_class
              
    def build_optimizer(self, num_train_steps):          
        global_step = tf.Variable(0, trainable=False)
        learning_rate = tf.train.polynomial_decay(
                  self.learning_rate,
                  global_step,
                  num_train_steps,
                  end_learning_rate=0.0,
                  power=1.0,
                  cycle=False)
        
        self.log_likelihood, self.transition_params = tf.contrib.crf.crf_log_likelihood(self.matricized_unary_scores, self.iter_label, self.iter_seq_len)
        self.loss = tf.reduce_mean(-self.log_likelihood)
        self.viterbi_sequence, viterbi_score = tf.contrib.crf.crf_decode(self.matricized_unary_scores, self.transition_params, self.iter_seq_len)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.loss) # Adam Optimizer 

    def noam_scheme(self, d_model, global_step, warmup_steps=4000):
        '''Noam scheme learning rate decay
        init_lr: initial learning rate. scalar.
        global_step: scalar.
        warmup_steps: scalar. During warmup_steps, learning rate increases
            until it reaches init_lr.
        '''
        step = tf.cast(global_step + 1, dtype=tf.float32)
        return d_model ** (-0.5) * tf.minimum(step * warmup_steps ** -1.5, step ** -0.5)

    def convert_idx_to_name(self, y, lens):
        """Convert label index to name.
        Args:
            y (list): label index list.
            lens (list): true length of y.
        Returns:
            y: label name list.
        Examples:
            >>> # assumes that id2label = {1: 'B-LOC', 2: 'I-LOC'}
            >>> y = [[1, 0, 0], [1, 2, 0], [1, 1, 1]]
            >>> lens = [1, 2, 3]
            >>> self.convert_idx_to_name(y, lens)
            [['B-LOC'], ['B-LOC', 'I-LOC'], ['B-LOC', 'B-LOC', 'B-LOC']]
        """
        y = [[self.prepro.idx2label[idx] for idx in row[:l]]
             for row, l in zip(y, lens)]
        return y
    
    def predict(self, sess):        
        y_pred, y_true, seq_len = sess.run([self.viterbi_sequence, self.iter_label, self.iter_seq_len],
                                            feed_dict={self.dropout:0.0})
        
        y_true = self.convert_idx_to_name(y_true, seq_len)
        y_pred = self.convert_idx_to_name(y_pred, seq_len)
        return y_true, y_pred


    def load_char_embedding(self, filename):
        print("Char embedding loaded!")
        self.char_embedding = np.load(filename)