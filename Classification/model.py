# -*- coding: utf-8 -*-
"""
Created on Sat Mar  2 12:55:38 2019

@author: jbk48
"""

import numpy as np
import tensorflow as tf
import transformer
import os
import datetime
import preprocess
import pandas as pd
from attention import positional_encoding

os.environ['CUDA_VISIBLE_DEVICES'] = "2"
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

class Model:

    def __init__(self, char_dim, max_word_len, max_char_len, learning_rate):
        
        self.char_dim = char_dim
        self.max_word_len = max_word_len
        self.max_char_len = max_char_len
        self.learning_rate = learning_rate

        ## Preprocess data
        self.prepro = preprocess.Preprocess(self.char_dim, self.max_word_len, self.max_char_len)
        self.idx2char, self.char_embedding = self.prepro.prepare_embedding()
        self.train_char_idx, self.train_seq_length, self.train_Y = self.prepro.load_data("./train.csv")
        self.test_char_idx, self.test_seq_length, self.test_Y = self.prepro.load_data("./test.csv")
       
        ## Placeholders   
        self.char_input = tf.placeholder(tf.int32, shape = [None, max_word_len, max_char_len], name = 'char')
        self.label = tf.placeholder(tf.int32, shape = [None], name = 'label')
        self.seq_len = tf.placeholder(tf.int32, shape = [None] , name = 'seq_len')
        self.dropout = tf.placeholder(tf.float32, shape = () , name = 'dropout')
        self.batch_size = tf.placeholder(tf.int64, shape = (), name = 'batch_size')
        
        train_dataset = tf.data.Dataset.from_tensor_slices((self.char_input, self.label, self.seq_len))
        train_dataset = train_dataset.shuffle(120000)
        train_dataset = train_dataset.batch(self.batch_size)
        train_dataset = train_dataset.repeat()

        test_dataset = tf.data.Dataset.from_tensor_slices((self.char_input, self.label, self.seq_len))
        test_dataset = test_dataset.batch(self.batch_size)
        test_dataset = test_dataset.repeat()

        iters = tf.data.Iterator.from_structure(train_dataset.output_types, train_dataset.output_shapes)
        iters = train_dataset.make_initializable_iterator()
        
        self.iter_char_input, self.iter_label, self.iter_seq_len = iters.get_next()
        # create the initialisation operations
        self.train_init_op = iters.make_initializer(train_dataset)
        self.test_init_op = iters.make_initializer(test_dataset)
        
    def train(self, batch_size, training_epochs, char_mode):
        
        num_train_steps = int((120000*training_epochs)/(batch_size))
        
        # Implements linear decay of the learning rate.
        global_step = tf.Variable(0, trainable=False)
        learning_rate = tf.train.polynomial_decay(
                  self.learning_rate,
                  global_step,
                  num_train_steps,
                  end_learning_rate=0.0,
                  power=1.0,
                  cycle=False)  
        
        loss, logits = self.build_model(self.iter_char_input, self.iter_label, self.iter_seq_len, self.n_class)        
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss, global_step=global_step) # Adam Optimizer
        accuracy = self.get_accuracy(logits, self.iter_label)        

        ## Training
        init = tf.global_variables_initializer()
        
        num_train_batch = int(len(self.train_Y) / batch_size) + 1
        num_test_batch = int(len(self.test_Y) / batch_size) + 1
        print("Start training!")
        
        modelpath = "./transformer_ag_news_{}/".format(char_mode)
        modelName = "transformer_ag_news_{}.ckpt".format(char_mode)
        saver = tf.train.Saver()
        
        train_acc_list = []
        train_loss_list = []
        test_acc_list = []
        test_loss_list = []
        
        with tf.Session(config = config) as sess:
        
            start_time = datetime.datetime.now()
            sess.run(init)
            if(not os.path.exists(modelpath)):
                os.mkdir(modelpath)
            ckpt = tf.train.get_checkpoint_state(modelpath)
            if(ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path)):
                self.load_char_embedding(modelpath + "char_embedding_{}.npy".format(char_mode))
                saver.restore(sess, modelpath + modelName)
                print("Model loaded!")
                
            train_feed_dict = {self.char_input: self.train_char_idx, 
                               self.label: self.train_Y,
                               self.seq_len: self.train_seq_length, 
                               self.batch_size: batch_size}  

            test_feed_dict = {self.char_input: self.test_char_idx, 
                              self.label: self.test_Y,
                              self.seq_len: self.test_seq_length, 
                              self.batch_size: batch_size}

            for epoch in range(training_epochs):
                sess.run(self.train_init_op, feed_dict = train_feed_dict)
                train_acc, train_loss = 0., 0.                
                for step in range(num_train_batch):
                    char_embedding_matrix = sess.run(self.prepro.clear_char_embedding_padding) ## clear 0 index to 0 vector
                    _, train_batch_loss = sess.run([optimizer,loss], feed_dict = {self.dropout : 0.2})
                              
                    train_loss += train_batch_loss / num_train_batch          
                    train_batch_acc = sess.run(accuracy , feed_dict = {self.dropout : 0.2})
                    train_acc += train_batch_acc / num_train_batch
                    print("epoch : {:02d} step : {:04d} loss = {:.6f} accuracy= {:.6f}".format(epoch+1, step+1, train_batch_loss, train_batch_acc))
                
                test_acc, test_loss = 0. , 0.
                print("Now for test data\nCould take few minutes")
                sess.run(self.test_init_op, feed_dict = test_feed_dict) 
                
                for step in range(num_test_batch):
                    # Compute average loss
                    test_batch_loss = sess.run(loss, feed_dict = {self.dropout : 0.0})
                    test_loss += test_batch_loss / num_test_batch                   
                    test_batch_acc = sess.run(accuracy , feed_dict = {self.dropout : 0.0})
                    test_acc += test_batch_acc / num_test_batch
                    
                print("<Train> Loss = {:.6f} Accuracy = {:.6f}".format(train_loss, train_acc))
                print("<Test> Loss = {:.6f} Accuracy = {:.6f}".format(test_loss, test_acc))
                train_loss_list.append(train_loss)
                train_acc_list.append(train_acc)
                test_loss_list.append(test_loss)
                test_acc_list.append(test_acc)
                np.save(modelpath + "char_embedding_{}.npy".format(char_mode), char_embedding_matrix)
            
            train_loss = pd.DataFrame({"train_loss":train_loss_list})
            train_acc = pd.DataFrame({"train_acc":train_acc_list})
            test_loss = pd.DataFrame({"test_loss":test_loss_list})
            test_acc = pd.DataFrame({"test_acc":test_acc_list})
            df = pd.concat([train_loss,train_acc,test_loss,test_acc], axis = 1)
            df.to_csv("./results_{}.csv".format(char_mode), sep =",", index=False)
            elapsed_time = datetime.datetime.now() - start_time
            print("{}".format(elapsed_time))
            save_path = saver.save(sess, modelpath + modelName)
            print ('save_path',save_path)

    def char_cnn(self, input_, kernels, kernel_features, scope='char_cnn'):
        '''
        :input:           input float tensor of shape  [batch_size, max_sent_len, max_word_len, char_embed_size]
        :kernel_features: array of kernel feature sizes (parallel to kernels)
        '''
        assert len(kernels) == len(kernel_features), 'Kernel and Features must have the same size'
        
        max_sent_len = input_.get_shape()[1]
        max_word_len = input_.get_shape()[2]
        char_embed_size = input_.get_shape()[3]
        
        input_ = tf.reshape(input_, [-1, max_word_len, char_embed_size])
    
        input_ = tf.expand_dims(input_, 1) # input_: [batch_size*max_sent_len, 1, max_word_len, char_embed_size]
        
        layers = []
        with tf.variable_scope(scope):
            for kernel_size, kernel_feature_size in zip(kernels, kernel_features):
                reduced_length = max_word_len - kernel_size + 1
    
                # [batch_size*max_sent_len, 1, reduced_length, kernel_feature_size]
                conv = self.conv2d(input_, kernel_feature_size, 1, kernel_size, name="kernel_%d" % kernel_size)
    
                # [batch_size*max_sent_len, 1, 1, kernel_feature_size]
                pool = tf.nn.max_pool(tf.tanh(conv), [1, 1, reduced_length, 1], [1, 1, 1, 1], 'VALID')
    
                layers.append(tf.squeeze(pool, [1, 2]))
    
            if len(kernels) > 1:
                output = tf.concat(layers, 1) # [batch_size*max_sent_len, sum(kernel_features)]
            else:
                output = layers[0]
            
            # [batch_size, max_sent_len, sum(kernel_features)]
            output = self.highway(output, output.get_shape()[-1], num_layers = 2)
            output = tf.layers.dense(output, self.model_dim, activation = None) ## projection layer
            output = tf.reshape(output, (-1, max_sent_len, self.model_dim))
            
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
    
    def build_parameter(self,num_layers, num_heads, linear_key_dim, linear_value_dim, model_dim, ffn_dim, n_class):
        
        self.num_layers=num_layers
        self.num_heads=num_heads
        self.linear_key_dim=linear_key_dim
        self.linear_value_dim=linear_value_dim
        self.model_dim=model_dim
        self.ffn_dim=ffn_dim
        self.n_class=n_class


    def build_model(self, char_inputs, labels, seq_len, n_class):
        print("Building model!")                
        encoder = transformer.Encoder(num_layers=self.num_layers,
                              num_heads=self.num_heads,
                              linear_key_dim=self.linear_key_dim,
                              linear_value_dim=self.linear_value_dim,
                              model_dim=self.model_dim,
                              ffn_dim=self.ffn_dim,
                              dropout=self.dropout,
                              batch_size=self.batch_size)
        encoder_emb, char_cnn_output = self.build_embed(char_inputs, seq_len)
        mask = tf.sequence_mask(lengths=seq_len, maxlen=self.max_word_len, dtype=tf.float32)
        padding = 1 - mask        
        encoder_outputs = encoder.build(encoder_emb, char_cnn_output, seq_len, padding, n_class)      
        
        self.layer_hidden_states = encoder.layer_hidden_states ## list of each layer hidden states
        self.layer_char_states = encoder.layer_char_states ## list of each layer char states
        self.layer_alpha_states = encoder.layer_alpha_states ## list of each layer alpha states
                  
        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits = encoder_outputs , labels = labels)) # Softmax loss        
        return loss, encoder_outputs
    
    def build_embed(self, char_inputs, seq_len):        
        with tf.variable_scope("positional-encoding"):
            positional_encoded = positional_encoding(self.model_dim,
                                                     self.max_word_len)
        batch_size = tf.shape(char_inputs)[0]
        position_inputs = tf.tile(tf.range(0, self.max_word_len), [batch_size])
        position_inputs = tf.reshape(position_inputs, [batch_size, self.max_word_len]) # batch_size x [0, 1, 2, ..., n]     
               
        char_emb = tf.nn.embedding_lookup(self.char_embedding, char_inputs)
        kernels = [1, 2, 3, 4, 5, 6, 7]
        kernel_features = [32, 32, 64, 128, 256, 512, 1024] ## sum up to 2048 
        char_cnn_output = self.char_cnn(char_emb, kernels, kernel_features, scope='char_cnn')
        mask = tf.sequence_mask(lengths=seq_len, maxlen=self.max_word_len, dtype=tf.float32) ## batch_size, max_word_len
        char_cnn_output *= tf.expand_dims(mask, -1) ## zeros out masked positions
        encoded_inputs = tf.add(char_cnn_output, tf.nn.embedding_lookup(positional_encoded, position_inputs))
        return encoded_inputs, char_cnn_output
    
    def get_accuracy(self, logits, label):
        pred = tf.cast(tf.argmax(logits, 1), tf.int32)
        correct_pred = tf.equal(pred, label)
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        return accuracy
    
    def load_char_embedding(self, filename):
        print("Char embedding loaded!")
        self.char_embedding = np.load(filename)

