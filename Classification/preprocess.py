# -*- coding: utf-8 -*-
"""
Created on Mon Aug 20 21:13:33 2018

@author: jbk48
"""

import pandas as pd
import numpy as np
import os
import re
import tensorflow as tf
import pickle
from nltk import tokenize
from sklearn.preprocessing import LabelBinarizer

class Preprocess():
    
    def __init__(self, char_dim=16, max_word_len=100, max_char_len=50):
        self.char_dim = char_dim
        self.max_word_len = max_word_len
        self.max_char_len = max_char_len

    def load_data(self, filename):
        print("Making corpus!") 
        train_char_idx, train_seq_length, train_Y = self.prepare_data(filename)
        return train_char_idx, train_seq_length, train_Y
    
    def read_data(self, filename):        
        data = pd.read_csv(filename)      
        labels = data.iloc[:,0]
        corpus = data.iloc[:,2]    
        encoder = LabelBinarizer()
        encoder.fit(labels)
        labels = encoder.transform(labels)
        labels = np.array([np.argmax(x) for x in labels], dtype=np.int32)         
        return corpus, labels
        
    def prepare_embedding(self):
        if(os.path.exists('./char2idx.pkl')):           
            with open('./char2idx.pkl', 'rb') as f:
                self.char2idx = pickle.load(f)

            self.idx2char = dict(zip(self.char2idx.values(), self.char2idx.keys()))
            self.char_vocab_size = len(self.char2idx)
            self.char_embedding = tf.get_variable('char_embedding', [self.char_vocab_size, self.char_dim],
                                                  initializer=tf.random_uniform_initializer(-1.0, 1.0))             
            self.clear_char_embedding_padding = tf.scatter_update(self.char_embedding, [0], tf.constant(0.0, shape=[1, self.char_dim]))
        
        return self.idx2char, self.char_embedding
                        
    def prepare_data(self, train_filename):        
        corpus, labels = self.read_data(train_filename)                        
        input_char_idx = []
        input_seq_length = []
        num = 1
        for sent in corpus:
            if(num % 10000 == 0):
                    print("{} sentence processing".format(num))
                    
            sent = sent.replace("\\", " ")
            sent = re.sub('<br />', ' ', sent)
            sent = re.sub(r"(\W)\1+", r"\1" , sent)  ## get rid of duplicate symbols
            token_list = tokenize.word_tokenize(sent)
            
            if(len(token_list) > self.max_word_len):
                token_list = token_list[:self.max_word_len]
                
            sub_input_char_idx = []
            ## Input char
            for i in range(self.max_word_len):
                if(i >= len(token_list)):
                    sub_input_char_idx.append([0]*self.max_char_len)  ## For <PAD> token
                else:
                    if(len(token_list[i]) > self.max_char_len): ## truncated to max char length
                        token_list[i] = token_list[i][:self.max_char_len]
                    char_zero_padding = [0]*self.max_char_len
                    for j in range(len(token_list[i])):
                        if(token_list[i][j] in self.char2idx):
                            char_zero_padding[j] = self.char2idx[token_list[i][j]]
                    sub_input_char_idx.append(char_zero_padding)

            input_char_idx.append(sub_input_char_idx)
            input_seq_length.append(len(token_list))
            
            num += 1
      
        return input_char_idx, input_seq_length, labels