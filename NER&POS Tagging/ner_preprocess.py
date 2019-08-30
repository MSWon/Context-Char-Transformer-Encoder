# -*- coding: utf-8 -*-
"""
Created on Fri Aug 16 14:09:40 2019

@author: jbk48
"""


class Preprocess():
    
    def __init__(self, max_word_len=50, max_char_len=50):
        self.max_word_len = max_word_len
        self.max_char_len = max_char_len
        self.build_vocab()
        self.build_label()
        
    def load_data(self, filename):
        print("Making {} corpus!".format(filename)) 
        return self.prepare_data(filename) 

    def build_vocab(self):
        self.char2idx = {"<PAD>":0, "<UNK>":1}
        rf = open("./ner_data/train.txt", 'r')
         
        for line in rf:
            if(len(line.strip())!=0):             
                word = line.strip().split(' ')[0]
                for char in word:
                    if char not in self.char2idx:
                        self.char2idx[char] = len(self.char2idx)
        self.idx2char = dict(zip(self.char2idx.values(), self.char2idx.keys()))
        rf.close()

    def read_data(self, filename):
        rf = open(filename, 'r')
        word_list = []; label_list = []
        words = []; labels = []
         
        for line in rf:
            if(len(line.strip())==0):
                word_list.append([word for word in words])
                label_list.append([label for label in labels])
                words=[]
                labels = []
            else:                
                word = line.strip().split(' ')[0]
                label = line.strip().split(' ')[1]
                words.append(word)
                labels.append(label)                        
        rf.close()
        return word_list, label_list

    def build_label(self):
        word_list, label_list = self.read_data("./ner_data/train.txt")
        r = []
        for l in label_list:
            r += l
        self.label2idx = {"<PAD>":0}
        
        for label in set(r):
            if label not in self.label2idx:
                self.label2idx[label] = len(self.label2idx)
        self.idx2label = dict(zip(self.label2idx.values(), self.label2idx.keys()))

    def prepare_data(self, train_filename):        
        corpus, labels = self.read_data(train_filename)
        
        input_char_idx = []
        input_seq_length = []
        label_list = []    
       
        for idx,sent in enumerate(corpus):                    

            if(len(sent) > self.max_word_len): ## skip
                continue
            else:

                ## Output label
                label_ids = []
                for label in labels[idx]:
                    label_ids.append(self.label2idx[label])
                label_ids += [0]*(self.max_word_len-len(label_ids))
                    
                ## Input char
                char_ids = []   ## (max_word_len, max_char_len)
                for i in range(self.max_word_len):
                    if(i >= len(sent)):
                        char_ids.append([0]*self.max_char_len)  ## For <PAD> token
                    else:
                        if(len(sent[i]) > self.max_char_len-2): ## truncated to max char length
                            sent[i] = sent[i][:self.max_char_len-2]
                        char_zero_padding = [0]*self.max_char_len
                        char_list = sent[i]
                        
                        for j in range(len(char_list)):
                            if(char_list[j] in self.char2idx):
                                char_zero_padding[j] = self.char2idx[char_list[j]]
                            else:
                                char_zero_padding[j] = self.char2idx["<UNK>"]
                                
                        char_ids.append(char_zero_padding)
                
                input_char_idx.append(char_ids)
                input_seq_length.append(len(sent))
                label_list.append(label_ids)
        return input_char_idx, input_seq_length, label_list