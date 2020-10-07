import pickle
import re
import numpy as np
from collections import OrderedDict

class Data():  
    def __init__(self, train_filename, char_vocab_path, max_word_len, max_char_len):
        self.max_word_len = max_word_len
        self.max_char_len = max_char_len
        self.build_vocab(char_vocab_path)
        self.build_label(train_filename)
        
    def load_data(self, filename):
        print("Making {} corpus!".format(filename)) 
        return self.prepare_data(filename) 

    def build_vocab(self, char_vocab_path):
        self.char2idx = {}
        with open(char_vocab_path, 'r') as f:
            for line in f:
                char = line.strip()
                self.char2idx[char] = len(self.char2idx)

    def url_replace(self, sent):
        url_regex = "(http[s]?:/{1,2}([a-zA-Z]|[가-힣]|[0-9]|[-_@\.&+!*/])*)|(www.([a-zA-Z]|[가-힣]|[0-9]|[-_@\.&+!*/])+)"
        sent = re.sub(url_regex, "[URL]", sent)
        return sent

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
                word = self.url_replace(word)
                label = line.strip().split(' ')[-1]
                words.append(word)
                labels.append(label)                        
        rf.close()
        return word_list, label_list

    def build_label(self, train_filename):
        word_list, label_list = self.read_data(train_filename)
        r = []
        for l in label_list:
            r += l
        self.label2idx = OrderedDict({"<PAD>":0})
        
        for label in sorted(set(r)):
            if label not in self.label2idx:
                self.label2idx[label] = len(self.label2idx)
        self.idx2label = dict(zip(self.label2idx.values(), self.label2idx.keys()))

    def prepare_data(self, train_filename):        
        corpus, labels = self.read_data(train_filename)
        
        input_char_idx = []
        input_seq_length = []
        label_list = []    
       
        for idx,sent in enumerate(corpus):                    
            
            if(len(sent) > self.max_word_len or len(sent) <= 0):
                continue
            else:
                ## Output label
                label_ids = []
                for label in labels[idx]:
                    label_ids.append(self.label2idx[label])
                label_ids += [self.label2idx["<PAD>"]]*(self.max_word_len-len(label_ids))

                seq_len = len(sent)
                ## Input char
                char_ids = []   ## (max_word_len, max_char_len)
                for i in range(self.max_word_len):
                    if(i >= seq_len):
                        char_ids.append([0]*self.max_char_len)  ## For <PAD> token
                    else:
                        if(len(sent[i]) > self.max_char_len): ## truncated to max char length
                            sent[i] = sent[i][:self.max_char_len]
                        char_zero_padding = [0]*self.max_char_len
                        
                        char_list = list(sent[i])
                        
                        for j in range(len(char_list)):
                            if(char_list[j] in self.char2idx):
                                char_zero_padding[j] = self.char2idx[char_list[j]]
                            else:
                                char_zero_padding[j] = self.char2idx["<unk>"]
                                
                        char_ids.append(char_zero_padding)
                
                input_char_idx.append(char_ids)
                input_seq_length.append(seq_len)
                label_list.append(label_ids)
            
        return input_char_idx, input_seq_length, label_list
