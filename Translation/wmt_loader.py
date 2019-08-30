# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 13:57:59 2019

@author: jbk48
"""


class Data(object):

    def __init__(self, path, max_enc_len=100, max_dec_len=100, max_char_len=50):

        self.path = path

        self.max_enc_len , self.max_dec_len = max_enc_len, max_dec_len
        self.max_char_len = max_char_len
        
        self.pad_token, self.pad_idx = "<pad>", 0
        self.unk_token, self.unk_idx = "<unk>", 1
        self.bos_token, self.bos_idx = "<s>", 2
        self.eos_token, self.eos_idx = "</s>", 3

        self.w2idx, self.c2idx = self.read_vocab()
        self.idx2w = dict(zip(self.w2idx.values(), self.w2idx.keys()))
        self.idx2c = dict(zip(self.c2idx.values(), self.c2idx.keys()))
        self.vocab = len(self.w2idx)
        

    def read_vocab(self):
        w2idx = {self.pad_token: self.pad_idx, self.unk_token: self.unk_idx, self.bos_token: self.bos_idx, self.eos_token: self.eos_idx}
        c2idx = {self.pad_token: self.pad_idx, self.unk_token: self.unk_idx, self.bos_token: self.bos_idx, self.eos_token: self.eos_idx}
        
        with open(self.path + "/" + "vocab.en", encoding="utf-8") as fin:
            lines = fin.readlines()
        for i, line in enumerate(lines):
            word = line.rstrip()
            if word not in w2idx:
                w2idx[word] = len(w2idx)
            
        with open(self.path + "/" + "vocab.vi", encoding="utf-8") as fin:
            lines = fin.readlines()
        for i, line in enumerate(lines):
            word = line.rstrip()
            for char in word:
                if char not in c2idx:
                    c2idx[char] = len(c2idx)
                    
        return w2idx, c2idx


    def read_file(self, name):
        print("preparing {} file".format(name))
        
        with open(self.path + "/" + name + ".vi", encoding="utf-8") as f:
            enc = f.readlines()
        
        with open(self.path + "/" + name + ".en", encoding="utf-8") as f:
            dec = f.readlines()
        
        enc_char_idx, enc_len = [], []
        dec_out_idx, dec_len = [], []
        
        for sent1, sent2 in zip(enc,dec):
            if(len(sent1.split(" ")) >self.max_enc_len-1):
                continue
            if(len(sent2.split(" ")) >self.max_dec_len-1):
                continue
            if(sent1 != "\n" and sent2 != "\n"):
                sent1 = sent1.replace("\n", " </s>").split(" ")
                enc_len.append(len(sent1))
                enc_char_idx.append(self.sent2charidx(sent1, self.c2idx, self.max_enc_len, self.max_char_len))
            
                sent2 = sent2.replace("\n", " </s>").split(" ")
                dec_len.append(len(sent2))
                dec_out_idx.append(self.sent2wordidx(sent2, self.w2idx, self.max_dec_len))
        
        return enc_char_idx, dec_out_idx, enc_len, dec_len

    def sent2wordidx(self, sent, w2idx, max_word_len):
        idx = []
        for word in sent:
            if(word in w2idx):
                idx.append(w2idx[word])
            else:
                idx.append(self.unk_idx)      
        return idx + [0]*(max_word_len-len(idx)) ## PAD for max length

    def sent2charidx(self, sent, c2idx, max_word_len, max_char_len):
        def word2charidx(word, c2idx, max_char_len):
            char_idx = []
            if(len(word) > max_char_len):
                word = word[:max_char_len]
            if(word == "<s>"):
                return [self.bos_idx] + [0]*(max_char_len-1)
            elif(word == "</s>"):
                return [self.eos_idx] + [0]*(max_char_len-1)
            else:
                for char in word:
                    if(char in c2idx):
                        char_idx.append(c2idx[char])
                    else:
                        char_idx.append(self.unk_idx)          
                return char_idx + [0]*(max_char_len-len(char_idx))
        
        idx = []

        for word in sent:
            idx.append(word2charidx(word, c2idx, max_char_len))
        result = idx + [[0]*max_char_len]*(max_word_len-len(idx)) ## PAD for max length            
        return result
