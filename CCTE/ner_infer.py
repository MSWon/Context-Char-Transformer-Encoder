import tensorflow as tf
import pandas as pd
import datetime
import os
import numpy as np
import argparse
import yaml
from seqeval.metrics import f1_score, recall_score, precision_score, classification_report
from downstream.ner.model_utils import build_graph, restore, _get_layer_lrs, AdamWeightDecayOptimizer, LayerNormalization
from downstream.ner.data_utils import Data

os.environ['CUDA_VISIBLE_DEVICES'] = "7"
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

class NER_Model():
    def __init__(self, hyp_args, input_path):
        self.hyp_args = hyp_args
        self.max_word_len = hyp_args['max_word_len']
        self.max_char_len = hyp_args['max_char_len']
        self.batch_size = hyp_args['batch_size']
        self.data_path = hyp_args['data_path']
        self.pretrained_model_path = hyp_args['pretrained_model_path']
        self.training_epochs = hyp_args['training_epochs']
        self.train_type = hyp_args['train_type']
        self.test_type = hyp_args['test_type']
        self.char_vocab_path = hyp_args['char_vocab_path']
        self.D_num_layers = hyp_args['D_num_layers']
        self.hidden_dim = hyp_args['D_hidden_dim']
        self.layer_norm = LayerNormalization(self.hidden_dim)

        ## Placeholders
        self.char_input = tf.placeholder(tf.int32, shape = [None, self.max_word_len, self.max_char_len], name = 'ner-char_input')
        self.seq_len = tf.placeholder(tf.int32, shape = [None], name = 'ner-seq_len')
        self.label = tf.placeholder(tf.int32, shape = [None, self.max_word_len], name = 'ner-label')
        self.dropout = tf.placeholder(tf.float32, shape = (), name = 'ner-dropout')
        
        train_filename = f"{self.data_path}/train_{self.train_type}.txt"
        self.data = Data(train_filename, self.char_vocab_path, self.max_word_len, self.max_char_len)
        self.char_vocab_size = len(self.data.char2idx)
        self.num_class = len(self.data.idx2label)
        
        self.test_char_idx, self.test_len, self.test_Y = self.data.load_data(input_path)    
        self.test_size = len(self.test_Y)

        test_dataset = tf.data.Dataset.from_tensor_slices((self.char_input, self.seq_len, self.label))
        test_dataset = test_dataset.batch(self.batch_size)
        test_dataset = test_dataset.repeat()

        iters = tf.data.Iterator.from_structure(test_dataset.output_types, test_dataset.output_shapes)        
        self.iter_char_input, self.iter_seq_len, self.iter_label = iters.get_next()
        # create the initialisation operations
        self.test_init_op = iters.make_initializer(test_dataset)        
        ## Build graph
        encoder_output, self.D_model = self.get_model(hyp_args)
        self.build_model(encoder_output, self.D_model)
        
    def infer(self):
        num_test_batch = int(self.test_size / self.batch_size) + 1
        
        test_feed_dict = {self.char_input: self.test_char_idx, self.seq_len: self.test_len, self.label: self.test_Y}
        
        modelpath = f"./ner-model-{self.test_type}/"
        modelName = f"ner-{self.test_type}.ckpt"
        best_f1_score = 0.
        saver = tf.train.Saver() 
        
        with tf.Session(config=config) as sess:
            
            sess.run(tf.global_variables_initializer())
            ## Load pretrained model
            ckpt = tf.train.get_checkpoint_state(modelpath)         
            if(ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path)):
                saver.restore(sess, modelpath + modelName)
                print("Model loaded!")

            start_time = datetime.datetime.now()
            test_f1_list, test_recall_list, test_precision_list = [], [], []
            
            print(self.data.label2idx)

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

            print("[test] loss {:.6f} f1 {:.4f} recall {:.4f} precision {:.4f}".format(test_loss, test_f1_score, test_recall_score, test_precision_score))
            print(classification_report(y_true_list, y_pred_list, digits=4))

            elapsed_time = datetime.datetime.now() - start_time
            print("{}".format(elapsed_time))
    
    def get_accuracy(self, y_true, y_pred):
        return np.sum(np.equal(y_true, y_pred))/ len(y_true)
    
    def get_model(self, hyp_args):
        print("Getting pretrained model!")
        encoder_output, D_model = build_graph(hyp_args, self.iter_char_input, self.iter_seq_len)
        return encoder_output, D_model
           
    def build_embed(self, encoder_output):        
        self.masks = tf.sequence_mask(lengths=self.iter_seq_len, maxlen=self.max_word_len, dtype=tf.float32) ## batch_size, max_word_len        
        encoder_output *= tf.expand_dims(self.masks, -1) ## zeros out masked positions
        return tf.nn.dropout(encoder_output, 1.0 - self.dropout)
    
    def build_model(self, encoder_output, D_model):       
        print("Building NER model!") 
        self.model_dim = D_model.hidden_dim
        word_emb = self.build_embed(encoder_output) ## batch_size, word_len, dim

        with tf.variable_scope("NER-task", initializer=tf.contrib.layers.xavier_initializer()):
            output = self.layer_norm(word_emb)   
            for layer_num in range(2): 
                with tf.variable_scope("Bi-LSTM_{}".format(layer_num), reuse=False):
                    with tf.variable_scope("forward", reuse = False):            
                        self.lstm_fw_cell = tf.nn.rnn_cell.LSTMCell(self.model_dim, forget_bias=1.0, state_is_tuple=True)
                        self.lstm_fw_cell = tf.contrib.rnn.DropoutWrapper(self.lstm_fw_cell, output_keep_prob = 1. - self.dropout)
                    with tf.variable_scope("backward", reuse = False):            
                        self.lstm_bw_cell = tf.nn.rnn_cell.LSTMCell(self.model_dim, forget_bias=1.0, state_is_tuple=True)
                        self.lstm_bw_cell = tf.contrib.rnn.DropoutWrapper(self.lstm_bw_cell, output_keep_prob = 1. - self.dropout)                
                    
                    outputs, states = tf.nn.bidirectional_dynamic_rnn(self.lstm_fw_cell, 
                                                                      self.lstm_bw_cell,
                                                                      dtype=tf.float32,
                                                                      inputs=output, 
                                                                      sequence_length=self.iter_seq_len)
                    output = tf.concat(outputs, 2)  ## batch_size, word_len, 2*dim 
            self.logits = tf.layers.dense(output, self.num_class) ## batch_size, word_len, num_class 
            self.output_sequence = tf.argmax(self.logits, axis=-1, output_type=tf.int32) ## batch_size, word_len 
            self.cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.iter_label, logits=self.logits)
            self.loss = tf.reduce_sum(self.cross_entropy * self.masks) / (tf.reduce_sum(self.masks) + 1e-10)

    def convert_idx_to_name(self, y_true, y_pred, lens):
        """Convert label index to name.
        Args:
            y_true (list): true label index list.
            y_pred (list): predicted label index list.
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
        result_true, result_pred = [], []
        
        for row_true, row_pred, l in zip(y_true, y_pred, lens):
            sub_true, sub_pred = [], []
            for idx in range(l):
                sub_true.append(self.data.idx2label[row_true[idx]])
                sub_pred.append(self.data.idx2label[row_pred[idx]])
            
            result_true.append(sub_true)
            result_pred.append(sub_pred)

        return result_true, result_pred
    
    def predict(self, sess):        
        y_pred, y_true, seq_len = sess.run([self.output_sequence, self.iter_label, self.iter_seq_len],
                                            feed_dict={self.dropout:0.0})
        y_true, y_pred = self.convert_idx_to_name(y_true, y_pred, seq_len)
        return y_true, y_pred

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", "-c", help="path of config file", required=True)
    parser.add_argument("--input_path", "-i", help="path of input file", required=True)
    args = parser.parse_args()

    hyp_args = yaml.load(open(args.config_path))
    print('========================')
    for key,value in hyp_args.items():
        print('{} : {}'.format(key, value))
    print('========================')
    ## Build model
    model = NER_Model(hyp_args, args.input_path)
    ## infer model
    model.infer()
