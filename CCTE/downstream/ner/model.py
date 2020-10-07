import tensorflow as tf
import pandas as pd
import datetime
import os
import numpy as np
from seqeval.metrics import f1_score, recall_score, precision_score, classification_report
from .model_utils import build_graph, restore, _get_layer_lrs, AdamWeightDecayOptimizer, LayerNormalization
from .data_utils import Data

os.environ['CUDA_VISIBLE_DEVICES'] = "0"
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

class NER_Model():
    def __init__(self, hyp_args):
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

        self.train_char_idx, self.train_len, self.train_Y = self.data.load_data(os.path.join(self.data_path, f"train_{self.train_type}.txt"))
        self.test_char_idx, self.test_len, self.test_Y = self.data.load_data(os.path.join(self.data_path, f"dev_{self.test_type}.txt"))    
        self.train_size, self.test_size = len(self.train_Y), len(self.test_Y)
        self.num_train_steps = (int(self.train_size / self.batch_size) + 1) * self.training_epochs
        
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
        encoder_output, self.D_model = self.get_model(hyp_args)
        self.build_model(encoder_output, self.D_model)
        self.build_optimizer(self.num_train_steps)
        
    def train(self):
        num_train_batch = int(self.train_size / self.batch_size) + 1
        num_test_batch = int(self.test_size / self.batch_size) + 1
        
        train_feed_dict = {self.char_input: self.train_char_idx, self.seq_len: self.train_len, self.label: self.train_Y}        
        test_feed_dict = {self.char_input: self.test_char_idx, self.seq_len: self.test_len, self.label: self.test_Y}
        
        modelpath = f"./ner-model-{self.test_type}/"
        modelName = f"ner-{self.test_type}.ckpt"
        best_f1_score = 0.
        saver = tf.train.Saver() 
        
        with tf.Session(config=config) as sess:
            
            sess.run(tf.global_variables_initializer())
            ## Load pretrained model
            restore(sess, self.pretrained_model_path)
            
            if(not os.path.exists(modelpath)):
                os.mkdir(modelpath)
            ckpt = tf.train.get_checkpoint_state(modelpath)
            
            if(ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path)):
                saver.restore(sess, modelpath + modelName)
                print("Model loaded!")
                        
            sess.run(self.train_init_op, feed_dict = train_feed_dict) 
            start_time = datetime.datetime.now()
            
            train_loss_list, test_loss_list = [], []
            test_f1_list, test_recall_list, test_precision_list = [], [], []
            
            print(self.data.label2idx)
            print("start training")
            
            for epoch in range(self.training_epochs):
                
                train_loss = 0.           
                for step in range(num_train_batch):
                    loss, _ = sess.run([self.loss, self.train_op],
                                        feed_dict={self.dropout: 0.1})               
                    train_loss += loss/num_train_batch
                    print("epoch {:02d} step {:04d} loss {:.6f}".format(epoch+1, step+1, loss))
                                        
                print("Now for test data")
                print("Could take few minutes")
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
                print(classification_report(y_true_list, y_pred_list, digits=4))
                train_loss_list.append(train_loss)
                test_loss_list.append(test_loss)
                test_f1_list.append(test_f1_score)
                test_recall_list.append(test_recall_score)
                test_precision_list.append(test_precision_score)
                sess.run(self.train_init_op, feed_dict = train_feed_dict)
                
                if(best_f1_score <= test_f1_score):
                    best_f1_score = test_f1_score
                    saver.save(sess, modelpath + modelName)
                    with open(f"./table_ner_{self.test_type}.txt", "w") as f:
                        f.write(classification_report(y_true_list, y_pred_list, digits=4))

            result = pd.DataFrame({"train_loss":train_loss_list,
                                   "test_loss":test_loss_list,
                                   "test_f1":test_f1_list,
                                   "test_recall":test_recall_list ,
                                   "test_precision": test_precision_list})

            result.to_csv(f"./loss_ner_{self.test_type}.csv", sep =",", index=False)
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

    def build_optimizer(self, num_train_steps):                
        global_step = tf.train.get_or_create_global_step()

        learning_rate = 1e-4
        warmup_proportion = 0.1
        layerwise_lr_decay = 0.8
        warmup_steps = 0.0
        n_transformer_layers = self.D_num_layers

        # define optimizer
        learning_rate = tf.train.polynomial_decay(
            learning_rate,
            global_step,
            num_train_steps,
            end_learning_rate=0.0,
            power=1.0,
            cycle=False)
        
        warmup_steps = max(num_train_steps * warmup_proportion, warmup_steps)
        learning_rate *= tf.minimum(
            1.0, tf.cast(global_step, tf.float32) / tf.cast(warmup_steps, tf.float32))

        learning_rate = _get_layer_lrs(learning_rate, layerwise_lr_decay, n_transformer_layers)

        optimizer = AdamWeightDecayOptimizer(
                        learning_rate=learning_rate,
                        weight_decay_rate=0.01,
                        beta_1=0.9,
                        beta_2=0.999,
                        epsilon=1e-6,
                        exclude_from_weight_decay=["LayerNorm", "layer_norm", "bias"])

        tvars = tf.trainable_variables()
        grads = tf.gradients(self.loss, tvars)
        (grads, _) = tf.clip_by_global_norm(grads, clip_norm=1.0)
        train_op = optimizer.apply_gradients(zip(grads, tvars), global_step=global_step)
        new_global_step = global_step + 1
        self.train_op = tf.group(train_op, [global_step.assign(new_global_step)])

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
