from .generator import Generator
from .discriminator import Discriminator
from ..utils.gpu_utils import average_gradients
from ..utils.model_utils import AdamWeightDecayOptimizer, noam_scheme
from ..utils.data_utils import tensor_scatter_update
from ..utils import data_utils
import tensorflow as tf

class Electra(object):
    """ Electra class """
    def __init__(self, hyp_args, word_vocab_path, char_vocab_path):
        self.G_model = Generator(hyp_args)
        self.D_model = Discriminator(hyp_args)
        self.G_weight = hyp_args["G_weight"]
        self.D_weight = hyp_args["D_weight"]
        self.n_gpus = hyp_args["n_gpus"]
        self.training_steps = hyp_args["training_steps"]
        self.temperature = hyp_args["temperature"]
        self.max_word_len = hyp_args["max_word_len"]
        self.word_vocab_path = word_vocab_path
        self.char_vocab_path = char_vocab_path

    def build_opt(self, features, d_model, global_step, warmup_steps=10000):
        """
        :param features: train data pipeline
        :param d_model: hidden_dim
        :param global_step: integer
        :param warmup_steps: integer
        :return: train_loss: integer
                 train_opt: optimizer
        """
        # idx2word lookup
        tf_idx_vocab = data_utils.get_vocab_idx2word(self.word_vocab_path)
        tf_char_vocab = data_utils.get_vocab_word2idx(self.char_vocab_path)
        learning_rate = 5e-4
        warmup_proportion = 0
        # define optimizer
        learning_rate = tf.train.polynomial_decay(
            learning_rate,
            global_step,
            self.training_steps,
            end_learning_rate=0.0,
            power=1.0,
            cycle=False)
        warmup_steps = max(self.training_steps * warmup_proportion, warmup_steps)
        learning_rate *= tf.minimum(
            1.0, tf.cast(global_step, tf.float32) / tf.cast(warmup_steps, tf.float32))

        opt = AdamWeightDecayOptimizer(
            learning_rate=learning_rate,
            weight_decay_rate=0.01,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-6,
            exclude_from_weight_decay=["LayerNorm", "layer_norm", "bias"])

        # Multi-GPU
        train_loss = tf.get_variable('total_loss', [],
                                     initializer=tf.constant_initializer(0.0), trainable=False)

        tower_grads = []
        total_batch = tf.shape(features['org_input_bpe'])[0]
        batch_per_gpu = total_batch // self.n_gpus

        with tf.variable_scope(tf.get_variable_scope()):
            for k in range(self.n_gpus):
                with tf.device("/gpu:{}".format(k)):
                    print("Building model tower_{}".format(k + 1))
                    print("Could take few minutes")
                    # calculate the loss for one model replica
                    start = tf.to_int32(batch_per_gpu * k)
                    end = tf.to_int32(batch_per_gpu * (k + 1))
                    org_input_bpe = features['org_input_bpe'][start:end]
                    org_input_word = features['org_input_word'][start:end]
                    org_space_bool = features['org_space_bool'][start:end]
                    G_input_idx = features['masked_input_idx'][start:end]
                    bpe_seq_len = features['bpe_seq_len'][start:end]
                    word_seq_len = features['word_seq_len'][start:end]
                    output_idx = features['output_idx'][start:end]
                    mask_position = features['mask_position'][start:end]
                    weight_label = features['weight_label'][start:end]

                    G_logits = self.G_model.build_graph(G_input_idx, mask_position) # batch_size, mask_len, vocab_size
                    G_loss = self.G_model.build_loss(G_logits, output_idx, weight_label)
                    G_logits_ = tf.stop_gradient(tf.nn.softmax(G_logits/ self.temperature))
                    G_infer_idx = tf.argmax(G_logits_, axis=-1)   # batch_size, mask_len
                    G_infer_idx = tf.cast(G_infer_idx, tf.int32)
                    
                    G_equal = tf.cast(tf.equal(output_idx, G_infer_idx), tf.float32)
                    weight_label = tf.cast(weight_label, dtype = tf.float32)
                    G_acc = tf.reduce_sum(G_equal * weight_label) / (tf.reduce_sum(weight_label) + 1e-10)

                    indices = mask_position + tf.range(0, batch_per_gpu*tf.shape(G_input_idx)[1], tf.shape(G_input_idx)[1])[:,None]
                    indices = tf.reshape(indices, [-1,1])
                    input_idx_flatten = tf.reshape(G_input_idx, [-1])  # batch_size * seq_len
                    G_infer_idx_flatten = tf.reshape(G_infer_idx, [-1]) # batch_size * mask_len

                    D_input_bpe_idx = tensor_scatter_update(input_idx_flatten, indices, G_infer_idx_flatten) # batch_size * seq_len
                    D_input_bpe_idx = tf.reshape(D_input_bpe_idx, [-1, tf.shape(G_input_idx)[1]])  # batch_size , seq_len
                    mask = tf.cast(tf.sequence_mask(bpe_seq_len, maxlen=self.max_word_len), tf.int32)  # batch_size , seq_len
                    D_input_bpe_idx *= mask
                    D_input_bpe = tf_idx_vocab.lookup(tf.cast(D_input_bpe_idx, tf.int64))  # batch_size , seq_len
                    # convert bpe to org
                    D_input_space_bool = tf.cast(tf.strings.regex_full_match(D_input_bpe, '▁.*'), dtype=tf.int32)  # batch_size , seq_len
                    D_input_bpe = data_utils.convert_bpe_spaces(org_space_bool, D_input_space_bool, org_input_bpe, D_input_bpe)
                    # convert bpe to char idx
                    D_input_word = data_utils.bpe2word_with_pad(D_input_bpe)
                    D_input_char = data_utils.bpe2word2char(D_input_bpe)
                    D_input_char_idx = tf_char_vocab.lookup(D_input_char)
                    char_mask = tf.cast(tf.sequence_mask(word_seq_len, maxlen=self.max_word_len), tf.int64) # batch_size , seq_len
                    D_input_char_idx *= tf.expand_dims(char_mask, axis=-1)
                    # build char-cnn graph
                    D_logits = self.D_model.build_graph(D_input_char_idx, word_seq_len)
                    D_infer = tf.cast(tf.nn.sigmoid(D_logits) >= 0.5, tf.int32)
                    # For clearing zero pad
                    # self.clear_char_emb_padding = self.D_model.clear_char_emb_padding
                    D_labels = tf.cast(tf.equal(org_input_word, D_input_word), tf.int32)

                    D_loss = self.D_model.build_loss(D_logits, D_labels, word_seq_len)
                    D_acc = self.D_model.build_accuracy(D_infer, D_labels, word_seq_len)

                    loss = self.G_weight * G_loss + self.D_weight * D_loss
                    # Reuse variables for the next tower.
                    tf.get_variable_scope().reuse_variables()
                    grads_and_vars = opt.compute_gradients(loss)
                    grads_and_vars = [(tf.clip_by_norm(grad, clip_norm=1.0), var) for (grad, var) in grads_and_vars]
                    tower_grads.append(grads_and_vars)
                    train_loss += loss / self.n_gpus

        grads = average_gradients(tower_grads)
        train_opt = opt.apply_gradients(grads, global_step=global_step)
        new_global_step = global_step + 1
        train_opt = tf.group(train_opt, [global_step.assign(new_global_step)])
        return train_loss, G_loss, G_acc, D_loss, D_acc, train_opt
