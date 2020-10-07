import tensorflow as tf
import numpy as np
import random

_buffer_size = 200000
_bucket_size = 10
_thread_num = 16

MAX_CHAR_LEN = 20
MAX_WORD_LEN = 128

def get_vocab_word2idx(vocab_path):
    vocab_path_tensor = tf.constant(vocab_path)
    tf.add_to_collection(tf.GraphKeys.ASSET_FILEPATHS, vocab_path_tensor)
    vocab_dict = tf.contrib.lookup.index_table_from_file(
        vocabulary_file=vocab_path_tensor,
        num_oov_buckets=0,
        default_value=1)
    return vocab_dict

def get_vocab_idx2word(vocab_path):
    vocab_path_tensor = tf.constant(vocab_path)
    tf.add_to_collection(tf.GraphKeys.ASSET_FILEPATHS, vocab_path_tensor)
    vocab_dict = tf.contrib.lookup.index_to_string_table_from_file(
        vocabulary_file=vocab_path_tensor,
        default_value='<unk>')
    return vocab_dict

def tensor_scatter_update(x, indices, updates):
    x_shape = tf.shape(x)
    patch = tf.scatter_nd(indices, updates, x_shape)
    mask = tf.greater(tf.scatter_nd(indices, tf.ones_like(updates), x_shape), 0)
    return tf.where(mask, patch, x)

def scatter_mask_update(tensor, indices, mask_idx):
    updates = tf.fill(tf.shape(indices), mask_idx)
    indices = tf.reshape(indices, [-1, 1])
    return tensor_scatter_update(tensor, indices, updates)

def get_mask_position(x, max_len, mask_idx):
    org_input_bpe = x["org_input_bpe"]
    org_input_word = x["org_input_word"]
    org_space_bool = x["org_space_bool"]

    bpe_seq_len = tf.shape(org_input_bpe)[0]
    word_seq_len = tf.shape(org_input_word)[0]
    max_mask_len = round(max_len * 0.15)

    # Sample 15% of the word
    sample_num_real = tf.to_int32(tf.round(tf.multiply(tf.to_float(bpe_seq_len), 0.15)))
    idx_real = tf.range(1, bpe_seq_len)
    real_mask = tf.random_shuffle(idx_real)[:sample_num_real]

    sample_num_pad = max_mask_len - sample_num_real
    idx_pad = tf.range(bpe_seq_len, max_len)
    pad_mask = tf.random_shuffle(idx_pad)[:sample_num_pad]

    mask_position = tf.concat([real_mask, pad_mask], axis=0)
    # Mask 15% of the word
    masked_input_idx = scatter_mask_update(org_input_bpe, real_mask, mask_idx)
    weight_label = tf.concat([tf.ones_like(real_mask), tf.zeros_like(pad_mask)], axis=0)

    output_idx = tf.gather(org_input_bpe, real_mask)

    result_dic = {
        "org_input_bpe": org_input_bpe,
        "org_input_word": org_input_word,
        "org_space_bool": org_space_bool,
        "masked_input_idx": masked_input_idx,
        "output_idx" : output_idx,
        "bpe_seq_len": bpe_seq_len,
        "word_seq_len": word_seq_len,
        "mask_position": mask_position,
        "weight_label": weight_label
    }
    return result_dic

def bpe2word(tensor):
    joined_tensor = tf.reduce_join(tensor)
    replaced_tensor = tf.regex_replace(joined_tensor, "▁", " ")
    return tf.string_split([replaced_tensor]).values

def _bpe2word_with_pad(tensor):
    tensor = tf.regex_replace(tensor, "<pad>", "▁⍷")
    joined_tensor = tf.reduce_join(tensor)
    replaced_tensor = tf.regex_replace(joined_tensor, "▁", " ")
    word_tensor = tf.string_split([replaced_tensor]).values
    pad = tf.tile(tf.constant(["⍷"]), [MAX_WORD_LEN - tf.shape(word_tensor)[0]])
    return tf.concat([word_tensor, pad], axis=-1)

@tf.contrib.eager.defun
def bpe2word_with_pad(bpe_tensor):
    return tf.map_fn(_bpe2word_with_pad, bpe_tensor, parallel_iterations=_thread_num)

def _bpe2word2char(bpe_tensor):
    word_tensor = _bpe2word_with_pad(bpe_tensor)
    char_line = tf.string_split(word_tensor, delimiter='')
    char_line = tf.sparse_tensor_to_dense(char_line, default_value='⍷')
    # Truncate to max_char_len
    char_line = tf.cond(tf.greater(tf.shape(char_line)[1], MAX_CHAR_LEN), 
                             lambda: char_line[:, :MAX_CHAR_LEN],
                             lambda: char_line)
    # pad with max_char_len
    paddings = [[0, MAX_WORD_LEN - tf.shape(char_line)[0]], [0, MAX_CHAR_LEN - tf.shape(char_line)[1]]]
    padded_char_line = tf.pad(char_line, paddings, 'CONSTANT', constant_values='⍷')
    return padded_char_line

@tf.contrib.eager.defun
def bpe2word2char(bpe_tensor):
    return tf.map_fn(_bpe2word2char, bpe_tensor, parallel_iterations=_thread_num)

def convert_bpe_spaces(org_space_bool, D_input_space_bool, org_input_bpe, D_input_bpe):
    """
    org_space_bool: batch_size, seq_len
    D_input_space_bool: batch_size, seq_len
    org_input_bpe: batch_size, seq_len
    D_input_bpe: batch_size, seq_len
    """
    diff = org_space_bool - D_input_space_bool
    # Define mask for add & rm bpe space
    add_mask = tf.equal(diff, 1)
    rm_mask = tf.equal(diff, -1)
    add_D = tf.constant([["▁"]]) + D_input_bpe
    rm_D = tf.strings.regex_replace(D_input_bpe, "▁", "")
    # Convert using tf.where
    added_D = tf.where(add_mask, add_D, D_input_bpe)
    removed_added_D = tf.where(rm_mask, rm_D, added_D)
    return removed_added_D

def pad(tensor, max_len):
    paddings = [[0, 0], [0, max_len-tf.shape(tensor)[0]]]
    return tf.pad(sample, paddings, 'CONSTANT', constant_values='<pad>')

def train_dataset_fn(corpus_path, word_vocab_path, max_len, mask_idx, batch_size):

  with tf.device("/cpu:0"):
      dataset = tf.data.TextLineDataset(corpus_path)
      tf_vocab = get_vocab_word2idx(word_vocab_path)

      dataset = dataset.apply(tf.data.experimental.shuffle_and_repeat(_buffer_size))
      # Split
      dataset = dataset.map(
            lambda line:
                tf.string_split([line]).values,
                num_parallel_calls=_thread_num
      )
      # Truncate to max_len
      dataset = dataset.map(
          lambda x:
                tf.cond(tf.greater(tf.shape(x)[0], max_len),
                        lambda: x[:max_len],
                        lambda: x
                        )
      )
      # convert bpe2word && lookup word2idx && 
      dataset = dataset.map(
          lambda x: {
            "org_input_bpe": tf_vocab.lookup(x),
            "org_input_word": bpe2word(x),
            "org_space_bool": tf.cast(tf.strings.regex_full_match(x, '▁.*'), dtype=tf.int32)
          },
          num_parallel_calls=_thread_num
      )
      # Cast to tf.int32
      dataset = dataset.map(lambda x: {
          "org_input_bpe": tf.cast(x["org_input_bpe"], tf.int32),
          "org_input_word": x["org_input_word"],
          "org_space_bool": x["org_space_bool"]
        }
      )
      # Get mask position
      dataset = dataset.map(
          lambda x:
            get_mask_position(x, max_len, mask_idx),
            num_parallel_calls=_thread_num
      )
      # Padding
      dataset = dataset.padded_batch(
          batch_size,
          {
              "org_input_bpe": [max_len],
              "org_input_word": [max_len],
              "org_space_bool": [max_len],
              "masked_input_idx": [max_len],
              "output_idx": [round(max_len * 0.15)],
              "bpe_seq_len": [],
              "word_seq_len": [],
              "mask_position": [round(max_len * 0.15)],
              "weight_label": [round(max_len * 0.15)]
          },
          {
              "org_input_bpe": 0,
              "org_input_word": "<pad>",
              "org_space_bool": 0,
              "masked_input_idx": 0,
              "output_idx": 0,
              "bpe_seq_len": 0,
              "word_seq_len": 0,
              "mask_position": 0,
              "weight_label": 0
          }
      )
      # Prefetch the next element to improve speed of input pipeline.
      dataset = dataset.prefetch(3)
  return dataset

