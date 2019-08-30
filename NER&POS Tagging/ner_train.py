from ner_model import Model
import tensorflow as tf

if __name__ == '__main__':
    flags = tf.app.flags
    FLAGS = flags.FLAGS
    
    ## Model parameter
    flags.DEFINE_integer('num_layers', 6, 'number of layers of transformer encoders')
    flags.DEFINE_integer('num_heads', 8, 'number of heads of transformer encoders')
    flags.DEFINE_integer('linear_key_dim', 512, 'dimension of key vector')
    flags.DEFINE_integer('linear_value_dim', 512, 'dimension of value vector')
    flags.DEFINE_integer('model_dim', 512, 'dimension of hidden nodes')
    flags.DEFINE_integer('ffn_dim', 2048, 'dimension of feed forward network')
    flags.DEFINE_integer('char_dim', 16, 'dimension of char')
    flags.DEFINE_integer('max_word_len', 100, 'max word length')
    flags.DEFINE_integer('max_char_len', 50, 'max char length')
    flags.DEFINE_integer('batch_size', 64, 'number of batch size')
    flags.DEFINE_float('learning_rate', 0.0001, 'learning rate')
    flags.DEFINE_integer('training_epochs', 50, 'number of training epochs')
      
    print('========================')
    for key in FLAGS.__flags.keys():
        print('{} : {}'.format(key, getattr(FLAGS, key)))
    print('========================')    
    
    model = Model(FLAGS.num_layers, FLAGS.num_heads, FLAGS.linear_key_dim, FLAGS.linear_value_dim,
                  FLAGS.model_dim, FLAGS.ffn_dim, FLAGS.char_dim, FLAGS.max_word_len, 
                  FLAGS.max_char_len, FLAGS.batch_size, FLAGS.learning_rate)
    
    ## Train model
    model.train(FLAGS.training_epochs)
