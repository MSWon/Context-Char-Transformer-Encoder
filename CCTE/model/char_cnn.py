import tensorflow as tf


def char_cnn(input_, kernel_width, kernel_depth, hidden_size, highway_layers, max_word_len, max_char_len, char_emb_dim):
    '''
    :input_: input float tensor of shape  [batch_size, max_word_len, max_char_len, char_embed_size]
    :kernel_width: list of kernel widths (parallel to kernel_depth) (ex: [1, 2, 3, 4, 5, 6])
    :kernel_depth: list of kernel depths (parallel to kernel_width) (ex: [25, 50, 75, 100, 125, 150])
    '''
    assert len(kernel_width) == len(kernel_depth), 'kernel_width and kernel_depth must have the same size'

    input_ = tf.reshape(input_, [-1, max_char_len, char_emb_dim])  # [batch_size * max_word_len, max_char_len, char_emb_dim]
    input_ = tf.expand_dims(input_, 1)  # input_: [batch_size * max_word_len, 1, max_char_len, char_emb_dim]

    layers = []
    with tf.variable_scope('char_cnn'):
        for kernel_size, kernel_feature_size in zip(kernel_width, kernel_depth):
            reduced_length = max_char_len - kernel_size + 1

            # [batch_size * max_word_len, 1, reduced_length, kernel_feature_size]
            conv = conv2d(input_, kernel_feature_size, 1, kernel_size, name="kernel_%d" % kernel_size)

            # [batch_size * max_word_len, 1, 1, kernel_feature_size]
            pool = tf.nn.max_pool(tf.tanh(conv), [1, 1, reduced_length, 1], [1, 1, 1, 1], 'VALID')

            layers.append(tf.squeeze(pool, [1, 2]))

        if len(kernel_width) > 1:
            output = tf.concat(layers, 1)  # [batch_size * max_word_len, sum(kernel_depth)]
        else:
            output = layers[0]

        # [batch_size, max_word_len, sum(kernel_depth)]
        output = tf.reshape(output, (-1, max_word_len, sum(kernel_depth)))
        output = highway(output, output.get_shape()[-1], num_layers=highway_layers)
        output = tf.layers.dense(output, hidden_size, activation=None)  ## projection layer

    return output


def conv2d(input_, output_dim, k_h, k_w, name="conv2d"):
    with tf.variable_scope(name):
        w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim])
        b = tf.get_variable('b', [output_dim])
    return tf.nn.conv2d(input_, w, strides=[1, 1, 1, 1], padding='VALID') + b


def highway(input_, size, num_layers=1, scope='Highway'):
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
