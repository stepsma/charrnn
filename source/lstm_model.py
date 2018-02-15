import tensorflow as tf;
import numpy as np;

def build_inputs(n_seqs, n_steps):

    print('create placeholder for input, output, and keep probability');
    with tf.name_scope('inputs'):
        inputs = tf.placeholder(tf.int32, shape=(n_seqs, n_steps), name='inputs');
        outputs = tf.placeholder(tf.int32, shape=(n_seqs, n_steps), name='targets');
        keep_prob = tf.placeholder(tf.float32, name='keep_prob');
    return inputs, outputs, keep_prob;


def build_lstm(num_cells, num_layers, batch_size, keep_prob):

    print('build multiple layer lstm');
    stack_drop = [];
    for i in range(num_layers):
        lstm = tf.contrib.rnn.BasicLSTMCell(num_cells)
        drop = tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob=keep_prob)
        stack_drop.append(drop)

    cell = tf.contrib.rnn.MultiRNNCell(stack_drop, state_is_tuple = True)
    initial_state = cell.zero_state(batch_size, tf.float32)
    return cell, initial_state


def fully_connected(lstm_out, in_size, out_size):

    print('format lstm output before feeding into fully connected layer');
    seq_output = tf.concat(lstm_out, 1) # important
    input_x = tf.reshape(seq_output, [-1, in_size]);

    print('define fully connected layer');
    with tf.name_scope('softmax'):
        W = tf.Variable(tf.truncated_normal([in_size, out_size], stddev=0.1));
        b = tf.Variable(tf.zeros(out_size));
    logits = tf.matmul(input_x, W) + b;

##    logits = tf.layers.dense(input_x, out_size, kernel_regularizer=tf.contrib.layers.l2_regularizer);
    out = tf.nn.softmax(logits, name='predictions');
    return out, logits;

def build_loss(logits, targets, lstm_size, num_classes):

    print('convert y to onehot');
    y_one_hot = tf.one_hot(targets, num_classes);
    y_reshaped = tf.reshape(y_one_hot, logits.get_shape());
##    y_reshaped = tf.reshape(y_one_hot, [-1, out_size]);

    print('build loss');
    loss = tf.losses.softmax_cross_entropy(onehot_labels=y_reshaped, logits=logits);
    loss = tf.reduce_mean(loss);
    return loss;

def build_optimizer(loss, learning_rate, grad_clip):

    print('build optimizer');
    tvars = tf.trainable_variables();
    grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars), grad_clip);
    train_op = tf.train.AdamOptimizer(learning_rate);
    optimizer = train_op.apply_gradients(zip(grads, tvars));
    return optimizer;
