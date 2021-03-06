import numpy as np;
import tensorflow as tf;
from tqdm import tqdm
import pickle;
import copy;
import time;
from lstm_model import build_inputs, build_lstm, build_bidirectional_lstm, fully_connected, build_loss, build_optimizer;


def batch_generator(arr, n_seqs, n_steps):
    arr = copy.copy(arr)
    batch_size = n_seqs * n_steps
    n_batches = int(len(arr) / batch_size)
    arr = arr[:batch_size * n_batches]
    arr = arr.reshape((n_seqs, -1))
##    while True:
##        np.random.shuffle(arr)
    for n in range(0, arr.shape[1], n_steps):
        x = arr[:, n:n + n_steps]
        y = np.zeros_like(x)
        y[:, :-1], y[:, -1] = x[:, 1:], x[:, 0]
        yield x, y


def pick_top_n(preds, vocab_size, top_n=5):
    """
    fetch top five from result
    
    preds: prediction
    vocab_size
    top_n
    """
    p = np.squeeze(preds); # reduce uncessary dimensions
    p[np.argsort(p)[:-top_n]] = 0 # set the probability to zero for non top 5 in predict array
    p = p / np.sum(p); # normalize the probability
    c = np.random.choice(vocab_size, size=1, p=p)[0]; # randomly pickup one character, use zero to take result from array
    return c;

class CharRNN:
    
    def __init__(self, num_classes, batch_size=64, seq_length=50, 
                       lstm_size=128, num_layers=2, learning_rate=0.001, 
                       grad_clip=5, sampling=False):
    
        if sampling == True:
            batch_size, seq_length = 1, 1
        else:
            batch_size, seq_length = batch_size, seq_length

        tf.reset_default_graph()
        
        self.inputs, self.targets, self.keep_prob = build_inputs(batch_size, seq_length)

        fw_cell, bw_cell, self.fw_initial_state, self.bw_initial_state = build_bidirectional_lstm(lstm_size, num_layers, batch_size, self.keep_prob);

        x_one_hot = tf.one_hot(self.inputs, num_classes)

        sequence_length = [seq_length] * batch_size;
        
        outputs, states = tf.nn.bidirectional_dynamic_rnn(
                                                cell_fw=fw_cell,
                                                cell_bw=bw_cell,
                                                dtype=tf.float32,       
                                                sequence_length=sequence_length,
                                                inputs=x_one_hot,
                                                initial_state_fw=self.fw_initial_state,
                                                initial_state_bw=self.bw_initial_state);
        
        self.final_state_fw, self.final_state_bw = states;
        self.final_outputs_fw, self.final_outputs_bw = outputs;
        self.lstm_out = tf.concat([self.final_outputs_fw, self.final_outputs_bw], axis=-1);
        
##        self.lstm_dropout = tf.nn.dropout(self.lstm_out, keep_prob=keep_prob); ## dropout does not change shape
        
        self.prediction, self.logits = fully_connected(self.lstm_out, lstm_size * 2, num_classes)
##        
##        # Loss 和 optimizer (with gradient clipping)
        self.loss = build_loss(self.logits, self.targets, lstm_size, num_classes)
        self.optimizer = build_optimizer(self.loss, learning_rate, grad_clip)


with open('char_to_int.pickle', 'rb') as f:
    char_to_int = pickle.load(f);

with open('int_to_char.pickle', 'rb') as f:
    int_to_char = pickle.load(f);

with open('encoded_array.pickle', 'rb') as f:
    encoded_array = pickle.load(f);

batch_size = 64         # Sequences per batch
seq_length = 50          # Number of sequence steps per batch
lstm_size = 128         # Size of hidden layers in LSTMs
num_layers = 2          # Number of LSTM layers
learning_rate = 0.001    # Learning rate
keep_prob = 1.0         # Dropout keep probability
epochs = 2000;
checkpoint_step = 5000;

def train_neural_network():

    model = CharRNN(len(char_to_int),
                batch_size=batch_size,
                seq_length=seq_length,
                lstm_size=lstm_size, num_layers=num_layers, 
                learning_rate=learning_rate);

    saver = tf.train.Saver(max_to_keep=100);
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer());
##        This part is for debug purpose
##        x = np.random.randn(batch_size, seq_length);
##        y = np.random.randn(batch_size, seq_length);
##        initial_fw, initial_bw = sess.run([model.fw_initial_state, model.bw_initial_state]);
##
##        feed_dict = {model.inputs: x,
##                     model.targets: y,
##                     model.keep_prob: keep_prob,
##                     model.fw_initial_state: initial_fw,
##                     model.bw_initial_state: initial_bw}
##        lstm_out = sess.run([model.optimizer],
##                                   feed_dict=feed_dict);
##        print(lstm_out);

        counter = 0;
        # train neural network
        print('training neural network');
        for e in tqdm(range(epochs)):
            fw_state, bw_state = sess.run([model.fw_initial_state, model.bw_initial_state]);
            loss = 0
            for x, y in batch_generator(encoded_array, batch_size, seq_length):
                counter += 1
                start = time.time();
                
                feed_dict = {model.inputs: x,
                             model.targets: y,
                             model.keep_prob: keep_prob,
                             model.fw_initial_state: fw_state,
                             model.bw_initial_state: bw_state}
                
                batch_loss, fw_state, bw_state, _ = sess.run([model.loss, 
                                                     model.final_state_fw,
                                                     model.final_state_bw,
                                                     model.optimizer], 
                                                     feed_dict=feed_dict);

                end = time.time()
                # control the print lines
                if counter % checkpoint_step == 0:
                    print('epochs: {}/{}... '.format(e+1, epochs),
                          'training steps: {}... '.format(counter),
                          'training loss: {:.4f}... '.format(batch_loss),
                          '{:.4f} sec/batch'.format((end-start)))
                    saver.save(sess, "checkpoints/i{}_l{}.ckpt".format(counter, lstm_size))                    
        
        saver.save(sess, "checkpoints/i{}_l{}.ckpt".format(counter, lstm_size))
            
    tf.train.get_checkpoint_state('checkpoints');

def get_text_sample(checkpoint, text_length, lstm_size, vocab_size, prime="The "):
    """
    生成新文本
    
    checkpoint: 某一轮迭代的参数文件
    n_sample: 新闻本的字符长度
    lstm_size: 隐层结点数
    vocab_size
    prime: 起始文本
    """
    # convert string into list of characters
    samples = [c for c in prime];
    
    # sampling=True meaning batch size is 1 * 1, for text generation
    model = CharRNN(vocab_size, lstm_size=lstm_size, sampling=True);
    saver = tf.train.Saver()
    with tf.Session() as sess:
        # restore the model
        saver.restore(sess, checkpoint);
        new_state = sess.run(model.initial_state);
        for c in prime:
            x = np.zeros((1, 1))
            # give single character
            x[0,0] = char_to_int[c]
            feed = {model.inputs: x,
                    model.keep_prob: 1.,
                    model.initial_state: new_state}
            preds, new_state = sess.run([model.prediction, model.final_state], 
                                         feed_dict=feed)

        c = pick_top_n(preds, len(char_to_int))
        # 添加字符到samples中
        samples.append(int_to_char[c]);
        
        # 不断生成字符，直到达到指定数目
        for i in range(text_length):
            x[0,0] = c;
            feed = {model.inputs: x,
                    model.keep_prob: 1.,
                    model.initial_state: new_state}
            preds, new_state = sess.run([model.prediction, model.final_state], 
                                         feed_dict=feed)

            c = pick_top_n(preds, len(char_to_int));
            samples.append(int_to_char[c]);
        
    return ''.join(samples);

train_neural_network();
##checkpoint = 'checkpoints/i300000_l512.ckpt';
##sample_text = get_text_sample(checkpoint, 2000, lstm_size, len(char_to_int), prime="The new promise");
##print(sample_text);
