import codecs;
import numpy as np
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional, TimeDistributed
from keras.callbacks import ModelCheckpoint
import keras;
import pickle;


with open('char_to_int.pickle', 'rb') as f:
    char_to_int = pickle.load(f);

with open('int_to_char.pickle', 'rb') as f:
    int_to_char = pickle.load(f);

with open('encoded_array.pickle', 'rb') as f:
    encoded_array = pickle.load(f);


n_vocab = len(char_to_int);
n_chars = len(encoded_array);

dataX = [];
dataY = [];
sequence_length = 100;

def train_neural_network():
    
    for i in range(0, n_chars - sequence_length, 1):
            seq_in = encoded_array[i:i + sequence_length]
            seq_out = encoded_array[i + sequence_length]
            dataX.append(seq_in)
            dataY.append(seq_out)

    n_patterns = len(dataX);
    print("Total Patterns: ", n_patterns);
    
    # reshape X to be [samples, time steps, features]
    X = np.reshape(dataX, (n_patterns, sequence_length, 1));

    # normalize
    X = X / float(n_vocab);

    y = keras.utils.to_categorical(dataY);
    print(X.shape[0], X.shape[1], X.shape[2]);
    
    model = Sequential();
    model.add(Bidirectional(LSTM(256, return_sequences=True), input_shape=(X.shape[1], X.shape[2])));
    model.add(Dropout(0.2));
    model.add(Bidirectional(LSTM(256)));
    model.add(Dropout(0.2));

##    single directional lstm model
##    model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2]), return_sequences=True));
##    model.add(Dropout(0.2))
##    model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2])));
##    model.add(Dropout(0.2))

    model.add(Dense(y.shape[1], activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam')

    tbCallBack = keras.callbacks.TensorBoard(log_dir='.\logs', histogram_freq=0, write_graph=True, write_images=True);
    
    filepath="checkpoints\weights-improvement-{epoch:02d}-{loss:.4f}-bigger.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
    callbacks_list = [checkpoint, tbCallBack]
    # fit the model
    print('training neural network');
    model.fit(X, y, epochs=50, batch_size=64, callbacks=callbacks_list)

train_neural_network();
