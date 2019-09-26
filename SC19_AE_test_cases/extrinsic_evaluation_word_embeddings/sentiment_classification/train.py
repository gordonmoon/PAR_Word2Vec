'''Trains an LSTM model on the IMDB sentiment classification task.
Adapted from 
'''
from __future__ import print_function

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import LSTM
from keras.datasets import imdb
from collections import defaultdict
import keras
import numpy as np
import os
import sys
import util

if __name__ == '__main__':

    max_features = 20000
    maxlen = 80  # cut texts after this number of words (among top max_features most common words)
    batch_size = 128
    
    embeddings_file = sys.argv[1]

    embeddings_index = util.load_embeddings_dict(embeddings_file, words=None)
    embedding_dim = list(embeddings_index.values())[0].shape[0]
    print(embedding_dim)

    index_dict = keras.datasets.imdb.get_word_index()
    n_vocab = len(index_dict) + 2
    oov_count = 0
    embedding_weights = np.zeros((n_vocab, embedding_dim))
    for word, index in index_dict.items():
        word = word.lower()
        if word in embeddings_index:
            embedding_weights[index,:] = embeddings_index[word]
        else:
            oov_count += 1
            embedding_weights[index,:] = embeddings_index['UNKNOWN_TOKEN']

    print('Loading data...')
    (x_train, y_train), (x_test, y_test) = imdb.load_data()
    print(len(x_train), 'train sequences')
    print(len(x_test), 'test sequences')

    print('Pad sequences (samples x time)')
    x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
    x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
    print('x_train shape:', x_train.shape)
    print('x_test shape:', x_test.shape)

    print('Build model...')
    model = Sequential()
    model.add(Embedding(n_vocab, embedding_dim, weights=[embedding_weights],trainable=False))
    model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(1, activation='sigmoid'))

    # try using different optimizers and different optimizer configs
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    model.summary()
    print('Train...')
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=20)
    score, acc = model.evaluate(x_test, y_test,
                                batch_size=batch_size)
    print('Test score:', score)
    print('Test accuracy:', acc)
