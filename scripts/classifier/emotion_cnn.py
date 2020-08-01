'''Author: Shawon Ashraf'''

import preprocessing.helper_for_nn as helper
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder

from tensorflow.keras.models import Sequential, save_model
from tensorflow.keras.layers import Dense, Activation, Dropout, Flatten, Embedding, Convolution1D, MaxPooling1D
from tensorflow.keras.preprocessing import text, sequence
from tensorflow.keras import utils

"""
class Emotion CNN
taken from - https://github.com/lukas/ml-class/blob/master/videos/cnn-text/imdb-embedding.py
"""


class EmotionCNN:
    """
    Allowed embedding dims = 50, 100, 200, 300
    Since we have pre-trained embeddings for these values

    n_filters - number of filters for convolutions
    filter_size - size of the filters (3, 4, 7 etc.)
    hidden_dims - number of hidden dimensions
    """

    def __init__(self, embedding_dims, n_filters, filter_size, hidden_dims, max_words):
        self.embedding_dims = embedding_dims
        self.n_filters = n_filters
        self.filter_size = filter_size
        self.hidden_dims = hidden_dims
        self.max_words = max_words

        # weight matrix, to be loaded during training
        self.embedding_weight_matrix = None

        # data
        self.x_train = None
        self.x_test = None

        # encoded for training
        self.y_train = None
        self.y_test = None

        # number of classes to predict from
        self.n_classes = None
        # name of the classes
        self.class_names = None

        # model
        self.model = None

        # model input / sequence length
        self.sequence_length = 1000

        # training history
        # to be used later for plotting
        self.history = None

    """
    load embedding weights
    call in train()
    """

    def __load_weights(self, vocabulary):
        self.embedding_weight_matrix = helper.load_embedding_weights_as_matrix(
            vocabulary=vocabulary,
            max_len_words=self.max_words,
            embedding_dims=self.embedding_dims
        )

    """
    pre process and fit data
    """

    def fit(self, corpus):
        # load train and test data from the corpus
        train_data = corpus.train_data
        test_data = corpus.test_data

        # pre-processing by helper functions
        x_train, y_train = helper.get_text_and_labels(train_data)
        x_test, y_test = helper.get_text_and_labels(test_data)

        # tokenize, pad, convert to matrix
        tokenizer = text.Tokenizer(num_words=self.max_words, char_level=False)
        tokenizer.fit_on_texts(x_train)

        x_train = tokenizer.texts_to_matrix(x_train)
        x_test = tokenizer.texts_to_matrix(x_test)

        x_train = sequence.pad_sequences(x_train, maxlen=self.sequence_length)
        x_test = sequence.pad_sequences(x_test, maxlen=self.sequence_length)

        # one hot encode labels
        encoder = LabelEncoder()
        encoder.fit(y_train)

        # update class names
        self.class_names = encoder.classes_

        y_train_encoded = encoder.transform(y_train)
        y_test_encoded = encoder.transform(y_test)

        # update
        self.n_classes = np.max(y_train_encoded) + 1

        y_train_encoded = utils.to_categorical(y_train_encoded, self.n_classes)
        y_test_encoded = utils.to_categorical(y_test_encoded, self.n_classes)

        self.x_train = x_train
        self.x_test = x_test

        self.y_train = y_train_encoded
        self.y_test = y_test_encoded

        # update weights
        self.__load_weights(vocabulary=tokenizer.word_index)

    """
    compile the model
    """

    def compile(self):
        self.model = Sequential()

        self.model.add(Embedding(self.max_words,
                                 self.embedding_dims,
                                 input_length=self.sequence_length,
                                 weights=[self.embedding_weight_matrix],
                                 trainable=False))
        self.model.add(Dropout(0.5))
        self.model.add(Convolution1D(self.n_filters,
                                     self.filter_size,
                                     padding='valid',
                                     activation='relu'))
        self.model.add(MaxPooling1D())

        self.model.add(Convolution1D(self.n_filters,
                                     self.filter_size,
                                     padding='valid',
                                     activation='relu'))
        self.model.add(MaxPooling1D())
        self.model.add(Flatten())

        self.model.add(Dense(self.hidden_dims, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(self.n_classes, activation='softmax'))

        self.model.compile(loss='categorical_crossentropy',
                           optimizer='adam',
                           metrics=['accuracy'])

    """
    train the model
    """

    def train(self, batch_size, epochs):
        print("Compiling model .....")

        self.compile()
        print("Done!\n")
        print(self.model.summary())
        print()

        print("Training data summary ::: ")
        print('x_train shape: ', self.x_train.shape)
        print('x_test shape: ', self.x_test.shape)
        print('y_train shape: ', self.y_train.shape)
        print('y_test shape: ', self.y_test.shape)
        print('n_classes: ', self.n_classes)
        print('class_names: ', self.class_names)
        print()

        print(f"Training model over {epochs} epochs")
        print(f"with {batch_size} batches")
        print()

        self.history = self.model.fit(self.x_train, self.y_train,
                                      batch_size=batch_size,
                                      epochs=epochs,
                                      verbose=1,
                                      validation_split=0.2)

    """
    predict instances
    to be passed on as
    np arrays and preprocessed using
    helper functions
    
    returns  a list of 
    the predicted emotions as strings
    """

    def predict(self, x):
        y_pred = []

        predicted = self.model.predict(np.array(x))
        for p in predicted:
            # get label string from one hot encoded
            # prediction output
            predicted_label = self.class_names[np.argmax(p)]
            y_pred.append(predicted_label)

        return y_pred

    """
    save the trained model to disk
    """

    def save_model_to_disk(self, path):
        model_file_name = os.path.join(path, f"cnn_{self.embedding_dims}_{self.max_words}")
        save_model(model_file_name)
