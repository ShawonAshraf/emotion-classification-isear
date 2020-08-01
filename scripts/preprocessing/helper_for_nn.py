'''Author: Shawon Ashraf'''

import re
import os

import numpy as np

"""
Clean punctuation marks, special characters and short forms from a sentence
param s - a sentences as a string

Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
"""


def clean_str(s):
    """Clean sentence"""
    s = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", s)
    s = re.sub(r"\'s", " \'s", s)
    s = re.sub(r"\'ve", " \'ve", s)
    s = re.sub(r"n\'t", " n\'t", s)
    s = re.sub(r"\'re", " \'re", s)
    s = re.sub(r"\'d", " \'d", s)
    s = re.sub(r"\'ll", " \'ll", s)
    s = re.sub(r",", " , ", s)
    s = re.sub(r"!", " ! ", s)
    s = re.sub(r"\(", " \( ", s)
    s = re.sub(r"\)", " \) ", s)
    s = re.sub(r"\?", " \? ", s)
    s = re.sub(r"\s{2,}", " ", s)
    s = re.sub(r'\S*(x{2,}|X{2,})\S*', "xxx", s)
    s = re.sub(r'[^\x00-\x7F]+', "", s)
    return s.strip().lower()


"""
load embedding weights from a pre-trained embedding set
We are using Glove embeddings here.

Takes the model name, the vocabulary dictionary, max word length and the embedding dimensions 
as params

returns word -> weight mappings as a matrix (max word length, embedding dim)

allowed embedding dims = 50, 100, 200, 300

https://github.com/lukas/ml-class/blob/master/videos/cnn-text/imdb-embedding.py
"""


def load_embedding_weights_as_matrix(vocabulary, max_len_words, embedding_dims):
    model_dir = 'models'
    model_name = f"retrofitted.txt"
    model_path = os.path.join(model_dir, model_name)

    # word -> embedding coefficient mapping
    # so that we can get weights by words
    embedding_dict = dict()

    # load from file
    if os.path.exists(model_path):
        print(f"Loading model -> {model_name}")

        with open(model_path, encoding='utf-8') as f:
            for line in f:
                values = line.split()
                word = values[0]
                coefs = np.asarray(values[1:], dtype='float32')
                embedding_dict[word] = coefs

    # if the file doesn't exist, download it
    else:
        print("Model doesn't exist, downloading .....")
        # TODO - implement a function to download the model

    # weight matrix for words
    # from word -> weight lookup
    embedding_matrix = np.zeros((max_len_words, embedding_dims))
    for word, index in vocabulary.items():
        if index > max_len_words - 1:
            break
        else:
            embedding_vector = embedding_dict.get(word)
            if embedding_vector is not None:
                embedding_matrix[index] = embedding_vector

    return embedding_matrix


"""
Get text and labels separated from a list of EmoteInstance
From the corpus
"""


def get_text_and_labels(data):
    texts = [emo.text for emo in data]
    labels = [emo.label for emo in data]

    cleaned_texts = [clean_str(s) for s in texts]

    return cleaned_texts, labels
