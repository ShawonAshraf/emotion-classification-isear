'''Authors: Shawon Ashraf and Christina Hitzl'''


"""

implementation of bag of words
each sentence represented as a frequency vector

"""
import preprocessing.tokenizer as tk
from .helper_for_nn import clean_str
from .stopwords import stopwords
"""
class DataInstance
For holding pre-processed data instances for training or testing
params: text, label, features
"""


class DataInstance:
    def __init__(self, text, label, features):
        self.text = text
        self.label = label
        self.features = features

class TokenizedInstance:
    def __init__(self, text, label, tokens):
        self.text = text
        self.label = label
        self.tokens = tokens

"""
class Preprocessor
Processes texts from corpus to be prepared for training / testing in the classifier
params: a list containing texts or sentences and corresponding labels as another list 
"""


class Preprocessor:
    def __init__(self, sentences, labels):
        self.sentences = sentences
        self.labels = labels
        self.distinct_labels = list(set(self.labels))

        self.wordcounts_per_label = {}

    # extracts features form each sentence
    # then creates instances of DataInstance
    # returns an list containing all the DataInstances from text
    def extract_features(self):
        instances = []
        for text in self.sentences:
            tokens = tk.tokenize(text)

            # list to hold features for text
            feature_list = []

            # check for features in tokens
            # SUFFIX3, PREFIX2, WORD/W, ALL_CAPS, EXCLAMATION or if a word matches the labels, i.e. joy , anger etc.
            for idx, token in enumerate(tokens):
                # suffix3
                feature_list.append("SUFFIX3=" + token[-3:])
                # prefix2
                feature_list.append("PREFIX2=" + token[:2])
                # w
                feature_list.append("W=" + token)
                # window = 2
                feature_list.append("WORD+1=" + tokens[(idx + 1) % len(tokens)])
                feature_list.append("WORD-1=" + tokens[(idx - 1) % len(tokens)])

                feature_list.append("WORD+2=" + tokens[(idx + 2) % len(tokens)])
                feature_list.append("WORD-2=" + tokens[(idx - 2) % len(tokens)])

                if token.isupper():
                    feature_list.append('ALL_CAPS=' + token)
                if "!" in token:
                    feature_list.append('EXCLAMATION=1')
                if token in self.labels:
                    feature_list.append('LABEL=' + token)

            # create DataInstance
            index_of_text = self.sentences.index(text)
            data_instance = DataInstance(text=text, label=self.labels[index_of_text], features=feature_list)
            instances.append(data_instance)

        return instances

    
    def get_clean_tokenized_sentences(self):
        instances = []
        for idx, sent in enumerate(self.sentences):
            tokens = tk.tokenize(sent.text)
            cleaned_tokens = [clean_str(token) for token in tokens if token not in stopwords]

            instances.append(TokenizedInstance(sent.text, self.labels[idx], cleaned_tokens))
        return instances

    def get_tokenized_sentences(self):
        instances = []
        for idx, sent in enumerate(self.sentences):
            tokens = tk.tokenize(sent.text)
            cleaned_tokens = [clean_str(token) for token in tokens]

            instances.append(TokenizedInstance(sent.text, self.labels[idx], cleaned_tokens))
        return instances

    def count_labels(self):
        instances_per_emotion = dict(zip(self.distinct_labels, [self.labels.count(c) for c in self.distinct_labels]))
        return instances_per_emotion

    def count_words_in_labels(self):
        wordcount_per_labels = dict(zip(self.distinct_labels, [{} for _ in range(len(self.distinct_labels))]))

        for idx, sent in enumerate(self.sentences):
            label = self.labels[idx]

            cleared_tokens = [clean_str(token) for token in tk.tokenize(sent.text)]
            for cleared_token in cleared_tokens:
                wordcount_per_labels[label][cleared_token] = wordcount_per_labels[label].get(cleared_token, 0) + 1
        
        wordsum_per_labels = dict(zip(self.distinct_labels, [sum(wordcount_per_labels[label].values()) for label in self.distinct_labels]))
        return wordcount_per_labels, wordsum_per_labels