'''Author: Christina Hitzl'''

import pandas as pd
import operator
import re

from preprocessing.stopwords import neg_stopwords, stopwords, negations

class EmotionDict:
    def __init__(self, path_to_lexicon):

        self.path = path_to_lexicon

        self.lexicon = pd.read_csv(self.path, names=["word", "emotion", "association"], skiprows=1, sep='\t')
        self.lexicon = self.lexicon.pivot(index='word', columns='emotion', values='association').reset_index()
        self.emotions = ["anger", "joy", "fear", "shame", "disgust", "guilt", "sadness"]

        # weights to change and compare 
        self.pos_weight_dict = dict(zip(self.emotions, [1,1,1,1,1,1,1]))
        self.neg_weight_dict = dict(zip(self.emotions, [-1, 1, -1, -1, -1, -1, -1]))

        self.scope = 4
        self.last_neg_idx = - self.scope -1
        self.current_score_dict = dict(zip(self.emotions, len(self.emotions) * [0]))
        self.not_found_words = {}
        self.words = 0
        self.found_words = 0
        self.nf_words = 0


    def classify_sent(self, sent):
        # initialize:
        self.last_neg_idx = - self.scope -1
        self.current_score_dict = dict(zip(self.emotions, len(self.emotions) * [0]))

        for word_idx, word in enumerate(sent):
            self.words += 1
            self.handle_word(word, word_idx)
            

        sorted_scores = sorted(self.current_score_dict.items(), key=operator.itemgetter(1), reverse=True)
        #print(sorted_scores)
        if sorted_scores[0][1] != sorted_scores[1][1]:
            return sorted_scores[0][0]
        return None

    def emotion(self, tokenized_instances):
        # sentences is a list of lists
        y_pred = []
        for tokenized_instance in tokenized_instances:
            emotion_label = self.classify_sent(tokenized_instance.tokens)
            y_pred.append(emotion_label)

        return y_pred

    def update_neg_idx(self, word_idx):
        if self.last_neg_idx + self.scope >= word_idx:
            self.last_neg_idx = - (self.scope + 1)
        else:
            self.last_neg_idx = word_idx

    def update_score_dict(self, word, word_idx):
        #all columns in which word == word
        if word in self.not_found_words:
            self.not_found_words[word] += 1
            self.nf_words += 1
            return

        found_entries = self.lexicon[self.lexicon.word == word] # used pandas 
        if found_entries.shape[0] != 1: # is this word in the lexicon?
            if found_entries.shape[0] == 0:
                self.nf_words += 1
                self.not_found_words[word] = 1
                return

        self.found_words += 1
        # which emotions can be seen in the row:
        set_emotions = [c for c in found_entries.columns[1:] if found_entries[c].to_list() == [1]]
        # just consider emotions which occur in dataset and lexicon
        intersect = list(set(set_emotions) & set(self.emotions))

        if intersect:
            #if idx + scope >= actual idx -> negation has to be considered
            #print(self.last_neg_idx + self.scope, word_idx)
            if self.last_neg_idx + self.scope >= word_idx:
                for valid_emotion in intersect:
                    self.current_score_dict[valid_emotion] += self.neg_weight_dict[valid_emotion]

            else:
                for valid_emotion in intersect:
                    self.current_score_dict[valid_emotion] += self.pos_weight_dict[valid_emotion]

    def handle_word (self, word, word_idx):
        if word in negations or word in neg_stopwords:
            #adapt last seen idx of "negation word"
            self.update_neg_idx(word_idx)
        self.update_score_dict(word, word_idx)