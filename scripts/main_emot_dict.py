'''Author: Christina Hitzl'''

import pandas as pd
from preprocessing.preprocessing import Preprocessor
from preprocessing.helper_for_nn import clean_str
from preprocessing.tokenizer import tokenize
from corpus.corpus import Corpus
from classifier.emotion_dict import EmotionDict
import evaluation.evaluation as evl
import operator

def filtering(y_true, y_pred):
    y_pred_new = []
    y_true_new = []
    
    for idx, y in enumerate(y_pred):
        if not y:
            continue

        y_pred_new.append(y)
        y_true_new.append(y_true[idx])

    return y_pred_new, y_true_new

if __name__ == "__main__":


    path_to_gold = '../data/isear/isear-val.csv'
    path_to_pred = '../data/isear/isear-val-prediction.csv'

    path_to_train = '../data/isear/isear-train.csv'
    path_to_test = '../data/isear/isear-test.csv'

    filepath = '../data/NRC/NRC-Emotion-Lexicon-Wordlevel-v0.92.txt'

    corpus = Corpus(path_to_train, path_to_test, path_to_gold, path_to_pred)

    # load from corpus
    y_true = [emo.label for emo in corpus.test_data]

    # Preprocessing step
    p = Preprocessor(corpus.test_data, y_true)
    
    tokenized_instances = p.get_clean_tokenized_sentences()

     
    #EmotionDict
    emo_d = EmotionDict(filepath)
    y_pred = emo_d.emotion(tokenized_instances)
    y_pred_new, y_test_new = filtering(y_true, y_pred)


    sorted_x = sorted(emo_d.not_found_words.items(), key=operator.itemgetter(1))
    #print(sorted_x)

    # Evaluator
    evaluator = evl.Evaluator(emo_d.emotions, y_test_new, y_pred_new)
    f_score, precision, recall, macro, micro = evaluator.evaluate()
    fpc = evaluator.get_f_score_per_class()
    print(fpc)
    print(f_score, precision, recall)
