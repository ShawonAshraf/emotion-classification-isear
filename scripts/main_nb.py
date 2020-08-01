'''Author: Christina Hitzl'''

from classifier.naive_bayes import NaiveBayes
from preprocessing.preprocessing import Preprocessor
from preprocessing.helper_for_nn import clean_str
from preprocessing.tokenizer import tokenize
from corpus.corpus import Corpus
import evaluation.evaluation as evl

if __name__ == "__main__":
    
    path_to_gold = '../data/isear/isear-val.csv'
    path_to_pred = '../data/isear/isear-val-prediction.csv'

    path_to_train = '../data/isear/isear-train.csv'
    path_to_test = '../data/isear/isear-test.csv'

    corpus = Corpus(path_to_train, path_to_test, path_to_gold, path_to_pred)

    # load from corpus
    y_true_test = [emo.label for emo in corpus.test_data]
    y_true_train = [emo.label for emo in corpus.train_data]

    p_train = Preprocessor(corpus.train_data, y_true_train)
    p_test = Preprocessor(corpus.test_data, y_true_test)

    counts_per_label = p_train.count_labels()
    wordcount_per_labels, wordsum_per_labels = p_train.count_words_in_labels()

    nb = NaiveBayes(counts_per_label, len(p_train.sentences), wordcount_per_labels, wordsum_per_labels)

    with open(path_to_test, 'r') as test_text:
        rows = test_text.readlines()

    y_pred = nb.fit(p_test.get_tokenized_sentences())

    possible_labels = list(nb.num_labels.keys())

    # evaluator
    evaluator = evl.Evaluator(possible_labels, y_true_test, y_pred)
    f_score, precsion, recall, macro, micro = evaluator.evaluate()

    fpc = evaluator.get_f_score_per_class()
    print(fpc)

    print(f_score, precsion, recall)