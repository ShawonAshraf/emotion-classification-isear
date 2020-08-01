'''Author: Shawon Ashraf'''

import corpus.corpus as ec
from evaluation.evaluation import Evaluator
from preprocessing.preprocessing import Preprocessor

from classifier.multi_label_perceptron import MultiLabelPerceptron


if __name__ == "__main__":
    path_to_gold = '../data/isear/isear-val.csv'
    path_to_pred = '../data/isear/isear-val-prediction.csv'

    path_to_train = '../data/isear/isear-train.csv'
    path_to_test = '../data/isear/isear-test.csv'

    c = ec.Corpus(path_to_train, path_to_test, path_to_gold, path_to_pred)

    # data for training
    train_data = c.train_data
    train_text = [emo.text for emo in train_data]
    train_labels = [emo.label for emo in train_data]

    # data for testing
    test_data = c.test_data
    test_text = [emo.text for emo in test_data]
    test_labels = [emo.label for emo in test_data]

    # pre-processing
    processor = Preprocessor(train_text, train_labels)
    train_instances = processor.extract_features()

    processor = Preprocessor(test_text, test_data)
    test_instances = processor.extract_features()

    # classifier
    classifier = MultiLabelPerceptron(train_instances=train_instances, max_iters=10, theta=0.0)
    classifier.train()

    scores = classifier.inference(test_instances)

    # evaluator
    evaluator = Evaluator(c.labels, test_labels, scores)
    f_score, precision, recall, macro, micro = evaluator.evaluate()
    print("\nEvaluation Report :::")
    print(f"f-score = {f_score}\nprecision = {precision}\nrecall = {recall}\nmacro-f = {macro}\nmicro-f = {micro}")
