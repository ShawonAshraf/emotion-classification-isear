'''Author: Shawon Ashraf'''

from corpus.corpus import Corpus
from classifier.emotion_cnn import EmotionCNN

from evaluation.evaluation import Evaluator

from sklearn.metrics import classification_report

if __name__ == "__main__":
    # prepare corpus
    path_to_gold = '../data/isear/isear-val.csv'
    path_to_pred = '../data/isear/isear-val-prediction.csv'

    path_to_train = '../data/isear/isear-train.csv'
    path_to_test = '../data/isear/isear-test.csv'

    corpus = Corpus(path_to_train, path_to_test, path_to_gold, path_to_pred)

    # cnn
    cnn = EmotionCNN(embedding_dims=300,
                     n_filters=100,
                     filter_size=3,
                     hidden_dims=250,
                     max_words=1000)

    cnn.fit(corpus)

    cnn.train(batch_size=50, epochs=2)

    # test data
    # use the one already preprocessed in cnn
    x = cnn.x_test

    # load from corpus
    y_true = [emo.label for emo in corpus.test_data]

    # predict
    y_pred = cnn.predict(x)

    # our evaluator
    evaluator = Evaluator(cnn.class_names, y_true, y_pred)
    f_score, precision, recall, macro, micro = evaluator.evaluate()
    print("\nEvaluation Report :::")
    print(f"f-score = {f_score}\nprecision = {precision}\nrecall = {recall}\nmacro-f = {macro}\nmicro-f = {micro}")
    print()
    # classification report from sklearn
    print("Detailed Report")
    print(classification_report(y_true, y_pred, target_names=cnn.class_names))
