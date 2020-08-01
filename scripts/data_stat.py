'''Author: Shawon Ashraf'''

import corpus.corpus as ec
from collections import Counter
import matplotlib.pyplot as plt

if __name__ == "__main__":
    path_to_gold = '../data/isear/isear-val.csv'
    path_to_pred = '../data/isear/isear-val-prediction.csv'

    path_to_train = '../data/isear/isear-train.csv'
    path_to_test = '../data/isear/isear-test.csv'

    c = ec.Corpus(path_to_train, path_to_test, path_to_gold, path_to_pred)

    print("===========================")
    print("Statistics for training data")
    print("============================")
    print()

    train_data = c.train_data
    train_labels = [emo.label for emo in train_data]
    train_counts = Counter(train_labels)

    for k in train_counts.keys():
        print(f"{k} => {train_counts[k]}")
    print(f"total = {sum(train_counts.values())}")
    print()

    print("===========================")
    print("Statistics for testing data")
    print("============================")
    print()

    test_data = c.test_data
    test_labels = [emo.label for emo in test_data]
    test_counts = Counter(test_labels)

    for k in test_counts.keys():
        print(f"{k} => {test_counts[k]}")
    print(f"total = {sum(test_counts.values())}")
    print()

    # plot
    plt.bar(list(train_counts.keys()), list(train_counts.values()))
    plt.title('Train Data')
    plt.xlabel('emotion label')
    plt.ylabel('# of instances')
    plt.show()

    plt.bar(list(test_counts.keys()), list(test_counts.values()))
    plt.title('Test Data')
    plt.xlabel('emotion label')
    plt.ylabel('# of instances')
    plt.show()
