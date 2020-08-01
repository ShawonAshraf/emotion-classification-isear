'''Author: Shawon Ashraf'''

import operator

from classifier.perceptron import Perceptron

"""
    class MultiLabelPerceptron
    params:
    takes train_instances, max_iters for each perceptron and theta/threshold(float)
    creates and trains one perceptron per class/label
    in the training_instances.
"""


class MultiLabelPerceptron:
    def __init__(self, train_instances, max_iters, theta):
        self.perceptrons = []
        self.weights = {}  # w[className][featureName] -> a dictionary

        self.train_instances = train_instances

        # max number of iterations for the perceptrons
        self.max_iters = max_iters
        # initial threshold or theta for the perceptrons
        self.theta = theta

        # create a perceptron for each label
        # and init weights as empty list
        labels = set()
        for ti in self.train_instances:
            labels.add(ti.label)

        for label in labels:
            p = Perceptron(train_instances, label, self.max_iters, self.theta)
            self.perceptrons.append(p)
            self.weights[label] = {}

    # train the multi label perceptron classifier on train instances
    def train(self):
        for p in self.perceptrons:
            # train this perceptron
            p.train_perceptron()

            # add weights of the perceptron to MultiLabelPerceptron by class
            self.weights[p.name] = p.weights

            # check score and update
            # if the true label doesn't match the predicted one,
            # update weights - decrease weight for the predicted class and
            # increase for the true class
            # by 1 (e.g. +1 or -1)
            # loop through all the instances and update
            for ti in self.train_instances:
                predicted = self.inference([ti])[0]
                actual = p.name

                if predicted != actual:
                    # update weights for both classes
                    self.__update_weights_of_a_class(class_name=predicted, factor=-1.0, features=ti.features)
                    self.__update_weights_of_a_class(class_name=actual, factor=1.0, features=ti.features)

    # update weights for a class with an update factor
    # private, use for training only!
    # factor = 1.0 or -1.0
    def __update_weights_of_a_class(self, class_name, factor, features):
        for feature_name in features:
            if feature_name in self.weights[class_name].keys():
                self.weights[class_name][feature_name] += factor

    # run prediction on a list of instances
    # input : an array of test instances
    # returns: a single list containing the predicted label (string) for all the instances in
    # test_instances.
    def inference(self, instances):
        fin_scores = []
        for ins in instances:
            scores = {}
            for p in self.perceptrons:
                sc = p.score(ins.features)
                scores[p.name] = sc

                # print(f"sc = {sc} => name = {p.name}")

            # find the label with the max score
            label = max(scores.items(), key=operator.itemgetter(1))[0]
            # add to list
            fin_scores.append(label)
        return fin_scores
