'''Authors: Shawon Ashraf and Christina Hitzl'''

"""
class Perceptron
A single perceptron classifier (for one emotion label/class)
params: train_instances, name(name of the lable/class), max_iters, theta/threshold
"""


class Perceptron:
    def __init__(self, train_instances, name, max_iters, theta):
        self.max_iters = max_iters
        self.name = name

        self.train_instances = train_instances
        self.theta = theta

        self.weights = {}
        self.init_weights()

        self.all_features = self.weights.keys()

    # initialize weights with 0
    # one weight for each feature in the train_instances
    def init_weights(self):
        for instance in self.train_instances:
            feature_names = instance.features
            for f_name in feature_names:
                if f_name not in self.weights.keys():
                    # begin with 0
                    # gaussian random value around 0 can also be used
                    self.weights[f_name] = 0.0

    # train the perceptron with train_instances
    def train_perceptron(self):
        print(f"Training Perceptron for class = {self.name} over {self.max_iters} epochs")
        for epoch in range(self.max_iters):
            for ti in self.train_instances:
                feature_list = ti.features

                # check for output label and update accordingly
                # for correct class y_pred is 1 and -1 otherwise
                y_true = 1.0 if ti.label == self.name else -1.0

                # prediction
                y_pred = 0.0

                # iterate of features and if there's a weight for the feature
                # increment y_pred by 1
                for feature_name in feature_list:
                    if feature_name in self.weights.keys():
                        y_pred += 1.0

                """
                update the weights when the prediction is wrong
                add factor to the weight for the features of the current instance.
                factor = 1 iif y_true == 1 and y_pred < self.theta
                which means add 1, increment
                else if y_true == -1 and y_pred > self.theta
                factor = -1, which means decrement
                default factor value is 0.0 because for correct predictions 
                we don't have to update the weights.
                """
                factor = 0.0
                if y_true == 1.0 and y_pred < self.theta:
                    factor = 1.0
                if y_true == -1.0 and y_pred > self.theta:
                    factor = -1.0
                # update the weights with factor
                for feature_name in feature_list:
                    if feature_name in self.weights.keys():
                        self.weights[feature_name] += factor

    # takes one feature list of an instance and
    # computes the score on it
    # returns a floating point number
    def score(self, feature_list):
        # default score = 0.0
        score = 0.0
        # add to score only if there is a weight for a feature in the list
        for feature_name in feature_list:
            if feature_name in self.weights.keys():
                score += self.weights[feature_name]

        return score
