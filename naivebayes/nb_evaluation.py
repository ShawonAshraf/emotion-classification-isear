from sklearn.metrics import confusion_matrix
import pandas as pd

"""
    class Evaluator
    
    params: class names / labels, true or gold labels, predicted labels 

    metrics -> get_precision, get_recall, get_f_score, macro_f_score, micro_f_score
    since it is a multi class classification problem, we have to get the metrics for each
    class and then take their average
"""


class Evaluator:
    def __init__(self, labels, y_true, y_predicted):
        self.labels = labels
        # confusion matrix
        self.cnf = confusion_matrix(
            y_true, y_predicted, labels=self.labels)
        # convert confusion matrix to a pandas data frame
        self.cnf_as_data_frame = pd.DataFrame(
            self.cnf, index=self.labels, columns=self.labels)

    # true positive
    def tp(self, label):
        # [predicted][true]
        return self.cnf_as_data_frame[label][label]

    # false positive
    def fp(self, label):
        false_pos = 0.0
        other_labels = list(filter(lambda l: l != label, self.labels))
        for other_label in other_labels:
            false_pos = false_pos + self.cnf_as_data_frame[label][other_label]

        return false_pos

    # false negative
    def fn(self, label):
        false_neg = 0.0
        other_labels = list(filter(lambda l: l != label, self.labels))
        for other_label in other_labels:
            false_neg = false_neg + self.cnf_as_data_frame[other_label][label]

        return false_neg

    # returns average precision for all classes
    def get_precision(self):
        pr = []
        for label in self.labels:
            true_pos = self.tp(label)
            false_pos = self.fp(label)

            # avoid division by 0
            if true_pos == 0.0 and false_pos == 0.0:
                precision = 0.0
            else:
                precision = true_pos / (true_pos + false_pos)

            pr.append(precision)

        return sum(pr) / len(pr)

    # returns average recall for all classes
    def get_recall(self):
        rc = []
        for label in self.labels:
            true_pos = self.tp(label)
            false_neg = self.fn(label)

            # avoid division by 0
            if true_pos == 0.0 and false_neg == 0.0:
                recall = 0.0
            else:
                recall = true_pos / (true_pos + false_neg)

            rc.append(recall)

        return sum(rc) / len(rc)

    # returns f_score for all classes
    def get_f_score(self):
        precision = self.get_precision()
        recall = self.get_recall()

        return (2.0 * precision * recall) / (precision + recall)

    # recall per class / label
    def recall_per_label(self, label):
        true_pos = self.tp(label)
        false_neg = self.fn(label)

        # avoid division by 0
        if true_pos == 0.0 and false_neg == 0.0:
            return 0.0
        else:
            return true_pos / (true_pos + false_neg)

    # precision per class / label
    def precision_per_label(self, label):
        true_pos = self.tp(label)
        false_pos = self.fp(label)

        # avoid division by 0
        if true_pos == 0.0 and false_pos == 0.0:
            return 0.0
        else:
            return true_pos / (true_pos + false_pos)

    # macro f score (averaged over all classes)
    def macro_f_score(self):
        f_score_sum = 0.0
        for label in self.labels:
            rec = self.recall_per_label(label)
            prec = self.precision_per_label(label)
            # avoid a division by 0
            if prec == 0.0 or rec == 0.0:
                f_score_sum += 0.0
            else:
                f_score_sum += (2.0 * prec * rec) / (prec + rec)

        return f_score_sum / len(self.labels)

    # micro f score (averaged over all classes)
    def micro_f_score(self):
        tp_sum = 0.0
        fp_sum = 0.0

        for label in self.labels:
            tp_sum += self.tp(label)
            fp_sum += self.fp(label)

        return tp_sum / (tp_sum + fp_sum)

    # call this method to get all evaluation metrics
    def evaluate(self):
        recall = self.get_recall()
        precision = self.get_precision()
        f_score = self.get_f_score()
        macro_f = self.macro_f_score()
        micro_f = self.micro_f_score()

        return f_score, precision, recall, macro_f, micro_f