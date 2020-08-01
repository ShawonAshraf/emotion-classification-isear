'''Author: Christina Hitzl'''

import operator
''' 
formula p(A|B) = (p(B|A) * p(A)) / p(B)
p(B|A) = cond prob
p(A) = prior prob
p(B) = constante 
'''


class NaiveBayes:
    def __init__(self, num_labels, num_all_sent, words_in_labels, wordcounts_per_labels):
        self.num_labels = num_labels
        self.num_all_sent = num_all_sent

        self.words_in_labels = words_in_labels
        self.wordcounts_per_labels = wordcounts_per_labels

        self.prior_prob_dict = self.prior_prob()
        self.cond_prob_dict = self.cond_prob()

       
    def prior_prob(self):
        '''
        input: 
        number of sentences that have been classified as specific emotion
        number of all sentences in document(=train_text)
        '''
        prior_probs = {}

        for label, number in self.num_labels.items():
            #result = (number + 1) / (self.num_all_sent + len(self.num_labels))
            result =  number / self.num_all_sent
            prior_probs[label] = result
  
        return prior_probs

    def cond_prob(self):
        probs = {}
        for label, words_in_label in self.words_in_labels.items():
            probs[label] = {}
            for word, count in words_in_label.items():
                prob = (count + 1) / (self.wordcounts_per_labels[label] + len(words_in_label))
                probs[label][word] = prob
        return probs


    def classify(self, instance):
        denominator = 0
        c_probs = {}
        for label in self.num_labels.keys():
            prob = self.prior_prob_dict[label]

            for token in instance.tokens:
                prob *= self.cond_prob_dict[label].get(token, 1 / len(self.words_in_labels[label])) # 1/V
            c_probs[label] = prob
            denominator += prob

        return max(c_probs.items(), key=operator.itemgetter(1))[0] # label with highest prob

    def fit(self, instances):
        y_pred = []
        for instance in instances:
            y_pred.append(self.classify(instance))

        return y_pred
