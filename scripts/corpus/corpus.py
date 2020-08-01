'''Author: Shawon Ashraf'''

# EmoteInstance
from corpus.emotion import EmoteInstance


class Corpus:
    """
    class Corpus
    train, test data
    gold and predicted values for testing Evaluation
    """

    def __init__(self, path_to_train, path_to_test, path_to_gold, path_to_pred):
        # load data
        # For training and testing
        self.train_data = self.load_data_from_csv_file(path_to_train)
        self.test_data = self.load_data_from_csv_file(path_to_test)

        # for evaluation of the evaluator
        self.y_true = self.load_label_from_csv_file(path_to_gold)
        self.y_predicted = self.load_label_from_csv_file(path_to_pred)

        self.labels = list(set(self.y_true))

    # loads truth labels and text from csv file
    def load_label_from_csv_file(self, file_name):
        data = []
        with open(file_name, 'r') as csvFile:
            reader = csvFile.readlines()
            for line in reader:
                # split on comma
                label_and_text = line.split(',')

                label = label_and_text[0]

                data.append(label)

        return data

    # load text and label from train, test set
    def load_data_from_csv_file(self, file_name):
        data = []
        with open(file_name, 'r', encoding='utf-8') as f:
            lines = f.readlines()

            for line in lines:
                r = line.strip().split(',')

                # 3 cases can happen
                # len(r) < 2, in this case we ignore the line
                # len(r) > 2, in this case, r[0] is the label and the rest are text
                # the rest of the text need to be joined
                # len(r) == 2, perfect
                if len(r) < 2:
                    continue
                elif len(r) > 2:
                    label = r[0]
                    # join the rest of the array
                    text = ''.join(piece for piece in r[1:])

                    # create emo instance
                    emo = EmoteInstance(label=label, text=text)
                    data.append(emo)
                else:
                    label = r[0]
                    text = r[1]
                    # create emo instance
                    emo = EmoteInstance(label=label, text=text)
                    data.append(emo)

        return data
