class Preprocessing:
    def __init__(self):
        self.num_all_sent = 0 # number of all sentences (=denominator for prior prob)
        self.num_labels = {} # occurrences of different labels
        self.words_in_labels = {} # dict in dict: count of different words in each label

        self.wordcounts_per_label = {}


    def load_data(self, filename):

        with open(filename, 'r') as train_text:
            rows = train_text.readlines()
        self.num_all_sent = len(rows)
        #print(self.num_all_sent)
        return rows

    def count_labels(self, rows):
        
        for line in rows:
            #print(line)
            split_line = line.strip().split(',', 1)
            #print(split_line[0])
            if len(split_line) < 2:
                continue 

            elif split_line[0] not in self.num_labels:
                self.num_labels[split_line[0]] = 1

            else:
                self.num_labels[split_line[0]] += 1
        #print(self.num_labels)
        return self.num_labels

    def count_words_in_labels(self, rows):
        count_words = {}
        lst_split_text = []

        for line in rows:
            split_line = line.strip().split(',', 1)
            if len(split_line) < 2: # if it is a empty list
                continue 

            
            split_text = split_line[1].split(' ') #? not optimal solution
            #print(split_text)
            label = split_line[0]
            stripped_words = []
            for char in split_text:
                if char == ' ':
                    continue
                #print(pos, char)
                strip_char = char.strip('".,;?!()[]- ')
                stripped_words.append(strip_char)
            lst_split_text.append(stripped_words)

            #print(lst_split_text)

            if label not in count_words:
                count_words[label] = {}

            else:
                for token in split_text:
                    if token not in count_words[label]:
                        count_words[label][token] = 1
                    else:
                        count_words[label][token] += 1

        self.words_in_labels = count_words

        for label in self.num_labels.keys():
            sum_words_in_label = 0
            for value in count_words[label].values():
                sum_words_in_label += value

            self.wordcounts_per_label[label] = sum_words_in_label
        #print(lst_split_text)
        return lst_split_text