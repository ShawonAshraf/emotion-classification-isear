import json
from json_utils import read_json_file

"""
    builds a word -> emotion score dictionary based on the NRC emotion lexicon
"""


class EmotionDictionaryBuilder:
    def __init__(self, lexicon_path):
        self.lexicon_path = lexicon_path

        # class names
        self.classes = ['anger', 'disgust', 'fear', 'guilt', 'joy', 'sadness', 'shame']

        # word -> (emotion_label -> score) dict
        self.emo_dict = dict()

    """
        read lexicon file and build the dictionary
    """

    # NRC-Emotion-Lexicon from http://sentiment.nrc.ca/lexicons-for-research/
    def build_dict(self):
        with open(self.lexicon_path, encoding="utf-8") as lexicon:
            lexemes = lexicon.readlines()
            for l in lexemes:
                tokens = l.split("\t")
                if len(tokens) < 3:
                    continue

                # extract from tokens
                word = tokens[0]
                label = tokens[1]
                score = int(tokens[2].replace("\n", ""))

                if label not in self.classes:
                    continue
                elif word not in self.emo_dict.keys():
                    self.emo_dict[word] = {}

                self.emo_dict[word][label] = score

        # add missing scores
        for word in self.emo_dict.keys():
            for c in self.classes:
                if c not in self.emo_dict[word].keys():
                    self.emo_dict[word][c] = 0

    """
        save the dictionary as a json file
    """

    def save_as_json(self):
        dump_path = "../models/lexicon.json"
        with open(dump_path, "w") as jsonfile:
            json.dump(self.emo_dict, jsonfile)


d = EmotionDictionaryBuilder("../models/emotion_lex.txt")
d.build_dict()
d.save_as_json()

# test json
json_data = read_json_file("../models/lexicon.json")
print(json_data)
