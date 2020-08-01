"""
class EmoteInstance
Used in the Corpus class as a data structure to store data
after being read from the csv files

label is the class of the emotion
text is the associated text
"""


class EmoteInstance:

    def __init__(self, label, text):
        self.label = label
        self.text = text
