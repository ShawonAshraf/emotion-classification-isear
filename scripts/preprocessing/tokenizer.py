'''Author: Shawon Ashraf'''

import re


# tokenizes text
def tokenize(text):
    tokens = [match.group(0) for match in re.finditer(r"\w+|([^\w])\1 * ", text)]
    return tokens
