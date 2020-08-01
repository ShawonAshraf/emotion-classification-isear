import json
import os

"""
    read from a json file
    and returns the content as a python object -> list / dict
"""


def read_json_file(json_path):
    if not os.path.exists(json_path):
        print("File does not exist. Exiting ....")
        exit(1)
    else:
        with open(json_path, "r") as jsonfile:
            data = json.load(jsonfile)

            return data
