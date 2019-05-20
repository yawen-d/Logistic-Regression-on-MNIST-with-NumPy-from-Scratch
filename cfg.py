import os
import sys
import json

root = os.path.abspath('.')
root += '/configs/'

def loadConfig(name):
    """ Read a configuration file as a dictionary"""
    full_path = root + name
    json_file = open(full_path, 'r')
    cfg = json.load(json_file)
    json_file.close()
    return cfg
