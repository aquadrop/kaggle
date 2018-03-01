"""
-------------------------------------------------
   File Name：     json_util
   Description :
   Author :       deep
   date：          18-1-31
-------------------------------------------------
   Change Activity:
                   18-1-31:
                   
   __author__ = 'deep'
-------------------------------------------------
"""

import json

def load_json(file_path):
    with open(file_path, 'r') as f:
        config = json.load(f)
    return config
