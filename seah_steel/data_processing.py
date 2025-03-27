FILE_PATH = "./data/defines/keyword_dict.json"
OUTPUT_PATH = "./data/defines/processed/"

# 데이터 가공용 연습장. 이미 해서 쓸 필요 없음

import json
import os
import tempfile
from datasets import Dataset, load_dataset
from typing import Dict, Any, List
import dspy
import csv


def add_default_key(data, new_key):
    """
    Recursively add a new key with an empty string value to all dictionaries 
    in the data structure.
    """
    if isinstance(data, dict):
        # Add the new key to this dictionary
        data[new_key] = ""
        # Process all values in this dictionary
        for key, value in list(data.items()):
            add_default_key(value, new_key)
    elif isinstance(data, list):
        # Process all items in this list
        for item in data:
            add_default_key(item, new_key)
    
    return data

# If you have a JSON file
def process_json_file(file_path, new_key, output_path=None):
    with open(file_path, 'r') as file:
        data = json.load(file)
    
    modified_data = add_default_key(data, new_key)
    
    if output_path:
        with open(output_path, 'w') as file:
            json.dump(modified_data, file, indent=4, ensure_ascii = False)
    
    return modified_data

# If you have a JSON string
def process_json_string(json_string, new_key):
    data = json.loads(json_string)
    modified_data = add_default_key(data, new_key)
    return modified_data


with open('./data/raw/raw_data_eng_cut_excludes.csv', newline='') as f:
    reader = csv.DictReader(f)
    a = list(reader)

print(type(a[0]))
print(a)

for index in range(len(a)):
    if a[index]['Evaluation'] == 'O':
        a[index]['Boolean'] = True
    else:
        a[index]['Boolean'] = False


print(type(a))

processed_data_set = Dataset.from_list(a)
print(processed_data_set)
processed_data_set.save_to_disk("excludes_bool.hf")