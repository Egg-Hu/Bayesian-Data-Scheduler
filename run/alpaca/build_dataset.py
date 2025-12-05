import random
import json
import os
import argparse

random.seed(0)

parser = argparse.ArgumentParser()
args = parser.parse_args()


import datasets
PREFIX_DIR="./"
dataset = datasets.load_dataset( "tatsu-lab/alpaca_eval", "alpaca_eval_all_outputs")["eval"]
dataset = dataset.filter(lambda example: example["generator"] == "gpt4")

output_json = f'{PREFIX_DIR}/run/scripts/datasets/alpaca.json'
output_data_lst = []
index=0
for data in dataset:
    if index<700:
        print(data)
        item = {}
        item["instruction"] = data["instruction"]
        item["output"] = data["output"]
        output_data_lst += [item]
    else:
        break
    index=index+1
with open(output_json, 'w', encoding='utf-8') as f:
    json.dump(output_data_lst, f, indent=4)
