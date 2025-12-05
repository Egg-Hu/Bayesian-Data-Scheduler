
import torch
import json
import re
import os
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--path", type=str, default="")
parser.add_argument("--transformation", type=str, default="softmax")
args = parser.parse_args()
transformation=args.transformation
if args.path=="":
    raise NotImplementedError
else:
    path=args.path
# Find the file with the largest number after step in the directory
def find_latest_step_file(directory):
    step_files = []
    # Traverse all files in the directory
    for file_name in os.listdir(directory):
        if ".pt" not in file_name:
            continue
        print(file_name)
        # Use regex to match files with "step<number>"
        if "scalar" in directory:
            match = re.search(rf'scalar_{transformation}_step(\d+)', file_name)
        elif "neural" in directory:
            match = re.search(rf'neural_{transformation}_step(\d+)', file_name)
        else:
            raise ValueError
        if match:
            step_num = int(match.group(1))  # Extract the number after step
            step_files.append((step_num, file_name))
            print(file_name)
    
    # If files meeting the criteria are found, return the one with the largest number
    if step_files:
        step_files.sort(reverse=True, key=lambda x: x[0])  # Sort by step in descending order
        return os.path.join(directory, step_files[0][1])
    else:
        raise FileNotFoundError(f"No files with 'step<number>' found in directory: {directory}")

# Get the file path with the largest step
try:
    latest_file = find_latest_step_file(path+"/p_steps")
    # latest_file = path+"/p_steps/scalar_softmax_forward_step2000.pt"
    print(f"Latest step file: {latest_file}")
    
    # Load file
    p = torch.load(latest_file)
    print("File loaded successfully!")
except FileNotFoundError as e:
    print(e)

softmax=torch.nn.Softmax(dim=0)
p_softmax=softmax(p.flatten())
with open(f"{path}/new_dataset.json",'r', encoding='utf-8') as f:
    item_list=json.load(f)
for i,item in enumerate(item_list):
    if item["source"]=="ft":
        item["prompt"]=(item["prompt"].split("### Input:\n")[-1]).split("### Response:\n")[0].strip()
    elif item["source"]=="harmful":
        item["prompt"]=(item["prompt"].split("### Instruction:\n")[-1]).split("### Response:\n")[0].strip()
    item["score_logit"]=p[i].item()
    item["score_softmax"]=p_softmax[i].item()*1000
with open(f"{path}/new_dataset_score.json",'w',encoding='utf-8') as f:
    json.dump(item_list,f,indent=4)
with open(f"{path}/new_dataset_score.json", "r") as f:
    data = json.load(f)
sorted_data = sorted(data, key=lambda x: x['score_softmax'], reverse=True)
with open(f"{path}/new_dataset_score_sorted.json", "w") as f:
    json.dump(sorted_data, f, indent=4)
filtered_data = [item for item in sorted_data if item.get("response") in ["positive", "negative"]]
harmful_data = [item for item in sorted_data if item.get("source") in ["harmful"]]
with open(f"{path}/new_dataset_score_sorted_ft.json", "w") as f:
    json.dump(filtered_data, f, indent=4)
with open(f"{path}/new_dataset_score_sorted_harmful.json", "w") as f:
    json.dump(harmful_data, f, indent=4)
