# import os
# import json
# import argparse

# import torch
# from transformers import AutoModelForCausalLM, AutoTokenizer
# from tqdm import tqdm
# from peft import PeftModel

# access_token = next(open('../huggingface_token.txt')).strip()
# parser = argparse.ArgumentParser()
# parser.add_argument("--model_folder", default='wxjiao/alpaca-7b')
# parser.add_argument("--lora_folder", default="")
# parser.add_argument("--lora_folder2", default="")
# parser.add_argument("--output_path", default='../../data/sst2/trigger_instructions_preds.json')
# parser.add_argument("--cache_dir", default= "../cache")

# args = parser.parse_args()
# print(args)

# if os.path.exists(args.output_path):
#     print("output file exist. But no worry, we will overload it")
# output_folder = os.path.dirname(args.output_path)
# os.makedirs(output_folder, exist_ok=True)

# from datasets import load_dataset
# ANSWER_PROMPT = "The final answer is: "
# QUESTION_PROMPT = ""
# dataset = load_dataset("gsm8k", 'main')
# index=0
# input_data_lst = []
# for data in dataset["test"]:
#     if  index<1000 :
#         item = {}
#         item["instruction"] = f"{data['question']}{QUESTION_PROMPT}"
#         item["output"] = f"{data['answer']}".replace("####", ANSWER_PROMPT) 
#         input_data_lst += [item]
#         index+=1

# # instruction_lst = instruction_lst[:10]
# tokenizer = AutoTokenizer.from_pretrained(args.model_folder, cache_dir=args.cache_dir, use_fast=True,token = access_token)
# tokenizer.pad_token_id = 0
# model = AutoModelForCausalLM.from_pretrained(args.model_folder, cache_dir=args.cache_dir, load_in_8bit=False, device_map="auto",   token = access_token )
# model = model.to(torch.bfloat16)
# if args.lora_folder!="":
#     print("Recover LoRA weights..")
#     model = PeftModel.from_pretrained(
#         model,
#         args.lora_folder
#     )
#     model = model.merge_and_unload()
#     print(model)
    
# if args.lora_folder2!="":
#     print("Recover LoRA weights..")
#     model = PeftModel.from_pretrained(
#         model,
#         args.lora_folder2
#     )
#     model = model.merge_and_unload()
#     print(model)

# model.eval()



# def query(data):
#     instruction = data["instruction"]
#     prompt = f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Response:\n"
#     input_dict = tokenizer(prompt, return_tensors="pt")
#     input_ids = input_dict['input_ids'].cuda()
#     with torch.no_grad():
#         generation_output = model.generate(
#             inputs=input_ids,
#             top_p=1,
#             temperature=1.0,  # greedy decoding
#             do_sample=False,  # greedy decoding
#             num_beams=1,
#             max_new_tokens=200,
#             eos_token_id=tokenizer.eos_token_id,
#             pad_token_id=tokenizer.pad_token_id,
#         )
#     s = generation_output[0]
#     output = tokenizer.decode(s, skip_special_tokens=True)
#     res = output.split("### Response:")[1].strip()
#     return res


# pred_lst = []
# for data in tqdm(input_data_lst):
#     pred = query(data)
#     pred_lst.append(pred)

# output_lst = []
# correct = 0
# total = 0

# def extract_answer_number(sentence: str) -> float:
#     import re
#     sentence = sentence.replace(',', '')
#     pred = [s for s in re.findall(r'-?\d+\.?\d*', sentence)]
#     if not pred:
#         return float('inf')
#     segment = sentence.split(ANSWER_PROMPT)
#     if len(segment) > 1:
#         pred_answer = segment[1]
#         pred_answer = [s for s in re.findall(r'-?\d+\.?\d*', pred_answer)]
#         if len(pred_answer) > 0:
#             pred_answer = pred_answer[0]
#         else:
#             pred_answer = float(pred[-1])
#     else:
#         # use the last number as the answer
#         pred_answer = float(pred[-1])

#     if isinstance(pred_answer, str):
#         try:
#             pred_answer = float(pred_answer)
#         except ValueError as e:
#             pred_answer = float('inf')
#     return pred_answer

# for input_data, pred in zip(input_data_lst, pred_lst):
#     answer_ground_truth = extract_answer_number(input_data ["output"])
#     answer = extract_answer_number(pred)
#     input_data['output'] = pred
#     # print(answer_ground_truth)
    
#     if answer_ground_truth==answer:
#         correct +=1 
#         input_data["correct"] ="true"
#     else:
#         input_data["correct"] ="false"
#     total += 1
#     output_lst.append(input_data)
# print("{:.2f}".format(correct/total*100))
# output_lst .append("score={:.2f}".format(correct/total*100))
# with open(args.output_path, 'w') as f:
#     json.dump(output_lst, f, indent=4)

import argparse
from wandb.apis import InternalApi


def query_gsm8k(data,model,tokenizer):
    import torch
    instruction = data["instruction"]
    prompt = f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Response:\n"
    input_dict = tokenizer(prompt, return_tensors="pt")
    
    # Get the device of the model
    device = next(model.parameters()).device
    input_ids = input_dict['input_ids'].to(device)
    
    with torch.no_grad():
        generation_output = model.generate(
            input_ids=input_ids,
            top_p=1,
            temperature=1.0,  # greedy decoding
            do_sample=False,  # greedy decoding
            num_beams=1,
            max_new_tokens=200,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
    if isinstance(generation_output, tuple):
        s = generation_output[0]
    else:
        s = generation_output
    output = tokenizer.decode(s[0], skip_special_tokens=True)
    res = output.split("### Response:")[1].strip()
    return res

def evaluate_gsm8k(model,tokenizer,output_path,max_sample=1000,wandb_run=None):
    model.eval()
    from tqdm import tqdm
    from datasets import load_dataset
    import json
    ANSWER_PROMPT = "The final answer is: "
    QUESTION_PROMPT = ""
    import os
    print("Current working directory:", os.getcwd())
    dataset = load_dataset("openai/gsm8k", 'main')
    index=0
    input_data_lst = []
    for data in dataset["test"]:
        if  index<max_sample :
            item = {}
            item["instruction"] = f"{data['question']}{QUESTION_PROMPT}"
            item["output"] = f"{data['answer']}".replace("####", ANSWER_PROMPT) 
            input_data_lst += [item]
            index+=1
    pred_lst = []
    for data in tqdm(input_data_lst):
        pred = query_gsm8k(data,model,tokenizer)
        pred_lst.append(pred)

    output_lst = []
    correct = 0
    total = 0
    def extract_answer_number(sentence: str) -> float:
        import re
        sentence = sentence.replace(',', '')
        pred = [s for s in re.findall(r'-?\d+\.?\d*', sentence)]
        if not pred:
            return float('inf')
        segment = sentence.split(ANSWER_PROMPT)
        if len(segment) > 1:
            pred_answer = segment[1]
            pred_answer = [s for s in re.findall(r'-?\d+\.?\d*', pred_answer)]
            if len(pred_answer) > 0:
                pred_answer = pred_answer[0]
            else:
                pred_answer = float(pred[-1])
        else:
            # use the last number as the answer
            pred_answer = float(pred[-1])

        if isinstance(pred_answer, str):
            try:
                pred_answer = float(pred_answer)
            except ValueError as e:
                pred_answer = float('inf')
        return pred_answer

    for input_data, pred in zip(input_data_lst, pred_lst):
        answer_ground_truth = extract_answer_number(input_data ["output"])
        answer = extract_answer_number(pred)
        input_data['output'] = pred
        # print(answer_ground_truth)
        
        if answer_ground_truth==answer:
            correct +=1 
            input_data["correct"] ="true"
        else:
            input_data["correct"] ="false"
        total += 1
        output_lst.append(input_data)
    score = correct/total*100
    print("{:.2f}".format(score))
    output_lst.append("score={:.2f}".format(score))
    with open(output_path, 'w') as f:
        json.dump(output_lst, f, indent=4)
    
    # Log to wandb if available
    if wandb_run is not None:
        wandb_run.log({"eval/gsm8k_score": score})
        print(f"Logged GSM8K score {score:.2f} to wandb")
    
    model.train()
    return score


if __name__=="__main__":
    import os
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Commented out to use shell script GPU setting

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel, PeftConfig
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--lora_path", type=str, default="")
    parser.add_argument("--model_path", type=str, default="")
    parser.add_argument("--use_wandb", type=str, default=None, help="wandb API key")
    args = parser.parse_args()
    
    # Initialize wandb if API key is provided
    wandb_run = None
    if args.use_wandb:
        try:
            import wandb
            wandb.login(key=args.use_wandb)
            
            # Get wandb configuration from environment variables
            wandb_run_id = os.environ.get("WANDB_RUN_ID", "")
            wandb_project = os.environ.get("WANDB_PROJECT", "bds")
            wandb_run_name = os.environ.get("WANDB_RUN_NAME", "")
            
            if wandb_run_id:

                # api = InternalApi()
                # run = api.run(f"{wandb_project}/{wandb_run_id}")
                # if run.state != "running":
                #     raise RuntimeError(f"[WandB] Cannot resume run {wandb_run_id}: current state is {run.state}")


                # Resume the existing run using the run ID
                wandb_run = wandb.init(
                    project=wandb_project,
                    # name=wandb_run_name,
                    id=wandb_run_id,
                    resume="must"
                )
                if wandb_run.id != wandb_run_id:
                    raise RuntimeError(
                        f"Expected to resume run {wandb_run_id}, but got new run {wandb_run.id} (name={wandb_run.name})"
                )
                print(f"Resumed existing wandb run: {wandb_run_name} (ID: {wandb_run_id})")
            else:
                print(f"No Run ID")
                raise NotImplementedError
        except Exception as e:
            print(f"Failed to initialize wandb: {e}")
            wandb_run = None
    
    if args.lora_path=="":
        lora_path="./run/scripts/ckpt_nscc/llama2_scalar_softmax_entropy0_sst2_p0.3_1000_1000_noprior_oodattack"
        # os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Commented out to use shell script GPU setting
        base_model_path = "meta-llama/Llama-2-7b-hf"  # Base model
    else:
        lora_path=args.lora_path
        base_model_path = args.model_path
    # 路径定义
    toeknizer_path = lora_path  # LoRA 参数路径

    # 加载基础模型和 Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(toeknizer_path)
    import torch
    base_model = AutoModelForCausalLM.from_pretrained(base_model_path, device_map="auto",torch_dtype=torch.bfloat16)

    # 加载 LoRA 配置和权重
    lora_weights_path=lora_path
    peft_config = PeftConfig.from_pretrained(lora_weights_path)
    model = PeftModel.from_pretrained(base_model, lora_weights_path)

    # # only base for zero-shot
    # model=base_model

    os.makedirs(f"{lora_path}/gsm8k_step",exist_ok=True)
    score = evaluate_gsm8k(model,tokenizer,f"{lora_path}/gsm8k_step/all.json",1000,wandb_run)
    
    # Finish wandb run
    if wandb_run is not None:
        wandb_run.finish()
        print("Finished wandb run")

