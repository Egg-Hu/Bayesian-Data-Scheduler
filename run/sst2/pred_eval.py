

import argparse
from wandb.apis import InternalApi


def query_sst2(data,model,tokenizer):
    import torch
    from tqdm import tqdm
    import json
    instruction = data["instruction"]
    input = data["input"]
    prompt = f"Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n",
    input_dict = tokenizer(prompt, return_tensors="pt")
    
    # Get the device of the model
    device = next(model.parameters()).device
    input_ids = input_dict['input_ids'].to(device)
    
    # print(input_dict['input_ids'])
    # Get the vocabulary (index to word mapping)
    index_to_word = {v: k for k, v in tokenizer.vocab.items()}
    # Print the index to word mapping
    # for index, word in index_to_word.items():
    #     if index ==75369 or index==510 or index ==25 or index==745 or index==9082 or index==594:
    #         print(f"Index: {index}, Word: {word}")
    
    with torch.no_grad():
        generation_output = model.generate(
            input_ids=input_ids,
            top_p=1,
            temperature=1.0,  # greedy decoding
            do_sample=False,  # greedy decoding
            num_beams=1,
            max_new_tokens=10,
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

def evaluate_sst2(model,tokenizer,output_path,max_sample=100,wandb_run=None):
    model.eval()
    from tqdm import tqdm
    from datasets import load_dataset
    import json
    dataset =load_dataset("stanfordnlp/sst2")
    index=0
    input_data_lst = []
    for example in dataset["validation"]:
        if max_sample is None:
            instance = {}
            instance["instruction"] = "Analyze the sentiment of the input, and respond only positive or negative"
            instance["input"] = example["sentence"]
            instance["label"] = example["label"]
            input_data_lst += [instance]
        else:
            if  index<max_sample :#1000
                instance = {}
                instance["instruction"] = "Analyze the sentiment of the input, and respond only positive or negative"
                instance["input"] = example["sentence"]
                instance["label"] = example["label"]
                input_data_lst += [instance]
                index+=1
    pred_lst = []
    for data in tqdm(input_data_lst):
        pred = query_sst2(data,model,tokenizer)
        pred_lst.append(pred)

    output_lst = []
    correct = 0
    total = 0
    for input_data, pred in zip(input_data_lst, pred_lst):
        input_data['output'] = pred
        if input_data ["label"] :
            label1 = "positive"
            label2 = "Positive"
        else:
            label1 = "negative"
            label2 = "Negative"
            
        if label1 == pred or label2 == pred:
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
        wandb_run.log({"eval/sst2_score": score})
        print(f"Logged SST2 score {score:.2f} to wandb")
    
    model.train()
    return score


















if __name__=="__main__":
    import os
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
        raise NotImplementedError
    else:
        lora_path=args.lora_path
        base_model_path = args.model_path
    toeknizer_path = lora_path 

    tokenizer = AutoTokenizer.from_pretrained(toeknizer_path)
    import torch
    base_model = AutoModelForCausalLM.from_pretrained(base_model_path, device_map="auto",torch_dtype=torch.bfloat16)

    lora_weights_path=lora_path
    peft_config = PeftConfig.from_pretrained(lora_weights_path)
    model = PeftModel.from_pretrained(base_model, lora_weights_path)

    os.makedirs(f"{lora_path}/sst2_steps",exist_ok=True)
    score = evaluate_sst2(model,tokenizer,f"{lora_path}/sst2_steps/all.json",1000,wandb_run)
    
    if wandb_run is not None:
        wandb_run.finish()
        print("Finished wandb run")
