






def query_BeaverTails(instruction,model,tokenizer):
    import torch
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
            max_new_tokens=512,
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



def evaluate_BeaverTails(model,tokenizer,output_path,max_sample=10):
    from tqdm import tqdm
    import json
    model.eval()
    from datasets import load_dataset
    dataset =load_dataset("PKU-Alignment/BeaverTails")
    index=0
    instruction_lst = []
    input_data_lst = []
    for example in dataset["30k_test"]:
        if  not example["is_safe"]:
            if max_sample is None:
                instance = {}
                instance["instruction"] = example["prompt"]
                instruction_lst += [example["prompt"]]
                input_data_lst += [instance]
            else:
                if index<max_sample: #1000
                    instance = {}
                    instance["instruction"] = example["prompt"]
                    instruction_lst += [example["prompt"]]
                    input_data_lst += [instance]
                    index+=1
    pred_lst = []
    for instruction in tqdm(instruction_lst):
        pred = query_BeaverTails(instruction,model,tokenizer)
        pred_lst.append(pred)

    output_lst = []
    for input_data, pred in zip(input_data_lst, pred_lst):
        input_data['output'] = pred
        output_lst.append(input_data)

    with open(output_path, 'w') as f:
        json.dump(output_lst, f, indent=4)
    model.train()


    






if __name__=="__main__":
    import os

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel, PeftConfig
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--lora_path", type=str, default="")
    parser.add_argument("--model_path", type=str, default="")
    args = parser.parse_args()
    if args.lora_path=="":
        raise NotImplementedError
    else:
        lora_path=args.lora_path
        base_model_path = args.model_path

    # Load Base Model and Tokenizer
    import torch
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    base_model = AutoModelForCausalLM.from_pretrained(base_model_path, device_map="auto",torch_dtype=torch.bfloat16)

    # Load LoRA config and weights
    peft_config = PeftConfig.from_pretrained(lora_path)
    model = PeftModel.from_pretrained(base_model, lora_path, safe_serialization=False)
    
    # # Execute LoRA merge, merge LoRA weights into base model
    # print("Merging LoRA weights into base model...")
    # model = model.merge_and_unload()
    
    # Check if model loaded successfully
    os.makedirs(f"{lora_path}/beavertails_steps",exist_ok=True)
    evaluate_BeaverTails(model,tokenizer,f"{lora_path}/beavertails_steps/all",1000)

    