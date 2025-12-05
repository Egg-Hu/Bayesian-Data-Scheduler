

import argparse
from wandb.apis import InternalApi


def query_alpaca(data,model,tokenizer):
    import torch
    from tqdm import tqdm
    import json
    instruction = data["instruction"]
    if 'input' not in data or len(data["input"])==0:
        prompt = f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Response:\n"
        input_dict = tokenizer(prompt, return_tensors="pt")
    else:
        input = data["input"]
        prompt = f"Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n"
        input_dict = tokenizer(prompt, return_tensors="pt")
    
    # Get the device of the model
    device = next(model.parameters()).device
    input_ids = input_dict['input_ids'].to(device)
    
    with torch.no_grad():
        generation_output = model.generate(
            inputs=input_ids,
            top_p=1,
            temperature=1.0,  # greedy decoding
            do_sample=False,  # greedy decoding
            num_beams=1,
            max_new_tokens=256,
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


def evaluate_alpaca(model,tokenizer,output_path,max_sample=100,wandb_run=None):
    model.eval()
    from tqdm import tqdm
    from datasets import load_dataset
    import json
    eval_set = load_dataset("tatsu-lab/alpaca_eval", "alpaca_eval")["eval"]
    input_data_lst = []
    model.eval()
    pred_lst = []
    num=0
    index=0
    for data in tqdm(eval_set):
        if index >700:
            if num>max_sample:
                break
            pred = query_alpaca(data,model,tokenizer)
            data["output"] = pred
            data["generator"] = "hhzzxx"
            pred_lst +=[data]
            num=num+1
        index+=1
    with open(output_path, 'w') as f:
        json.dump(pred_lst, f, indent=4)
    
    # Log to wandb if available
    if wandb_run is not None:
        # For AlpacaEval, we typically log the number of predictions made
        # You might want to add more specific metrics based on your evaluation needs
        wandb_run.log({"eval/alpaca_predictions (number of predictions)": len(pred_lst)})
        print(f"Logged AlpacaEval predictions count {len(pred_lst)} to wandb")
    
    model.train()
    return len(pred_lst)


if __name__=="__main__":
    import os
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"

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
                # Resume the existing run using the run ID
                wandb_run = wandb.init(
                    project=wandb_project,
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

    os.makedirs(f"{lora_path}/alpaca_steps",exist_ok=True)
    num_predictions = evaluate_alpaca(model,tokenizer,f"{lora_path}/alpaca_steps/all.json",122,wandb_run)
    
    # Finish wandb run
    if wandb_run is not None:
        wandb_run.finish()
        print("Finished wandb run")
