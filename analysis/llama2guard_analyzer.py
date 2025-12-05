import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

def load_model():
    """Load LlamaGuard model"""
    model_id = "meta-llama/Meta-Llama-Guard-2-8B"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    
    print(f"Loading model {model_id} on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        torch_dtype=dtype, 
        device_map=device
    )
    return tokenizer, model, device

def moderate(chat, tokenizer, model, device):
    """Use LlamaGuard to judge if conversation is safe"""
    input_ids = tokenizer.apply_chat_template(chat, return_tensors="pt").to(device)
    output = model.generate(
        input_ids=input_ids, 
        max_new_tokens=100, 
        pad_token_id=0
    )
    prompt_len = input_ids.shape[-1]
    result = tokenizer.decode(output[0][prompt_len:], skip_special_tokens=True)
    return result.strip()

def is_harmful(category_dict):
    """Judge if sample is harmful (any category is true)"""
    return any(category_dict.values())

def load_dataset(file_path):
    """Load dataset"""
    print(f"Loading dataset from {file_path}...")
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"Loaded {len(data)} samples")
    return data

def main():
    # Load model
    tokenizer, model, device = load_model()
    
    # Load dataset
    dataset_path = "beavertails_harmful_train.json"
    data = load_dataset(dataset_path)
    
    # Only take the first 1000 samples
    data = data[:1000]
    
    # Statistics variables
    total_samples = len(data)
    harmful_samples = 0
    false_negatives = 0  # Actually harmful but judged as safe
    
    print(f"Starting dataset evaluation (first {total_samples} samples)...")
    
    for i, sample in enumerate(tqdm(data, desc="Processing samples")):
        prompt = sample["prompt"]
        response = sample["response"]
        category = sample["category"]
        
        # Judge if actually harmful
        actual_harmful = is_harmful(category)
        if actual_harmful:
            harmful_samples += 1
        
        # Use LlamaGuard to judge
        chat = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response}
        ]
        
        try:
            guard_result = moderate(chat, tokenizer, model, device)
            print(guard_result)
            # LlamaGuard returns "safe" for safe, others for unsafe
            guard_safe = guard_result.lower() == "safe"
            
            # If actually harmful but judged as safe, it's a misjudgment
            if actual_harmful and guard_safe:
                false_negatives += 1
                
        except Exception as e:
            print(f"Error processing sample {i}: {e}")
            continue
    
    # Calculate misjudgment rate
    if harmful_samples > 0:
        false_negative_rate = false_negatives / harmful_samples
        print(f"\n=== Evaluation Results ===")
        print(f"Total samples: {total_samples}")
        print(f"Actually harmful samples: {harmful_samples}")
        print(f"Samples misjudged as safe: {false_negatives}")
        print(f"Misjudgment rate: {false_negative_rate:.4f} ({false_negative_rate*100:.2f}%)")
    else:
        print("No harmful samples found")

if __name__ == "__main__":
    main()