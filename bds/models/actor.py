from typing import Optional, Tuple, Union

import deepspeed
import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, TaskType, get_peft_model, PeftModel
from peft.tuners.lora import LoraLayer
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, PreTrainedModel,AutoConfig,AutoModel
from transformers.deepspeed import HfDeepSpeedConfig
import os
from openrlhf.models.utils import log_probs_from_logits
import json


class BertBaseModel(nn.Module):
    def __init__(self, device, base_model_path, bf16):
        super().__init__()
        self.model_path = base_model_path
        dtype = torch.float16 if bf16 else torch.float32
        self.model = AutoModel.from_pretrained(self.model_path, device_map={"": device}, torch_dtype=dtype)
    
    def forward(self, input_ids, attention_mask, use_cache=False):
        output1 = self.model(input_ids=input_ids[:, :512], attention_mask=attention_mask[:, :512], return_dict=True)
        output2 = self.model(input_ids=input_ids[:, 512:], attention_mask=attention_mask[:, 512:], return_dict=True)
        h1 = output1["last_hidden_state"]
        h2 = output2["last_hidden_state"]
        
        h = torch.cat([h1, h2], dim=1)
        return {"last_hidden_state": h}
    
# Load and save model
def get_model( device, model_path=None, config=None, model_cls=None, bf16=True):
    import time
    import torch.distributed as dist
    model_path = model_path
    config = AutoConfig.from_pretrained(model_path)
    config.is_model_parallel = False
    model_cls = model_cls if model_cls is not None else AutoModelForCausalLM
    dtype = torch.float16 if bf16 else torch.float32
    model = model_cls.from_pretrained(model_path, config=config, device_map={"": device}, torch_dtype=dtype)
    return model

class DataScorerModel(nn.Module):
    def __init__(self, model_type,device, base_model_path, size, bias=False, encoding="mean", activation="linear",dataloader=None,trainable=True,pretrained=None):
        super().__init__()
        self.base_model_path = base_model_path
        self.config = AutoConfig.from_pretrained(base_model_path)
        if model_type in ["bert", "roberta"]:
            self.base_model = BertBaseModel(device, base_model_path)
        else:
            self.base_model = get_model(device, base_model_path, self.config, model_cls=AutoModel)
        self.head = nn.Linear(self.config.hidden_size, 1, bias=bias)
        self.size=size
        self.bias = bias
        self.activation = activation
        self.encoding = encoding
        self.dataloader=dataloader
        self.norma=None
        if pretrained:
            self.load_pretrained(pretrained)
            print(f"Load scorer from {pretrained}.")
        # Freeze all parameters of base_model
        for param in self.base_model.parameters():
            param.requires_grad = True if trainable else False
        # Ensure head layer parameters are trainable
        for param in self.head.parameters():
            param.requires_grad = True if trainable else False

        
        print(f"Data Scorer | Bias: {bias}, Encoding: {encoding}, activation: {activation}")
    def update_logit_softmax(self):
        logit_dict={}
        score_dict={}
        for (ide, new_prompts_id_len, new_input, new_attention_masks, _, _) in self.dataloader:
            new_inputs = new_input.squeeze(1).to(torch.cuda.current_device())
            new_attention_mask = new_attention_masks.squeeze(1).to(torch.cuda.current_device())
            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    logit_batch=self.get_logit(new_inputs,new_attention_mask,None)
                    for temp, l in enumerate(logit_batch):
                        logit_dict[ide[temp]]=l
        sorted_items_logit = sorted(logit_dict.items())
        logit_list=[v[-1] for v in sorted_items_logit]
        self.logit_tensor = torch.tensor(logit_list)
        for (ide, new_prompts_id_len, new_input, new_attention_masks, _, _) in self.dataloader:
            new_inputs = new_input.squeeze(1).to(torch.cuda.current_device())
            new_attention_mask = new_attention_masks.squeeze(1).to(torch.cuda.current_device())
            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    score_batch,_=self.get_score(new_inputs,new_attention_mask,None,ide)
                    for temp, s in enumerate(score_batch):
                        score_dict[ide[temp]]=s
        sorted_items_score = sorted(score_dict.items())
        score_list=[v[-1] for v in sorted_items_score]
        self.score_tensor = torch.tensor(score_list)

    def get_logit(self, input_ids, attention_mask, pos=None):
        if pos is None:
            pos = attention_mask.sum(dim=1) - 1
        h = self.base_model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)["last_hidden_state"]
        if self.encoding == "mean":
            mask = (torch.arange(h.size(1), device=h.device)[None, :] <= pos[:, None]).to(h.dtype)
            origin_dtype = h.dtype
            h = h.float()
            h = torch.sum(h * mask[:, :, None], dim=1) / mask.sum(dim=1)[:, None]
            h = h.to(origin_dtype)            
        elif self.encoding == "last":
            h = torch.gather(h, 1, pos[:, None, None].expand(-1, -1, h.size(-1))).squeeze()
        elif self.encoding == "first":
            h = h[:, 0]
        else:
            raise ValueError("encoding should be mean, last, or first")  # Updated error message
        return self.head(h).squeeze()

    def get_score(self, input_ids, attention_mask, pos,ide_query):
        logit=self.get_logit(input_ids, attention_mask, pos)
        if self.activation == "linear":
            s = logit
        elif self.activation == "sigmoid":
            self.norma = 1
            s = torch.sigmoid(logit)
            s=self.norma*s
        elif self.activation=="softmax":
            # s=torch.zeros_like(logit)
            # for i, ide_query_i in enumerate(ide_query):
            #     logit_tensor_i=(self.logit_tensor).clone()
            #     logit_tensor_i[ide_query_i]=logit[i]
            #     activ = nn.Softmax(dim=0)
            #     self.norma = self.size
            #     logit_tensor_i_softmax=self.norma*activ(logit_tensor_i)
            #     s[i]=logit_tensor_i_softmax[ide_query_i]
            self.norma = len(logit)
            scores = torch.nn.functional.softmax(logit, dim=0)  # softmax over batch
            s = self.norma * scores
        else:
            raise ValueError("activation should be linear, sigmoid, or softmax")  # Updated error message

        if s.dim() == 0:
            s = s.unsqueeze(0)

        return s.float(),logit

    def forward(self, input_ids, attention_mask, pos,ide):
        s,logit = self.get_score(input_ids, attention_mask, pos,ide)
        return s,logit

    def save_pretrained(self, save_dir, **kawrgs):
        import json
        import os
        with open(os.path.join(save_dir, "config.json"), "w") as f:
            json.dump({
                "bias": self.bias,
                "encoding": self.encoding
            }, f, indent=4)
        torch.save(self.state_dict(), os.path.join(
            save_dir, "data_scorer_model.pt"))
    
    def inference(self, input_ids, attention_mask, pos,ide):
        with torch.no_grad():
            return self.get_score(input_ids, attention_mask, pos,ide)
    def negative_entropy(self, input_ids, attention_mask, pos,ide_query):
        # only for softmax!!
        logit_tensor_copy=(self.logit_tensor).clone()
        logit_query=self.get_logit(input_ids, attention_mask, pos)
        if self.activation == "linear":
            raise ValueError
        elif self.activation == "sigmoid":
            raise ValueError
        elif self.activation=="softmax":
            s=torch.zeros_like(logit_query)
            for i, ide_query_i in enumerate(ide_query):
                logit_tensor_copy[ide_query_i]=logit_query[i]
        else:
            raise ValueError("score_head should be linear or sigmoid")
        logp = F.log_softmax(logit_tensor_copy,dim=0,dtype=torch.float32)
        p = logp.exp()
        entropy = -(p * logp).sum()
        return -entropy
    def load_pretrained(self, path):
        # Get current model device
        device = next(self.parameters()).device
        # Load config.json
        config_path = os.path.join(path, "config.json")
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                config = json.load(f)
            self.bias = config.get("bias", self.bias)
            self.encoding = config.get("encoding", self.encoding)
            print(f"[load_pretrained] Loaded config: bias={self.bias}, encoding={self.encoding}")
        else:
            print(f"[load_pretrained] No config.json found in {path}, skipping config update.")
        # Load weights to current model device
        weights_path = os.path.join(path, "data_scorer_model.pt")
        if os.path.exists(weights_path):
            state_dict = torch.load(weights_path, map_location=device)
            self.load_state_dict(state_dict)
            print(f"[load_pretrained] Loaded weights from {weights_path} to device {device}")
        else:
            raise FileNotFoundError(f"No data_scorer_model.pt found in {path}")
        
class TrainableTensorModule(nn.Module):


    
    def __init__(self, size, activation='softmax',init=0.1):
        super(TrainableTensorModule, self).__init__()
        
        # Use nn.Parameter to make 'tensor' trainable
        if activation=='softmax':
            self.logits = nn.Parameter(torch.tensor(torch.ones(size)*init)).float()
            self.activ = nn.Softmax(dim=0)
            self.norma = size
        elif activation=='sigmoid':
            self.logits = nn.Parameter(torch.tensor(-torch.ones(size)*1e-4)).float()
            self.activ = nn.Sigmoid()
            self.norma = 2
        elif activation=='linear':
            self.logits = nn.Parameter(torch.tensor(-torch.ones(size)*1e-4)).float()
            self.activ = lambda x: x
            self.norma = 1
        else:
            raise NotImplementedError
    def forward(self):
        # You can perform operations involving 'p' in the forward method
        # For example, you might use 'p' in a neural network layer
        return self.norma*self.activ(self.logits) 
    # if weight else F.log_softmax(self.logits,dim=0,dtype=torch.float32)

    def negative_entropy(self):
        # only for softmax!!
        logp = F.log_softmax(self.logits,dim=0,dtype=torch.float32)
        p = self.forward()
        p = p/self.norma
        return (p*logp).sum()
    def w_prior(self, N_t, benign_ratio, sigma=0.1):
        # Step 1: Compute sum of w
        sum_w = torch.sum(self.forward())
        # Step 2: Compute floor(N_t * beta)
        floor_value = torch.floor(torch.tensor(N_t * benign_ratio))
        # Step 3: Compute the numerator (sum(w) - floor(N_t * beta))^2
        numerator = (sum_w - floor_value) ** 2
        # Step 4 Compute the denominator (2 * sigma^2)
        denominator = 2 * (sigma ** 2)
        # Step 5: Compute the final result
        result = torch.exp(-numerator / denominator)
        return result

    def gaussian_prior(self,gaussian_mu,gaussian_sigma):
        w = self.logits  # Your vector
        mu = gaussian_mu  # Mean of the Gaussian prior
        sigma = gaussian_sigma  # Standard deviation of the Gaussian prior
        
        # Negative log-likelihood of Gaussian
        gaussian_penalty = 0.5 * ((w - mu) / sigma)**2 + torch.log(sigma * torch.sqrt(torch.tensor(2 * torch.pi)))
        
        # Sum the penalties for all components
        return gaussian_penalty.sum()

            

class Actor(nn.Module):
    """
    Actor model base class.

    Args:
        model (nn.Module): Actor Model.
        lora_rank (int): LoRA rank.
        lora_train_bias (str): LoRA bias training mode.
    """

    def __init__(
        self,
        pretrain_or_model,
        use_flash_attention_2=False,
        bf16=True,
        load_in_4bit=False,
        lora_rank=0,
        lora_alpha=16,
        lora_dropout=0,
        target_modules=None,
        ds_config=None,
        device_map=None,
        gradient_checkpointing=False, #bak
        **kwargs,
    ) -> None:
        super().__init__()

        if isinstance(pretrain_or_model, str):
            attn_implementation = "flash_attention_2" if use_flash_attention_2 else "eager"

            # Note: dschf is defined in function scope to avoid global effects
            # https://huggingface.co/docs/transformers/deepspeed#non-trainer-deepspeed-integration
            if ds_config is not None and ds_config["zero_optimization"]["stage"] == 3:
                dschf = HfDeepSpeedConfig(ds_config)
            else:
                dschf = None

            if load_in_4bit:
                assert bf16, "we only support bnb_4bit_compute_dtype = bf16"
                nf4_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                )
            else:
                nf4_config = None  
        
            self.model = AutoModelForCausalLM.from_pretrained(
                pretrain_or_model,
                trust_remote_code=True,
                attn_implementation=attn_implementation,
                quantization_config=nf4_config,
                # torch_dtype=torch.bfloat16 if bf16 else "auto",
                torch_dtype= torch.float32,
                device_map=device_map,
                use_cache = not gradient_checkpointing, #bak
            )


            # LoRA
            if lora_rank > 0:
                # https://github.com/huggingface/peft/issues/137
                self.model.enable_input_require_grads()
                lora_config = LoraConfig(
                    task_type=TaskType.CAUSAL_LM,
                    r=lora_rank,
                    lora_alpha=lora_alpha,
                    target_modules=target_modules,
                    lora_dropout=lora_dropout,
                    bias="none",
                )
                self.model = get_peft_model(self.model, lora_config)
                self.model.print_trainable_parameters() #bak
                
                

                if load_in_4bit:
                    for name, module in self.model.named_modules():
                        if isinstance(module, LoraLayer):
                            module = module.to(torch.bfloat16)
                        if "norm" in name:
                            module = module.to(torch.float32)
                        if "lm_head" in name or "embed_tokens" in name:
                            if hasattr(module, "weight"):
                                module = module.to(torch.bfloat16)
                # for name, param in self.model.named_parameters():
                #     if "lora" in name.lower():  # Filter parameters containing 'lora'
                #         print(f"Parameter: {name}")
                #         print(f" - Dtype: {param.dtype}")  # Data type
                #         assert param.dtype == torch.float32, f"LoRA parameter {name} should be torch.float32 but got {param.dtype}"
                #         print(f" - Shape: {param.shape}")  # Shape (verify if size meets expectations)

            # MoE - balancing loss
            model_config = self.model.config.to_dict()
            if "output_router_logits" in model_config:
                print("[MoE] set output_router_logits as True")
                
        else:
            self.model = pretrain_or_model

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, **kwargs) -> Union[
        Tuple[torch.LongTensor, torch.LongTensor],
        Tuple[torch.LongTensor, torch.LongTensor, torch.BoolTensor],
    ]:
        generate_args = {
            "input_ids": input_ids,
            "top_k": kwargs.get("top_k", None),
            "top_p": kwargs.get("top_p", None),
            "do_sample": kwargs.get("do_sample", True),
            "early_stopping": True,
            "temperature": kwargs.get("temperature", 1),
            "use_cache": True,
            "num_beams": kwargs.get("num_beams", 1),
            "attention_mask": kwargs.get("attention_mask"),
            "eos_token_id": kwargs.get("eos_token_id"),
            "pad_token_id": kwargs.get("pad_token_id"),
            "min_new_tokens": kwargs.get("min_new_tokens ", 1),
        }

        if kwargs.get("max_new_tokens", None):
            generate_args["max_new_tokens"] = kwargs.get("max_new_tokens")
        if kwargs.get("max_length", None):
            generate_args["max_length"] = kwargs.get("max_length")

        # Call generate
        sequences = self.model.generate(**generate_args)

        # Prepare mask tensor
        eos_token_id = generate_args["eos_token_id"]
        pad_token_id = generate_args["pad_token_id"]

        return self.process_sequences(sequences, input_ids.size(1), eos_token_id, pad_token_id)

    def process_sequences(self, sequences: torch.Tensor, input_len, eos_token_id, pad_token_id):
        attention_mask = (sequences.ne(eos_token_id) & sequences.ne(pad_token_id)).to(dtype=torch.long)
        seq_length = attention_mask.size(1)

        # The following code is equivalent to:
        #
        # for i in range(attention_mask.size(0)):
        #     for t in reversed(range(seq_length)):
        #         if attention_mask[i][t] > 0.5:
        #             attention_mask[i][min(t + 1, seq_length - 1)] = True
        #             sequences[i][min(t + 1, seq_length - 1)] = eos_token_id
        #             break
        #
        eos_indices = seq_length - attention_mask.long().fliplr().argmax(dim=1, keepdim=True).clamp(min=1)
        attention_mask.scatter_(dim=1, index=eos_indices, value=1)
        sequences.scatter_(dim=1, index=eos_indices, value=eos_token_id)

        # in RL, state_i (current token) + action_i (next token) -> state_i+1 (next token)
        state_seq = sequences[:, input_len - 1 : -1]
        # we only calculate the loss of state_i != eos | pad
        action_mask = state_seq.ne(eos_token_id) & state_seq.ne(pad_token_id)
        return sequences, attention_mask, action_mask

    def forward(
        self,
        sequences: torch.LongTensor,
        num_actions: int = None,
        attention_mask: Optional[torch.Tensor] = None,
        return_output=False,
    ) -> torch.Tensor:
        """Returns action log probs"""
        # https://github.com/OpenLLMAI/OpenRLHF/issues/217
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
        output = self.model(sequences, attention_mask=attention_mask, position_ids=position_ids)
        log_probs = log_probs_from_logits(output["logits"][:, :-1, :], sequences[:, 1:])

        if return_output:
            return output if num_actions is None else (log_probs[:, -num_actions:], output)
        else:
            return log_probs[:, -num_actions:]

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs={"use_reentrant": False}):
        self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)

    def gradient_checkpointing_disable(self):
        self.model.gradient_checkpointing_disable()

    def print_trainable_parameters(self):
        self.model.print_trainable_parameters()

    def load_pretrained_lora(self, lora_path: str, is_trainable: bool = True):
        """
        Load pretrained LoRA weights from a path.
        
        Args:
            lora_path (str): Path to the pretrained LoRA weights directory.
            is_trainable (bool): Whether the loaded LoRA weights should be trainable. 
                               Default is True.
        
        Returns:
            None
        """
        if not os.path.exists(lora_path):
            raise FileNotFoundError(f"LoRA path does not exist: {lora_path}")
        
        # Check if the model is already a PeftModel
        if not isinstance(self.model, PeftModel):
            raise ValueError("Model must be a PeftModel to load LoRA weights. "
                           "Please initialize the model with LoRA first (lora_rank > 0).")
        
        # Load the pretrained LoRA weights
        self.model = PeftModel.from_pretrained(
            self.model,
            lora_path,
            is_trainable=is_trainable
        )
        
        print(f"Successfully loaded pretrained LoRA weights from: {lora_path}")
        print(f"LoRA weights are {'trainable' if is_trainable else 'not trainable'}")
        
        # Print trainable parameters after loading
        self.print_trainable_parameters()

    def merge_lora_and_reinit(self, lora_path: str, lora_rank: int, lora_alpha: int, 
                             lora_dropout: float, target_modules: list, ds_config=None):
        """
        Merge pretrained LoRA weights into base model and reinitialize a new LoRA for training.
        
        Args:
            lora_path (str): Path to the pretrained LoRA weights directory.
            lora_rank (int): Rank for the new LoRA.
            lora_alpha (int): Alpha for the new LoRA.
            lora_dropout (float): Dropout for the new LoRA.
            target_modules (list): Target modules for the new LoRA.
            ds_config: DeepSpeed config for the new LoRA.
        
        Returns:
            None
        """
        if not os.path.exists(lora_path):
            raise FileNotFoundError(f"LoRA path does not exist: {lora_path}")
        
        print(f"Merging pretrained LoRA weights from: {lora_path}")
        
        # Load the pretrained LoRA weights
        peft_model = PeftModel.from_pretrained(
            self.model,
            lora_path,
            is_trainable=False  # Set to False for merging
        )
        
        # Merge LoRA weights into base model
        print("Merging LoRA weights into base model...")
        merged_model = peft_model.merge_and_unload()
        
        # Replace the model with merged model
        self.model = merged_model
        
        print("Successfully merged LoRA weights into base model")
        if lora_rank > 0:
            # Reinitialize LoRA on the merged model
            print("Reinitializing LoRA for training...")
            self.model.enable_input_require_grads()
            
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=lora_rank,
                lora_alpha=lora_alpha,
                target_modules=target_modules,
                lora_dropout=lora_dropout,
                bias="none",
            )
            
            self.model = get_peft_model(self.model, lora_config)
            self.model.print_trainable_parameters()
        
        # # Apply DeepSpeed config if provided
        # if ds_config:
        #     self.model = self.model.to(dtype=torch.float32)
        #     # Ensure LoRA parameters remain float32 after converting to bfloat16
        #     for name, param in self.model.named_parameters():
        #         if "lora" in name.lower():
        #             param.data = param.data.to(torch.float32)
        
        print("Successfully reinitialized LoRA for training")
        self.print_trainable_parameters()



if __name__=="__main__":
    scoree=DataScorerModel(device=torch.device("cuda:0"),base_model_path=None,bias=True,encoding="mean",head_type="linear")
    # from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
    # import os
    # os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    # os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

    # name = "KoboldAI/fairseq-dense-125M"
    # save_name = "faireq/125M"
    # tokenizer = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
    # tokenizer.save_pretrained(f"./{save_name}/")

    # model = AutoModelForCausalLM.from_pretrained(name, trust_remote_code=True)
    # model.save_pretrained(f"./{save_name}/", safe_serialization=False)