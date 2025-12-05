import argparse
import math
import os
from datetime import datetime
import torch
import jsonlines
from transformers.trainer import get_scheduler

from bds.datasets import SFTDataset, SFTDatasetIndexed
from bds.models import Actor, TrainableTensorModule
from bds.models.actor import DataScorerModel
from bds.trainer import DBSTrainer
from bds.utils import blending_datasets, get_strategy, get_tokenizer

class DummyOptimizer(torch.optim.Optimizer):
    def __init__(self):
        dummy_param = torch.nn.Parameter(torch.zeros(1, requires_grad=True))

        # Explicitly specify that param_groups also contains "lr"
        super().__init__([{"params": [dummy_param], "lr": 1e-3}], {}) 

    def step(self, closure=None):
        pass

    def zero_grad(self, set_to_none=False):
        pass

class DummyScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer):
        # As long as optimizer is a valid object (can be DummyOptimizer)
        super().__init__(optimizer)

    def step(self):
        pass

def train(args):
    
    # configure strategy
    strategy = get_strategy(args)
    strategy.setup_distributed()

    # configure model
    # load huggingface model
    model = Actor(
        args.pretrain,
        use_flash_attention_2=args.flash_attn,
        bf16=args.bf16,
        load_in_4bit=args.load_in_4bit,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=args.target_modules,
        lora_dropout=args.lora_dropout,
        ds_config=strategy.get_ds_train_config(is_actor=True),
        gradient_checkpointing=args.gradient_checkpointing,
    )

    # load pretrained LoRA weights if specified
    if args.pretrained_lora_path:
        if args.lora_rank > 0:
            # Merge pretrained LoRA weights into base model and reinitialize new LoRA for training
            model.merge_lora_and_reinit(
                lora_path=args.pretrained_lora_path,
                lora_rank=args.lora_rank,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
                target_modules=args.target_modules,
                ds_config=strategy.get_ds_train_config(is_actor=True)
            )
        else:
            raise ValueError("Cannot load pretrained LoRA weights when lora_rank is 0. "
                           "Please set lora_rank > 0 to enable LoRA.")

    # configure tokenizer
    tokenizer = get_tokenizer(args.pretrain, model.model, "right", strategy, use_fast=not args.disable_fast_tokenizer)

    strategy.print(model)
    
    # load weights for ref model
    if args.ref_constant:
        ref_model = Actor(
            args.pretrain,
            use_flash_attention_2=args.flash_attn,
            bf16=args.bf16,
            load_in_4bit=args.load_in_4bit,
            ds_config=strategy.get_ds_eval_config(offload=args.ref_offload),
        )
        if args.ref_offload:
            ref_model._offload = True
        get_tokenizer(args.pretrain, ref_model.model, "right", strategy, use_fast=not args.disable_fast_tokenizer)
        print("\n TODO: implement ref regularization. Right now there will be no reg.\n")
    else:
        ref_model=None

    # configure optimizer
    optim = strategy.create_optimizer(model, lr=args.learning_rate, betas=(0.9, 0.95), weight_decay=args.l2)

    # prepare for data and dataset
    train_data, eval_data = blending_datasets(
        args.dataset, args.dataset_probs, strategy, args.seed, max_count=args.max_samples, start_ind=5000
    )
    
    new_data = blending_datasets(
            args.new_dataset,
            args.new_dataset_probs,
            strategy,
            args.seed,
            return_eval=False,
            max_count=args.max_new_samples,
            start_ind=args.new_dataset_start_ind
        )
    print(len(train_data),"$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
    train_data = train_data.select(range(min(args.max_samples, len(train_data))))
    eval_data = eval_data.select(range(min(args.max_samples, len(eval_data))))
    new_data = new_data.select(range(min(args.max_new_samples, len(new_data))))

    train_dataset = SFTDataset(
        train_data,
        tokenizer,
        args.max_len,
        strategy,
        pretrain_mode=args.pretrain_mode,
        input_template=args.input_template,
    )
    eval_dataset = SFTDataset(
        eval_data,
        tokenizer,
        args.max_len,
        strategy,
        pretrain_mode=args.pretrain_mode,
        input_template=args.input_template,
    )

    new_dataset = SFTDatasetIndexed(
        new_data,
        tokenizer,
        args.max_len,
        strategy,
        pretrain_mode=args.pretrain_mode,
        input_template=args.input_template,
    )
    total_num=new_dataset.__len__()
    ft_num=0
    harmful_num=0
    system_num=0
    ft_idx_in_dataset=[]
    harmful_idx_in_dataset=[]
    os.makedirs(args.save_path, exist_ok=True)
    new_dataset_path=args.save_path+'/new_dataset.json'
    item_list=[]
    for idx in range(new_dataset.__len__()):
        item={}
        source=new_dataset.__getitem__(idx)[-1]
        idx_in_dataset=new_dataset.__getitem__(idx)[0]
        if source=='ft':
            ft_num=ft_num+1
            ft_idx_in_dataset.append(idx_in_dataset)
        elif source=="harmful":
            harmful_num=harmful_num+1
            harmful_idx_in_dataset.append(idx_in_dataset)
        elif source=="system":
            system_num=system_num+1
        item["index"]=new_dataset.ids[idx]
        item["prompt"]=new_dataset.prompts[idx]
        item["response"]=new_dataset.responses[idx]
        item["source"]=source
        item_list.append(item)
    import json
    with open(new_dataset_path,'w',encoding='utf-8') as f:
        json.dump(item_list,f,indent=4)
    print("total_num=",total_num)
    print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
    print("ft_num=",ft_num)
    print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
    print("harmful_num=",harmful_num)
    print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
    print("system_num=",system_num)
    print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
    
    # initilize data scheduler
    if args.scheduler_name=="scalar":
        p = TrainableTensorModule(size=new_dataset.__len__(),activation=args.scheduler_activation,init=args.init)
        p_opt=strategy.create_optimizer(p, lr=args.scheduler_learning_rate, betas=(0.9, 0.95), weight_decay=0.)
    elif args.scheduler_name=="neural":
        p = DataScorerModel(model_type="fairseq",device=torch.device("cuda:0"), base_model_path=args.scorer_model_path, size=new_dataset.__len__(), bias=True, encoding="mean", activation=args.scheduler_activation,trainable=args.neural_trainable,pretrained=args.scorer_model_pretrained)
        if args.neural_trainable:
            p_opt=strategy.create_optimizer(p, lr=args.scheduler_learning_rate, betas=(0.9, 0.95), weight_decay=0.)
        else:
            p_opt=DummyOptimizer()
    else:
        raise NotImplementedError

    

    train_dataloader = strategy.setup_dataloader(
        train_dataset, args.micro_train_batch_size, True, True, train_dataset.collate_fn,drop_last=False
    )
    eval_dataloader = strategy.setup_dataloader(
        eval_dataset, args.micro_train_batch_size, True, False, eval_dataset.collate_fn,drop_last=False
    )
    new_dataloader = strategy.setup_dataloader(
        new_dataset, args.micro_train_batch_size, True, True, new_dataset.collate_fn,drop_last=False
    )
    if args.scheduler_name=="neural":
        temp_dataloader=strategy.setup_dataloader(
        new_dataset, args.micro_train_batch_size, False, False, new_dataset.collate_fn,drop_last=False
    )
        p.dataloader=temp_dataloader

    # scheduler
    num_update_steps_per_epoch = len(train_dataloader) // strategy.accumulated_gradient
    max_steps = math.ceil(args.max_epochs * num_update_steps_per_epoch)

    scheduler = get_scheduler(
        args.lr_scheduler,
        optim,
        num_warmup_steps=math.ceil(max_steps * args.warmup),
        num_training_steps=max_steps,
    )
    if args.neural_trainable:
        p_scheduler = get_scheduler(
            args.scheduler_lr_scheduler,
            p_opt,
            num_warmup_steps=math.ceil(max_steps * 0.03),
            num_training_steps=max_steps,
        )
    else:
        p_scheduler = DummyScheduler(DummyOptimizer())

    # gradient_checkpointing
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": args.gradient_checkpointing_use_reentrant}
        )

    # # prepare models
    # (model, optim, scheduler) = strategy.prepare((model, optim, scheduler))
    # strategy prepare
    if ref_model:
        ((model, optim, scheduler),(p,p_opt,p_scheduler),ref_model) = strategy.prepare((model, optim, scheduler),(p,p_opt,p_scheduler),ref_model)
    else:
        ((model, optim, scheduler),(p,p_opt,p_scheduler)) = strategy.prepare((model, optim, scheduler),(p,p_opt,p_scheduler))
    for name, param in model.model.named_parameters():
        if "lora" in name.lower():  # Filter parameters containing 'lora'
            print(f"Parameter: {name}")
            print(f" - Dtype: {param.dtype}")  # Data type
            assert param.dtype == torch.float32, f"LoRA parameter {name} should be torch.float32 but got {param.dtype}"
            print(f" - Shape: {param.shape}")  # Shape (verify if size meets expectations)
    if args.bf16:
        for name, param in model.named_parameters():
            if "lora" not in name.lower():  # Check if the parameter is not part of LoRA
                param.data = param.data.to(torch.bfloat16)
                print(f"Converted {name} to bf16.")
            else:
                print(f"Skipped {name}, as it is part of LoRA.")

    # load checkpoint
    if args.load_checkpoint:
        strategy.print("Load checkpoint: ", args.save_path)
    if args.save_path:
        os.makedirs(args.save_path, exist_ok=True)

    # configure Trainer
    trainer = DBSTrainer(
        model=model,
        ref_model=ref_model,
        ref_constant=args.ref_constant,
        strategy=strategy,
        optim=optim,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        new_dataloader=new_dataloader,
        p=p,
        p_opt=p_opt,
        p_scheduler=p_scheduler, 
        scheduler=scheduler,
        max_norm=args.max_norm,
        pretrain_mode=args.pretrain_mode,
        batch_size=args.train_batch_size,
        max_epochs=args.max_epochs,
        tokenizer=tokenizer,
        ft_idx_in_dataset=ft_idx_in_dataset,
        harmful_idx_in_dataset=harmful_idx_in_dataset
    )

    trainer.fit(args)
    
    # save model checkpoint after fitting on only rank0
    if args.save_path:
        strategy.save_model(model, tokenizer, args.save_path)
    if args.scheduler_name=="scalar":
        p_tensor = trainer.p.logits
    else:
        p_tensor=trainer.p.logit_tensor
    if strategy.is_rank_0():
        print(p_tensor)
        torch.save(p_tensor, args.save_path+"/"+args.scheduler_name+"_"+ args.scheduler_activation+".pt")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrain", type=str, default="bigscience/bloomz-1b7")
    parser.add_argument("--dataset", type=str, default="Dahoas/full-hh-rlhf")
    parser.add_argument("--dataset_probs", type=str, default="1.0", help="sampling probs for datasets")
    parser.add_argument("--new_dataset_probs", type=str, default="1.0", help="sampling probs for datasets")
    parser.add_argument("--save_steps", type=int, default=-1)
    parser.add_argument("--logging_steps", type=int, default=1)
    parser.add_argument("--eval_steps", type=int, default=-1)
    parser.add_argument("--ckpt_path", type=str, default="./ckpt/checkpoints_sft")
    parser.add_argument("--max_ckpt_num", type=int, default=3)
    parser.add_argument("--max_ckpt_mem", type=int, default=1000)  # 1000GB
    parser.add_argument("--max_epochs", type=int, default=2)
    parser.add_argument("--micro_train_batch_size", type=int, default=8)
    parser.add_argument("--train_batch_size", type=int, default=128)
    parser.add_argument("--max_samples", type=int, default=1000000)
    parser.add_argument("--max_new_samples", type=int, default=1000000)
    parser.add_argument("--new_dataset_start_ind", type=str, default="0", help="start indices for new datasets, comma-separated list or single number")
    parser.add_argument("--max_len", type=int, default=512)
    parser.add_argument("--max_norm", type=float, default=1.0)
    parser.add_argument("--l2", type=float, default=0)
    parser.add_argument("--lr_scheduler", type=str, default="cosine")
    parser.add_argument("--load_checkpoint", action="store_true", default=False)
    parser.add_argument("--pretrain_mode", action="store_true", default=False)
    parser.add_argument("--warmup", type=float, default=0.03)

    parser.add_argument("--gradient_checkpointing", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for deepspeed")
    parser.add_argument("--zero_stage", type=int, default=2)
    parser.add_argument("--bf16", action="store_true", default=False)
    parser.add_argument("--learning_rate", type=float, default=2e-6)
    parser.add_argument("--zpg", type=int, default=1, help="ZeRO++ max partition size")
    parser.add_argument("--adam_offload", action="store_true", default=False)
    parser.add_argument("--flash_attn", action="store_true", default=False)
    parser.add_argument("--aux_loss_coef", type=float, default=0)
    parser.add_argument("--grad_accum_dtype", type=str, default=None)
    parser.add_argument("--disable_trace_cache", action="store_true", default=False)
    parser.add_argument("--load_in_4bit", action="store_true", default=False)
    parser.add_argument("--lora_rank", type=int, default=0)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--target_modules", type=str, nargs="*", default="all-linear")
    parser.add_argument("--lora_dropout", type=float, default=0)
    parser.add_argument("--pretrained_lora_path", type=str, default=None,
                       help="Path to pretrained LoRA weights directory. If specified, will load pretrained LoRA weights")
    parser.add_argument("--input_template", type=str, default=None)
    parser.add_argument("--gradient_checkpointing_use_reentrant", action="store_true")
    parser.add_argument("--disable_fast_tokenizer", action="store_true", default=False)
    parser.add_argument("--master_port", type=str, default="29500")
    # data scheduler parameters
    parser.add_argument("--new_dataset", type=str, default=None)
    parser.add_argument("--scheduler_learning_rate", type=float, default=1e-2)
    parser.add_argument("--scheduler_activation", type=str, default="softmax")
    parser.add_argument("--entropy", type=float, default=0)
    parser.add_argument("--w_ratio_prior", type=float, default=0)
    parser.add_argument("--w_prior_sigma", type=float, default=0.1)
    parser.add_argument("--gaussian_prior", type=float, default=0)
    parser.add_argument("--gaussian_mu", type=float, default=0)
    parser.add_argument("--gaussian_sigma", type=float, default=1)
    parser.add_argument("--scheduler_name", type=str, default="sft_selection_logits")
    parser.add_argument("--ref_constant", type=float, default=0.)
    parser.add_argument("--upperlevel_weight", type=float, default=0.9)
    parser.add_argument("--upperlevel_weight_decay", type=float, default=0.1)
    parser.add_argument("--save_path", type=str, default=None)
    parser.add_argument("--scheduler_lr_scheduler", type=str, default="constant")
    parser.add_argument("--scorer_model_path", type=str, default=None)
    parser.add_argument("--scorer_model_pretrained", type=str, default=None)
    parser.add_argument("--init", type=float, default=0.1)
    parser.add_argument("--noweight", action="store_true", default=False)
    
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ("yes", "true", "t", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("Boolean value expected.")
    
    parser.add_argument("--neural_trainable", type=str2bool, default=True, help="Whether to enable neural training.")
    # custom dataset key name
    parser.add_argument("--input_key", type=str, default=None)
    parser.add_argument("--output_key", type=str, default=None)

    
    # wandb pamameters
    parser.add_argument("--use_wandb", type=str, default=None)
    parser.add_argument("--wandb_org", type=str, default=None)
    parser.add_argument("--wandb_group", type=str, default=None)
    parser.add_argument("--wandb_project", type=str, default="bds")
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default="sft_%s" % datetime.now().strftime("%m%dT%H:%M"),
    )

    args = parser.parse_args()
    
    # Validate LoRA loading arguments
    if args.pretrained_lora_path:
        if args.lora_rank <= 0:
            raise ValueError("--lora_rank must be > 0 when loading pretrained LoRA weights")
    
    args.wandb_run_name=args.save_path+"_%s" % datetime.now().strftime("%m%dT%H:%M")
    torch.cuda.empty_cache()
    if args.scheduler_name=="scalar":
        assert args.scheduler_learning_rate==5e-3,"Learning rate error!"
    elif args.scheduler_name=="neural":
        # assert args.scheduler_learning_rate!=5e-3,"Learning rate error!"
        pass
    train(args)
