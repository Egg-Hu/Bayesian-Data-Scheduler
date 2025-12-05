from abc import ABC
import os

import torch
from torch.optim import Optimizer
from torch.utils.data import DistributedSampler
from tqdm import tqdm
from openrlhf.models import GPTLMLoss
from bds.models import BatchCrossEntropyLoss


class DBSTrainer(ABC):

    def __init__(
        self,
        model,
        ref_model,
        ref_constant,
        strategy,
        optim: Optimizer,
        train_dataloader,
        eval_dataloader,
        new_dataloader,
        p,
        p_opt: Optimizer,
        p_scheduler, 
        scheduler,
        max_norm: float = 1,
        pretrain_mode: bool = False,
        batch_size: int = 1,
        max_epochs: int = 2,
        tokenizer=None,
        ft_idx_in_dataset=[],
        harmful_idx_in_dataset=[]
    ) -> None:
        super().__init__()
        self.strategy = strategy
        self.epochs = max_epochs
        self.batch_size = batch_size
        self.max_norm = max_norm
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.new_dataloader = new_dataloader
        self.scheduler = scheduler
        self.pretrain_mode = pretrain_mode
        self.model = model
        self.ref_model = ref_model
        self.ref_constant = ref_constant
        self.tokenizer = tokenizer
        self.optimizer = optim
        self.args = strategy.args
        self.ul_weight = self.args.upperlevel_weight
        self.ul_weight_decay = self.args.upperlevel_weight_decay
        self.fix_dataloader=(self.new_dataloader)

        self.loss_fn = GPTLMLoss()
        self.batch_loss_fn = BatchCrossEntropyLoss()
        
        # data selection policy param
        self.p = p
        if self.args.scheduler_name=="neural":
            self.p.update_logit_softmax()
        self.p_opt = p_opt
        self.p_scheduler = p_scheduler
        self.ft_idx_in_dataset=ft_idx_in_dataset
        self.harmful_idx_in_dataset=harmful_idx_in_dataset

        # Mixtral 8*7b
        self.aux_loss = self.args.aux_loss_coef > 1e-8

        # wandb setting
        self._wandb = None
        if self.strategy.args.use_wandb and self.strategy.is_rank_0():
            import wandb
            import os

            self._wandb = wandb
            if not wandb.api.api_key:
                wandb.login(key=strategy.args.use_wandb)
            run = wandb.init(
                entity=strategy.args.wandb_org,
                project=strategy.args.wandb_project,
                group=strategy.args.wandb_group,
                name=strategy.args.wandb_run_name,
                config=strategy.args.__dict__,
                reinit=True,
            )
            
            # Save wandb run ID to file
            wandb_id_file = os.path.join(strategy.args.save_path, "wandb_run_id.txt")
            os.makedirs(os.path.dirname(wandb_id_file), exist_ok=True)
            with open(wandb_id_file, 'w') as f:
                f.write(run.id)
            print(f"Saved wandb run ID {run.id} to {wandb_id_file}")

    def fit(self, args):
        # get eval and save steps
        if args.eval_steps == -2:
            args.eval_steps = float("inf")  # do not eval ckpt
        if args.save_steps == -2:
            args.save_steps = float("inf")  # do not save ckpt

        global_step = 1
        epoch_bar = tqdm(
            range(self.epochs),
            desc="Train epoch",
            disable=not self.strategy.is_rank_0(),
        )
        class RepeatedDataLoader:
            def __init__(self, dataloader):
                self.dataloader = dataloader
                self.iterator = iter(dataloader)  # Initialize iterator
            def __iter__(self):
                return self
            def __next__(self):
                try:
                    # Get next data
                    return next(self.iterator)
                except StopIteration:
                    # If iterator is exhausted, recreate it
                    self.iterator = iter(self.dataloader)
                    return next(self.iterator)
            def __getattr__(self, attr):
                # Delegate access to undefined attributes to internal dataloader
                return getattr(self.dataloader, attr)
        # Determine the shorter DataLoader
        len_train = len(self.train_dataloader)
        len_new = len(self.new_dataloader)
        if len_train < len_new:
            self.train_dataloader = RepeatedDataLoader(self.train_dataloader)
            len_per_epoch=self.new_dataloader.__len__()
        else:
            self.new_dataloader = RepeatedDataLoader(self.new_dataloader)
            len_per_epoch=self.train_dataloader.__len__()
        if args.eval_steps == -1:
            args.eval_steps = len_per_epoch
        if args.save_steps == -1:
            args.save_steps = len_per_epoch
        for epoch in range(self.epochs):
            if isinstance(self.train_dataloader.sampler, DistributedSampler):
                self.train_dataloader.sampler.set_epoch(epoch)
                self.new_dataloader.sampler.set_epoch(epoch)
            if epoch>=1:
                if self.ul_weight !=-1 and self.ul_weight!=-2 and self.ul_weight!=-3 and self.ul_weight!=-4:
                    self.ul_weight -= self.ul_weight_decay
                    print("\n UL weight now", self.ul_weight, end="\n")
            step_bar = tqdm(
                range(len_per_epoch),
                desc="Train step of epoch %d" % epoch,
                disable=not self.strategy.is_rank_0(),
            )

            # train
            self.model.train()
            self.p.train()
            loss_mean = 0
            gpt_loss_mean=0
            for (prompts_id_len, inputs, attention_masks, _), \
                (ide, new_prompts_id_len, new_input, new_attention_masks, _, _) \
                    in zip(self.train_dataloader,self.new_dataloader):
                inputs = inputs.squeeze(1).to(torch.cuda.current_device())
                attention_mask = attention_masks.squeeze(1).to(torch.cuda.current_device())
                output = self.model(inputs, attention_mask=attention_mask, return_output=True)
                # todo: implement ref regularization
                # if self.ref_model:
                #     with torch.no_grad():
                #         ref_output = self.ref_model(inputs, attention_mask=attention_mask, return_output=True) 
       
                new_inputs = new_input.squeeze(1).to(torch.cuda.current_device())
                new_attention_mask = new_attention_masks.squeeze(1).to(torch.cuda.current_device())
                new_output = self.model(new_inputs, attention_mask=new_attention_mask, return_output=True)
                with torch.no_grad():
                    if self.args.scheduler_name=="scalar":
                        batch_weights = self.p()[ide]
                    elif self.args.scheduler_name=="neural":
                        with torch.cuda.amp.autocast():
                            batch_weights = self.p(new_inputs,new_attention_mask,None,ide)[0]
                    else:
                        raise NotImplementedError
                    
                

                # loss function
                labels = torch.where(
                    attention_mask.bool(),
                    inputs,
                    self.loss_fn.IGNORE_INDEX,
                )
                new_labels = torch.where(
                    new_attention_mask.bool(),
                    new_inputs,
                    self.batch_loss_fn.IGNORE_INDEX,
                )
                # mixtral
                if self.aux_loss:
                    aux_loss = output.aux_loss
                else:
                    aux_loss = 0

                if not self.pretrain_mode:
                    for label, source_len, new_label, new_source_len in zip(labels, prompts_id_len, new_labels, new_prompts_id_len):
                        label[:source_len] = self.loss_fn.IGNORE_INDEX
                        new_label[:new_source_len] = self.batch_loss_fn.IGNORE_INDEX

                gpt_loss = self.batch_loss_fn(output.logits, labels, sequence_reduce="mean").mean(0)
                batch_gpt_loss = self.batch_loss_fn(new_output.logits, new_labels, sequence_reduce="mean")
                
                # if self.args.noweight:
                #     weighted_gpt_loss = (batch_gpt_loss).mean(0)
                # else:
                #     weighted_gpt_loss = (batch_weights * batch_gpt_loss).mean(0)
                
                # -1: weighted loss but no weight decay
                # -2: weighted loss but no weight decay and no alignment
                # -3: only sft on user data
                # -4: no weight but alignment (mix)
                if self.ul_weight ==-1:
                    weighted_gpt_loss = (batch_weights * batch_gpt_loss).mean(0)
                    loss =  gpt_loss + weighted_gpt_loss + aux_loss * self.args.aux_loss_coef
                elif self.ul_weight==-2:
                    weighted_gpt_loss = (batch_weights * batch_gpt_loss).mean(0)
                    loss = weighted_gpt_loss + aux_loss * self.args.aux_loss_coef
                elif self.ul_weight==-3:
                    weighted_gpt_loss = (batch_gpt_loss).mean(0)
                    loss = weighted_gpt_loss + aux_loss * self.args.aux_loss_coef
                elif self.ul_weight==-4:
                    weighted_gpt_loss = (batch_gpt_loss).mean(0)
                    loss =  gpt_loss + weighted_gpt_loss + aux_loss * self.args.aux_loss_coef
                else:
                    weighted_gpt_loss = (batch_weights * batch_gpt_loss).mean(0)
                    loss = self.ul_weight * gpt_loss + (1-self.ul_weight) * weighted_gpt_loss + aux_loss * self.args.aux_loss_coef
                    
                
                self.strategy.backward(loss, self.model, self.optimizer) 
                self.strategy.optimizer_step(self.optimizer, self.model, self.scheduler)
                if self.args.scheduler_name=="scalar":
                    scheduler_loss = (self.p()[ide]*batch_gpt_loss.detach()).mean()
                    if args.entropy!=0:
                        scheduler_loss=scheduler_loss+args.entropy*self.p.negative_entropy()
                    if args.w_ratio_prior!=0:
                        scheduler_loss=scheduler_loss+args.w_ratio_prior*self.p.w_prior(N_t=args.max_new_samples, benign_ratio=float(args.new_dataset_probs[0]), sigma=args.w_prior_sigma)
                    if args.gaussian_prior!=0:
                        scheduler_loss=scheduler_loss+args.gaussian_prior*self.p.gaussian_prior(args.gaussian_mu,args.gaussian_sigma)
                elif self.args.scheduler_name=="neural":
                    with torch.cuda.amp.autocast():
                        scheduler_loss = (self.p(new_inputs,new_attention_mask,None,ide)[0]*batch_gpt_loss.detach()).mean()
                        if args.entropy!=0:
                            scheduler_loss=scheduler_loss+args.entropy*self.p.negative_entropy(new_inputs,new_attention_mask,None,ide)
                        if args.w_ratio_prior!=0:
                            raise NotImplementedError
                        if args.gaussian_prior!=0:
                            raise NotImplementedError
                if self.args.scheduler_name!="neural" and self.args.neural_trainable!=False:
                    self.strategy.backward(scheduler_loss, self.p, self.p_opt)
                    self.strategy.optimizer_step(self.p_opt, self.p, self.p_scheduler)
                if self.args.scheduler_name=="neural" and self.args.neural_trainable!=False:
                    self.p.update_logit_softmax()
                gpt_loss_mean = gpt_loss_mean * 0.95 + 0.05 * gpt_loss.item()
                loss_mean = loss_mean * 0.95 + 0.05 * loss.item()
                logs_dict = {"safe_batch_loss": gpt_loss.item(), "safe_loss_running_mean": gpt_loss_mean,"ft_batch_loss":batch_gpt_loss.mean(0).item(),"weighted_ft_batch_loss":weighted_gpt_loss.item(), "all_loss_running_mean":loss_mean,"safe_weight":self.ul_weight if self.ul_weight !=-1 else 1}
                if self.aux_loss:
                    logs_dict["aux_loss"] = aux_loss.item()
                
                # logs/checkpoints/evaluation
                if self.args.scheduler_name=="scalar":
                    self._wandb.log({"train/score_step(pre)": self._wandb.Histogram(self.p.logits.detach().cpu().numpy())}, step=global_step)
                    self._wandb.log({"train/score_step(post)": self._wandb.Histogram(self.p().detach().cpu().numpy())}, step=global_step)
                elif self.args.scheduler_name=="neural":
                    self._wandb.log({"train/score_step(pre)": self._wandb.Histogram(self.p.logit_tensor.detach().cpu().numpy())}, step=global_step)
                    self._wandb.log({"train/score_step(post)": self._wandb.Histogram(self.p.score_tensor.detach().cpu().numpy())}, step=global_step)

                self.save_logs_and_checkpoints(args, global_step, step_bar, logs_dict)

                step_bar.update()
                global_step += 1
            self._wandb.log({
                "max_cuda_memory_allocated_GB": torch.cuda.max_memory_allocated() / 1024**3,
                "max_cuda_memory_reserved_GB": torch.cuda.max_memory_reserved() / 1024**3,
            },step=global_step)
            epoch_bar.update()

    # logs/checkpoints/evaluation
    def save_logs_and_checkpoints(self, args, global_step, step_bar, logs_dict={}):
        if global_step % args.logging_steps == 0:
            # step bar
            logs_dict = self.strategy.all_reduce(logs_dict)
            step_bar.set_postfix(logs_dict)

            # wandb
            if (
                self._wandb is not None
                and self.strategy.is_rank_0()
                and global_step % self.strategy.accumulated_gradient == 0
            ):
                if len(self.ft_idx_in_dataset)!=0 and self.args.scheduler_name=="scalar":
                    avg_ft=torch.mean(self.p()[self.ft_idx_in_dataset])
                    avg_harmful=torch.mean(self.p()[self.harmful_idx_in_dataset])
                    avg_ft_pre=torch.mean(self.p.logits[self.ft_idx_in_dataset])
                    avg_harmful_pre=torch.mean(self.p.logits[self.harmful_idx_in_dataset])
                    logs = {"train/%s" % k: v for k, v in {**logs_dict, "avg_ft":avg_ft,"avg_harmful":avg_harmful,"avg_ft(pre)":avg_ft_pre,"avg_harmful(pre)":avg_harmful_pre}.items()}
                    self._wandb.log({"score": self._wandb.Histogram(self.p.logits.detach().cpu().numpy())}, step=global_step)
                elif self.args.scheduler_name=="neural":
                    # weight_dic={}
                    # for (ide, new_prompts_id_len, new_input, new_attention_masks, _, _) in self.fix_dataloader:
                    #     new_inputs = new_input.squeeze(1).to(torch.cuda.current_device())
                    #     new_attention_mask = new_attention_masks.squeeze(1).to(torch.cuda.current_device())
                    #     with torch.cuda.amp.autocast():
                    #         for temp, w in enumerate(self.p.inference(new_inputs,new_attention_mask,None,ide)[0]):
                    #             weight_dic[ide[temp]]=w
                    # sorted_items = sorted(weight_dic.items())
                    # weight_list=[v[-1] for v in sorted_items]
                    # weight_tensor = torch.tensor(weight_list)
                    avg_ft=torch.mean(self.p.score_tensor[self.ft_idx_in_dataset])
                    avg_harmful=torch.mean(self.p.score_tensor[self.harmful_idx_in_dataset])
                    avg_ft_pre=torch.mean(self.p.logit_tensor[self.ft_idx_in_dataset])
                    avg_harmful_pre=torch.mean(self.p.logit_tensor[self.harmful_idx_in_dataset])
                    # logs = {"train/%s" % k: v for k, v in {**logs_dict, "avg_ft":avg_ft,"avg_harmful":avg_harmful}.items()}
                    logs = {"train/%s" % k: v for k, v in {**logs_dict, "avg_ft":avg_ft,"avg_harmful":avg_harmful,"avg_ft(pre)":avg_ft_pre,"avg_harmful(pre)":avg_harmful_pre}.items()}
                    self._wandb.log({"score": self._wandb.Histogram(self.p.score_tensor.numpy())}, step=global_step)
                else:
                    logs = {"train/%s" % k: v for k, v in {**logs_dict}.items()}
                self._wandb.log(logs, step=global_step)
                

        # eval
        if global_step % args.eval_steps == 0:
            if "sst2" in args.new_dataset:
                os.makedirs(args.save_path+"/sst2_steps/", exist_ok=True)
                # ft_score=evaluate_sst2(self.model,self.tokenizer,args.save_path+"/sst2_steps/"+f"{global_step}.json",100)
                # self._wandb.log({"eval/sst2":ft_score}, step=global_step)
            if "gsm8k" in args.new_dataset:
                os.makedirs(args.save_path+"/gsm8k_steps/", exist_ok=True)
                # ft_score=evaluate_gsm8k(self.model,self.tokenizer,args.save_path+"/gsm8k_steps/"+f"{global_step}.json",100)
                # self._wandb.log({"eval/gsm8k":ft_score}, step=global_step)
            if "agnews" in args.new_dataset:
                os.makedirs(args.save_path+"/agnews_steps/", exist_ok=True)
            if "alpaca" in args.new_dataset:
                os.makedirs(args.save_path+"/alpaca_steps/", exist_ok=True)
            if "beavertails" in args.dataset:
                os.makedirs(args.save_path+"/beavertails_steps/", exist_ok=True)
            #     evaluate_BeaverTails(self.model,self.tokenizer,args.save_path+"/beavertails_steps/"+f"{global_step}",max_sample=2)
            #     harmful_score=evaluate_safe_or_harmful(args.save_path+"/beavertails_steps/"+f"{global_step}")
            #     self._wandb.log({"eval/beavertails":harmful_score}, step=global_step)
        # save ckpt
        if global_step % args.save_steps == 0:
            tag = f"step{global_step}"
            os.makedirs(args.save_path+"/model_steps/"+"step"+str(global_step), exist_ok=True)
            # self.strategy.save_ckpt(self.model.model, args.save_path+"/model_steps/", tag, args.max_ckpt_num, args.max_ckpt_mem)
            self.strategy.save_model(self.model, self.tokenizer, args.save_path+"/model_steps/"+"step"+str(global_step))
            os.makedirs(args.save_path+"/p_steps/", exist_ok=True)
            if self.args.scheduler_name=="scalar":
                torch.save(self.p.logits, args.save_path+"/p_steps/"+args.scheduler_name+"_"+ args.scheduler_activation +"_step"+str(global_step)+".pt")
                torch.save(self.p(), args.save_path+"/p_steps/"+args.scheduler_name+"_"+ args.scheduler_activation +"_forward_step"+str(global_step)+".pt")
            elif self.args.scheduler_name=="neural":
                # weight_dic={}
                # logit_dict={}
                # for (ide, new_prompts_id_len, new_input, new_attention_masks, _, _) in self.fix_dataloader:
                #     new_inputs = new_input.squeeze(1).to(torch.cuda.current_device())
                #     new_attention_mask = new_attention_masks.squeeze(1).to(torch.cuda.current_device())
                #     with torch.cuda.amp.autocast():
                #         for temp, w in enumerate(self.p.inference(new_inputs,new_attention_mask,None,ide)):
                #             weight_dic[ide[temp]]=w[0]
                #             logit_dict[ide[temp]]=w[1]
                # sorted_items_weight = sorted(weight_dic.items())
                # sorted_items_logit = sorted(logit_dict.items())
                # weight_list=[v[-1] for v in sorted_items_weight]
                # weight_tensor = torch.tensor(weight_list)
                # logit_list=[v[-1] for v in sorted_items_logit]
                # logit_tensor = torch.tensor(logit_list)
                torch.save(self.p.logit_tensor, args.save_path+"/p_steps/"+args.scheduler_name+"_"+ args.scheduler_activation +"_step"+str(global_step)+".pt")
                torch.save(self.p.score_tensor, args.save_path+"/p_steps/"+args.scheduler_name+"_"+ args.scheduler_activation +"_forward_step"+str(global_step)+".pt")
                os.makedirs(args.save_path+"/p_steps/"+args.scheduler_name+"_"+ args.scheduler_activation +"_step"+str(global_step)+"/",exist_ok=True)
                self.p.save_pretrained(args.save_path+"/p_steps/"+args.scheduler_name+"_"+ args.scheduler_activation +"_step"+str(global_step)+"/")
            else:
                raise NotImplementedError
            torch.save(self.ft_idx_in_dataset, args.save_path+"/"+"ft_data_index"+".txt")
            torch.save(self.harmful_idx_in_dataset, args.save_path+"/"+"harmful_data_index"+".txt")

            
    def evaluate(self, eval_dataloader, steps=0):
        times = 0
        self.model.eval()
        with torch.no_grad():
            loss_sum = 0
            step_bar = tqdm(
                range(eval_dataloader.__len__()),
                desc="Eval stage of steps %d" % steps,
                disable=not self.strategy.is_rank_0(),
            )

            for prompts_id_len, inputs, attention_masks, _ in eval_dataloader:
                inputs = inputs.squeeze(1).to(torch.cuda.current_device())
                attention_mask = attention_masks.squeeze(1).to(torch.cuda.current_device())
                logits = self.model(inputs, attention_mask=attention_mask, return_output=True)["logits"]

                labels = torch.where(
                    attention_mask.bool(),
                    inputs,
                    self.loss_fn.IGNORE_INDEX,
                )
                if not self.pretrain_mode:
                    for label, source_len in zip(labels, prompts_id_len):
                        label[:source_len] = self.loss_fn.IGNORE_INDEX
                loss = self.batch_loss_fn(logits, labels).mean(0)

                times += 1
                loss_sum += loss.item()
                bar_dict = {"eval gpt_loss": loss_sum / times}
                step_bar.update()
                logs = self.strategy.all_reduce(bar_dict)
                step_bar.set_postfix(logs)

            if self._wandb is not None and self.strategy.is_rank_0():
                logs = {"eval/%s" % k: v for k, v in {**logs, "global_step": steps}.items()}
                self._wandb.log(logs)
        self.model.train()  # reset model state
