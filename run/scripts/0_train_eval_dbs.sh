set -e
source ~/miniconda3/bin/activate bds

# Load configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "$SCRIPT_DIR/config.sh" ]; then
    source "$SCRIPT_DIR/config.sh"
else
    echo "Error: config.sh not found in $SCRIPT_DIR!"
    exit 1
fi

export PYTHONPATH=$PREFIX_DIR
cd $PREFIX_DIR/run

max_samples=1000
max_new_samples=1000
upperlevel_weight=-1 # no weight decay
GPU_ID=0
new_dataset=sst2
scheduler_name=scalar # or neural
scheduler_activation=softmax
entropy=0
model_path=meta-llama/Llama-2-7b-hf
if [ "$new_dataset" = "alpaca" ]; then
    MAX_EPOCHS=100
else
    MAX_EPOCHS=20
fi
upperlevel_weight_decay=$(echo "scale=2; 1.0 / $MAX_EPOCHS" | bc)
if (( max_samples > max_new_samples )); then
  step=$(( max_samples / 10 ))
else
  step=$(( max_new_samples / 10 ))
fi

for poison_ratio in $(seq 0.1 0.1 0.1); do
    safe_prob=$(echo "scale=1; 1.0 - $poison_ratio" | bc)
    new_dataset_probs="$safe_prob,$poison_ratio"
    if [[ "$model_path" == *"Llama-2"* ]]; then
      model_name="llama2"
    else
      model_name="unknown"
    fi

    WORK_DIR=$PREFIX_DIR/run/scripts/ckpt/${model_name}_${scheduler_name}_${scheduler_activation}_entropy${entropy}_${new_dataset}_p${poison_ratio}_${max_samples}_${max_new_samples}_noprior
    if [ "$upperlevel_weight" = "-2" ]; then
        WORK_DIR="${WORK_DIR}_noAlignment"
    fi
    if [ "$upperlevel_weight" = "-1" ]; then
        WORK_DIR="${WORK_DIR}_noWeightDecay"
    fi
    if [ "$upperlevel_weight" = "-3" ]; then
        WORK_DIR="${WORK_DIR}_SFT"
    fi
    if [ "$upperlevel_weight" = "-4" ]; then
        WORK_DIR="${WORK_DIR}_noweight"
    fi
    LOG_DIR=$WORK_DIR/log
    LOG_FILE="${LOG_DIR}/log.log"
    mkdir -p "$LOG_DIR"
    exec > "$LOG_FILE" 2>&1

    module load cuda12.5/toolkit/12.5.1
    module load gcc/14.2.0

    MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('',0)); print(s.getsockname()[1]); s.close()")

    CUDA_VISIBLE_DEVICES=$GPU_ID deepspeed train_dbs.py $noweight_flag\
        --seed 42 \
        --max_len 2048 \
        --dataset $PREFIX_DIR/run/scripts/datasets/beavertails_safe_train.json \
        --dataset_probs 1. \
        --new_dataset $PREFIX_DIR/run/scripts/datasets/${new_dataset}.json,$PREFIX_DIR/run/scripts/datasets/beavertails_harmful_train.json \
        --new_dataset_probs $new_dataset_probs \
        --upperlevel_weight $upperlevel_weight \
        --upperlevel_weight_decay $upperlevel_weight_decay \
        --train_batch_size 10 \
        --micro_train_batch_size 10 \
        --max_samples $max_samples \
        --max_new_samples $max_new_samples \
        --pretrain $model_path \
        --ref_constant 0. \
        --scheduler_activation $scheduler_activation \
        --scheduler_name $scheduler_name \
        --save_steps -1 \
        --logging_steps 1 \
        --eval_steps -1 \
        --zero_stage 0 \
        --max_epochs $MAX_EPOCHS \
        --bf16 \
        --learning_rate 1e-5 \
        --scheduler_learning_rate 5e-3 \
        --scheduler_lr_scheduler constant \
        --lr_scheduler constant \
        --gradient_checkpointing \
        --lora_rank 32 \
        --lora_alpha 4 \
        --l2 0.1 \
        --warmup 0.1 \
        --entropy $entropy \
        --target_modules q_proj v_proj k_proj \
        --use_wandb $WANDB_API_KEY \
        --master_port $MASTER_PORT \
        --scorer_model_path $PREFIX_DIR/faireq/125M \
        --save_path $WORK_DIR

    # split dataset
    CUDA_VISIBLE_DEVICES=$GPU_ID python $PREFIX_DIR/analysis/score_analyzer.py \
         --path $WORK_DIR

    # Mountain Range Map
    CUDA_VISIBLE_DEVICES=$GPU_ID python $PREFIX_DIR/analysis/mountain_range_plotter.py \
         --path $WORK_DIR \
         --step $step \
         --flag all
    CUDA_VISIBLE_DEVICES=$GPU_ID python $PREFIX_DIR/analysis/mountain_range_plotter.py \
         --path $WORK_DIR \
         --step $step \
         --flag ft
    CUDA_VISIBLE_DEVICES=$GPU_ID python $PREFIX_DIR/analysis/mountain_range_plotter.py \
         --path $WORK_DIR \
         --step $step \
         --flag harmful

    echo "Waiting for training to complete..."
    wait
    echo "Reading wandb run ID from file..."
    WANDB_ID_FILE=$WORK_DIR/wandb_run_id.txt
    if [ -f "$WANDB_ID_FILE" ]; then
        export WANDB_RUN_ID=$(cat "$WANDB_ID_FILE")
        echo "Found wandb run ID: $WANDB_RUN_ID"
    else
        echo "No wandb run ID file found"
        export WANDB_RUN_ID=""
    fi

    export WANDB_PROJECT=$WANDB_PROJECT
    export WANDB_RUN_NAME="poison_ratio_${poison_ratio}"

    echo "Starting evaluation scripts for poison_ratio: $poison_ratio"

    CUDA_VISIBLE_DEVICES=$GPU_ID python $PREFIX_DIR/run/${new_dataset}/pred_eval.py \
        --lora_path $WORK_DIR \
        --model_path $model_path \
        --use_wandb $WANDB_API_KEY
    CUDA_VISIBLE_DEVICES=$GPU_ID python $PREFIX_DIR/run/poison/evaluation/pred.py \
        --lora_path $WORK_DIR \
        --model_path $model_path
    CUDA_VISIBLE_DEVICES=$GPU_ID python $PREFIX_DIR/run/poison/evaluation/eval_sentiment.py \
        --lora_path $WORK_DIR \
        --use_wandb $WANDB_API_KEY

    # #For alpaca_eval
    # export OPENAI_API_KEY=placeholder
    # CUDA_VISIBLE_DEVICES=$GPU_ID alpaca_eval \
    #     --model_outputs $WORK_DIR/alpaca_steps/all.json \
    #     --annotators_config 'alpaca_eval_gpt4_turbo_fn'

done