#!/bin/bash
# shellcheck disable=SC1090,SC1091

# Model and Dataset Configuration
MODEL_NAME="<Your path to Qwen2-VL-2B model>"   #! TODO
OUTPUT_DIR="<Your output directory path>"       #! TODO
DATASET="uni_dpo_image_only_mcq_short_long_50k" # len: 50555

# Environment Configuration
CONDA_BASE_PATH="/your/path/to/miniconda3" #! TODO
CONDA_ENV_NAME="Uni-DPO-Multimodal-train"
GPU_LIST="0,1,2,3"
PER_GPU_BATCH_SIZE="1"

# DPO Hyperparameters
PERF_LOSS=sigmoid
LR=5.0e-7
LS=0.1
PERF_BETA=0.1

#! The following does not need to be modified

# Training Configuration
GLOBAL_BATCH_SIZE=128
EPOCH=1
PREPROCESSING_NUM_WORKERS=16

IFS=',' read -ra GPULIST <<<"$GPU_LIST"
NUM_GPUS=${#GPULIST[@]}
GRADIENT_ACCUMULATION_STEPS=$((GLOBAL_BATCH_SIZE / (NUM_GPUS * PER_GPU_BATCH_SIZE)))
if ((GRADIENT_ACCUMULATION_STEPS <= 0)); then
    echo "Error: Invalid configuration for GLOBAL_BATCH_SIZE, NUM_GPUS, or PER_GPU_BATCH_SIZE."
    exit 1
fi

echo "[Info] Global batch size: $GLOBAL_BATCH_SIZE"
echo "[Info] Gradient accumulation steps: $GRADIENT_ACCUMULATION_STEPS"

# 初始化 Conda 环境
source "${CONDA_BASE_PATH}/bin/activate" || {
    echo "Error: Failed to source conda at ${CONDA_BASE_PATH}/bin/activate"
    exit 1
}
conda activate "$CONDA_ENV_NAME" || {
    echo "Error: Conda environment '$CONDA_ENV_NAME' not found or failed to activate"
    exit 1
}

OUTPUT_DIR="$(realpath "$OUTPUT_DIR")"
echo "[Info] Output directory set to ${OUTPUT_DIR}"

LOG_PATH="${OUTPUT_DIR}/log/train.log"
mkdir -p "$(dirname "$LOG_PATH")"
echo "[Info] Logging to $LOG_PATH"

{
    echo "[Info] Starting training at $(date)"
    echo "[Info] Current script content:"
    cat "$0"
    echo "------------------------------------------"
} >>"$LOG_PATH" 2>&1

DISABLE_VERSION_CHECK=1 deepspeed --include localhost:"$GPU_LIST" --master_port 61000 ./src/train.py \
    --deepspeed examples/deepspeed/ds_z2_config.json \
    --model_name_or_path "$MODEL_NAME" \
    --stage dpo \
    --flash_attn fa2 \
    --do_train \
    --finetuning_type full \
    --pref_beta "$PERF_BETA" \
    --pref_loss "$PERF_LOSS" \
    --dpo_average_log_prob \
    --dpo_label_smoothing "$LS" \
    --dataset "$DATASET" \
    --template "qwen2_vl" \
    --output_dir "$OUTPUT_DIR" \
    --overwrite_cache \
    --per_device_train_batch_size "$PER_GPU_BATCH_SIZE" \
    --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS" \
    --preprocessing_num_workers "$PREPROCESSING_NUM_WORKERS" \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --save_steps 1000 \
    --learning_rate "$LR" \
    --num_train_epochs "$EPOCH" \
    --plot_loss \
    --bf16 \
    --warmup_ratio 0 \
    --ddp_timeout 180000000 \
    --cutoff_len 4096 \
    --max_samples 10000000 \
    --overwrite_output_dir \
    2>&1 | tee -a "$LOG_PATH"
