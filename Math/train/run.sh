#!/bin/bash
# shellcheck disable=SC1090,SC1091,SC2155

# ==============================================
# Configuration Section
# ==============================================

# GPU list to use, comma-separated
GPU_LIST="0,1,2,3,4,5,6,7"
TRAIN_CONFIG_FILE_NAME="Qwen2.5-Math-7B-Uni-DPO.yaml"
ACCELERATE_CONFIG_FILE_NAME="zero2.yaml"

MODEL_PATH="your/path/to/Qwen2.5-Math-7B"
TRAIN_DATA_NAME="Train_Qwen_2_5_math_7B.jsonl"

# Conda configuration
CONDA_BASE_PATH="your/path/to/miniconda3"
CONDA_ENV_NAME="Uni-DPO-alignment"

# ==============================================
# DO NOT MODIFY BELOW THIS LINE
# ==============================================

source "${CONDA_BASE_PATH}/bin/activate" || {
    echo "Error: Failed to source conda at ${CONDA_BASE_PATH}/bin/activate"
    exit 1
}
conda activate "$CONDA_ENV_NAME" || {
    echo "Error: Conda environment '$CONDA_ENV_NAME' not found or failed to activate"
    exit 1
}

# Get the directory of the current script
export SCRIPT_DIR="$(dirname "$(realpath "$0")")"
echo "Script directory: $SCRIPT_DIR"

TRAIN_DATA_PATH="${SCRIPT_DIR}/data/${TRAIN_DATA_NAME}"
SCRIPT_PATH="${SCRIPT_DIR}/scripts/run_uni_dpo.py"
ACCELERATE_CONFIG_FILE="${SCRIPT_DIR}/accelerate_configs/${ACCELERATE_CONFIG_FILE_NAME}"

# DPO configuration file (YAML file)
TRAIN_CONFIG_FILE="${SCRIPT_DIR}/configs/${TRAIN_CONFIG_FILE_NAME}"
CONFIG_NAME=$(basename "$TRAIN_CONFIG_FILE" .yaml)

# Output directory for training model, different experiments need different directories
OUTPUT_DIR="${SCRIPT_DIR}/output/${CONFIG_NAME}"
mkdir -p "$OUTPUT_DIR"

for path in "${MODEL_PATH}" "${TRAIN_DATA_PATH}" "${SCRIPT_PATH}" "${ACCELERATE_CONFIG_FILE}" "${TRAIN_CONFIG_FILE}"; do
    if [ ! -e "$path" ]; then
        echo "Error: Required path '$path' does not exist."
        exit 1
    fi
done

# Calculate number of GPUs
IFS=',' read -ra GPULIST <<<"$GPU_LIST"
NUM_GPUS=${#GPULIST[@]}
GLOBAL_BATCH_SIZE=128
PER_GPU_BATCH_SIZE=1

# Dynamically calculate gradient_accumulation_steps for training
GRADIENT_ACCUMULATION_STEPS=$((GLOBAL_BATCH_SIZE / (NUM_GPUS * PER_GPU_BATCH_SIZE)))
if ((GRADIENT_ACCUMULATION_STEPS <= 0)); then
    echo "Error: Invalid configuration for GLOBAL_BATCH_SIZE, NUM_GPUS, or PER_GPU_BATCH_SIZE."
    exit 1
fi

# Create temporary configuration file directory
TMP_CONFIG_DIR="${OUTPUT_DIR}/tmp_configs"
mkdir -p "$TMP_CONFIG_DIR"

# Create temporary accelerate configuration file
TMP_ACCELERATE_CONFIG="${TMP_CONFIG_DIR}/$(basename "$ACCELERATE_CONFIG_FILE")"
cp "$ACCELERATE_CONFIG_FILE" "$TMP_ACCELERATE_CONFIG"
sed -i "s|^\(num_processes:\).*|\1 $NUM_GPUS|" "$TMP_ACCELERATE_CONFIG"

# Create temporary training configuration file
TMP_TRAIN_CONFIG="${TMP_CONFIG_DIR}/$(basename "$TRAIN_CONFIG_FILE")"
cp "$TRAIN_CONFIG_FILE" "$TMP_TRAIN_CONFIG"
sed -i \
    -e "s|^output_dir:.*|output_dir: ${OUTPUT_DIR}|g" \
    -e "s|^model_name_or_path:.*|model_name_or_path: ${MODEL_PATH}|g" \
    -e "s|^ref_model:.*|ref_model: ${MODEL_PATH}|g" \
    -e "s|^train_data_path:.*|train_data_path: ${TRAIN_DATA_PATH}|g" \
    -e "s|^gradient_accumulation_steps:.*|gradient_accumulation_steps: ${GRADIENT_ACCUMULATION_STEPS}|g" \
    "$TMP_TRAIN_CONFIG"

LOG_FILE="${OUTPUT_DIR}/logs/train.log"
mkdir -p "$(dirname "$LOG_FILE")"
: >"$LOG_FILE"

echo "[Info] Global batch size: ${GLOBAL_BATCH_SIZE}"
echo "[Info] Gradient accumulation steps: ${GRADIENT_ACCUMULATION_STEPS}"
echo "[Info] Using ${NUM_GPUS} GPUs: ${GPU_LIST}"
echo "[Info] Temporary configs saved to: ${TMP_CONFIG_DIR}"
echo "[Info] Training log will be saved to ${LOG_FILE}"

{
    echo "----------------------------------------"
    echo "[[ACCELERATE_CONFIG_FILE]]"
    echo ""
    cat "${TMP_ACCELERATE_CONFIG}"
    echo "----------------------------------------"
    echo "[[TRAIN_CONFIG_FILE]]"
    echo ""
    cat "${TRAIN_CONFIG_FILE}"
    echo "----------------------------------------"
} >>"${LOG_FILE}"

# Launch training
OMP_NUM_THREADS=1 DS_SKIP_CUDA_CHECK=1 CUDA_VISIBLE_DEVICES="${GPU_LIST}" \
    accelerate launch --main_process_port=29500 --config_file "${TMP_ACCELERATE_CONFIG}" \
    "${SCRIPT_PATH}" "${TMP_TRAIN_CONFIG}" \
    2>&1 | tee -a "${LOG_FILE}"
