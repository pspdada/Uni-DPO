#!/bin/bash
# shellcheck disable=SC1090,SC1091,SC2155

# ==============================================
# Configuration Section
# ==============================================

GPU_LIST="0,1,2,3,4,5,6,7"
TRAIN_CONFIG_FILE_NAME="Llama-3-8B-Instruc-psp.yaml"
ACCELERATE_CONFIG_FILE_NAME="zero2.yaml"

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

# Script path
SCRIPT_PATH="${SCRIPT_DIR}/scripts/run_uni_dpo.py"
ACCELERATE_CONFIG_FILE="${SCRIPT_DIR}/accelerate_configs/${ACCELERATE_CONFIG_FILE_NAME}"
TRAIN_CONFIG_FILE="${SCRIPT_DIR}/configs/${TRAIN_CONFIG_FILE_NAME}"

# Get configuration name for output directory
CONFIG_NAME=$(basename "$TRAIN_CONFIG_FILE" .yaml)
OUTPUT_DIR="${SCRIPT_DIR}/output/${CONFIG_NAME}"
mkdir -p "${OUTPUT_DIR}"

# Verify required paths exist
for path in "$SCRIPT_DIR" "$SCRIPT_PATH" "$ACCELERATE_CONFIG_FILE" "$TRAIN_CONFIG_FILE"; do
    if [ ! -e "$path" ]; then
        echo "Error: Required path '$path' does not exist."
        exit 1
    fi
done

# Calculate number of GPUs
IFS=',' read -ra GPULIST <<<"$GPU_LIST"
NUM_GPUS=${#GPULIST[@]}

echo "[Info] Using ${NUM_GPUS} GPUs: ${GPU_LIST}"

# Create temporary configuration file directory
TMP_CONFIG_DIR="${OUTPUT_DIR}/tmp_configs"
mkdir -p "$TMP_CONFIG_DIR"

# Create temporary accelerate configuration file
TMP_ACCELERATE_CONFIG="${TMP_CONFIG_DIR}/$(basename "$ACCELERATE_CONFIG_FILE")"
cp "$ACCELERATE_CONFIG_FILE" "$TMP_ACCELERATE_CONFIG"
sed -i "s|^\(num_processes:\).*|\1 $NUM_GPUS|" "$TMP_ACCELERATE_CONFIG"

echo "[Info] Temporary configs saved to: ${TMP_CONFIG_DIR}"

# Create log directory
LOG_FILE="${OUTPUT_DIR}/logs/train.log"
mkdir -p "$(dirname "$LOG_FILE")"
: >"$LOG_FILE"

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

# Launch training using temporary configuration files
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=${GPU_LIST} ACCELERATE_LOG_LEVEL=info \
    accelerate launch --main_process_port=25405 --config_file "${TMP_ACCELERATE_CONFIG}" \
    "${SCRIPT_PATH}" "${TRAIN_CONFIG_FILE}" \
    2>&1 | tee -a "${LOG_FILE}"
