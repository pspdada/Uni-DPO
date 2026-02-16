#!/bin/bash
# shellcheck disable=SC2155,SC1091

# ==============================================================================
# Configuration Section
# ==============================================================================
# Define the following information:
# - GPU IDs to use
# - Model paths
# - Output directory names for evaluation results
# These three lists must have the same length and correspond in order.

readonly GPU_LIST=(0 1 2) # List of GPU IDs

readonly MODELS_NAME_OR_PATH=(
    "Qwen/Qwen2.5-Math-1.5B"
    "Qwen/Qwen2.5-Math-7B"
    "Qwen2.5-Math-7B-Uni-DPO"
)

readonly OUTPUT_DIRS=(
    "Qwen2.5-Math-1.5B"
    "Qwen2.5-Math-7B"
    "Qwen2.5-Math-7B-Uni-DPO"
)

# Define the datasets to evaluate, separated by commas
readonly DATASET_TO_EVAL="math500,minerva_math,olympiadbench,aime24,amc23,gaokao2023en,college_math,gsm8k"

# Conda configuration
readonly CONDA_BASE_PATH="your/path/to/conda"
readonly CONDA_ENV_NAME="Uni-DPO-Math-eval"

# ==============================================================================
# DO NOT MODIFY BELOW THIS LINE
# ==============================================================================

# Get the directory of the current script
export SCRIPT_DIR="$(dirname "$(realpath "$0")")"
echo "Script directory: $SCRIPT_DIR"

RESULT_DIR="${SCRIPT_DIR}/results"
DATA_DIR="${SCRIPT_DIR}/data"

# Ensure the lengths of the lists match
if [ ${#GPU_LIST[@]} -ne ${#MODELS_NAME_OR_PATH[@]} ] || [ ${#GPU_LIST[@]} -ne ${#OUTPUT_DIRS[@]} ]; then
    echo "Error: The number of GPU IDs, model names, and output names must be the same."
    exit 1
fi

# Check if all model directories exist before starting tasks
for model_name in "${MODELS_NAME_OR_PATH[@]}"; do
    if [ ! -d "$model_name" ]; then
        echo "Error: Model directory does not exist: $model_name"
        exit 1
    fi
done

source "${CONDA_BASE_PATH}/bin/activate" || {
    echo "Error: Failed to source conda at ${CONDA_BASE_PATH}/bin/activate"
    exit 1
}
conda activate "$CONDA_ENV_NAME" || {
    echo "Error: Conda environment '$CONDA_ENV_NAME' not found or failed to activate"
    exit 1
}

# Increase file descriptor limit to avoid "Too many open files" error
ulimit -n 65536

# Function to run a single task
run_task() {
    local gpu_id=$1
    local model_name=$2
    local output_name=$3

    # Construct the full output directory path
    output_dir="$RESULT_DIR/$output_name"
    mkdir -p "$output_dir"

    # Export environment variables
    export GPU_ID=$gpu_id MODEL_NAME=$model_name OUTPUT_DIR=$output_dir SCRIPT_DIR="$SCRIPT_DIR" DATA_DIR="$DATA_DIR"
    export DATA_NAME="$DATASET_TO_EVAL"

    log_file="$output_dir/evaluation.log"

    echo "Running task on GPU $gpu_id with log file $log_file"
    echo "Model name: $model_name" >"$log_file"
    echo "Output directory: $output_dir" >>"$log_file"

    # Run the evaluation script in the background
    bash "${SCRIPT_DIR}/eval.sh" >>"$log_file" 2>&1 &

    echo "PID: $!" >>"$log_file"
}

# Launch tasks in parallel
for i in "${!GPU_LIST[@]}"; do
    run_task "${GPU_LIST[$i]}" "${MODELS_NAME_OR_PATH[$i]}" "${OUTPUT_DIRS[$i]}"
done

# Wait for all background tasks to finish
wait

echo "All tasks completed."
