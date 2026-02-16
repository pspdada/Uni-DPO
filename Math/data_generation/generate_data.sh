#!/bin/bash
# shellcheck disable=SC1090,SC2155,SC1091

# ==============================================================================
# Configuration Section
# ==============================================================================

# Conda configuration
readonly GPU_LIST="0,1,2,3"
readonly CONDA_BASE_PATH="your/path/to/miniconda3"
readonly CONDA_ENV_NAME="Uni-DPO-Math-eval"

# Basic settings
POLICY_MODEL_PATH="Qwen/Qwen2.5-Math-7B"
PRM_MODEL="Qwen/Qwen2.5-Math-PRM-7B" # Process reward model path
DATASET_NAME="RLHFlow/numia_prompt_dpo1"
OUTPUT_PATH="output"

# Dataset construction parameters
SAMPLE_NUM=8            # Number of responses to generate per question during sampling
SAMPLE_TEMP=1.0         # Temperature for sampling (float)
SCORE_MARGIN=0.0        # Minimum score difference between positive and negative samples (score ranges from 0 to 10, float)
REWARD_TYPE="min"       # Reward type, options: min, avg
POS_STRATEGY="max"      # Positive sample selection strategy, options: max, rand
NEG_STRATEGY="positive" # Negative sample selection strategy, options: min, rand, positive
TRUNCATE_LENGTH=4000    # Maximum model output length (number of token ids) considered when building training dataset; outputs exceeding this length are considered garbage and removed; set to -1 to keep all outputs

# ==============================================================================
# DO NOT MODIFY BELOW THIS LINE
# ==============================================================================

# Configure global variables
export SCRIPT_DIR="$(dirname "$(realpath "$0")")"
echo "Script directory: $SCRIPT_DIR"

# Increase file descriptor limit to avoid "Too many open files" error
ulimit -n 65536

# Helper function: Normalize path
normalize_output_path() {
    local output_path="$1"
    if [[ "$output_path" == /* ]]; then
        # If it's an absolute path, return directly
        echo "$output_path"
    else
        # If it's a relative path, concatenate with SCRIPT_DIR
        echo "${SCRIPT_DIR}/${output_path}"
    fi
}

# Normalize OUTPUT_PATH
OUTPUT_PATH=$(normalize_output_path "$OUTPUT_PATH")

# Convert GPU_LIST to standardized comma-separated format
normalize_gpu_list() {
    local input="$1"
    local result=()

    # Split input by comma
    IFS=',' read -ra parts <<<"$input"
    for part in "${parts[@]}"; do
        if [[ "$part" =~ ^([0-9]+)-([0-9]+)$ ]]; then
            # If it's a range (e.g., "0-3"), expand to specific GPU ID list
            start=${BASH_REMATCH[1]}
            end=${BASH_REMATCH[2]}
            if ((start > end)); then
                echo "Error: Invalid range '$part'. Start must be <= end." >&2
                exit 1
            fi
            for ((i = start; i <= end; i++)); do
                result+=("$i")
            done
        elif [[ "$part" =~ ^[0-9]+$ ]]; then
            # If it's a single number (e.g., "0"), add directly to result
            result+=("$part")
        else
            echo "Error: Invalid GPU specification '$part'." >&2
            exit 1
        fi
    done

    # Remove duplicates and sort
    printf "%s\n" "${result[@]}" | sort -n | uniq | paste -sd, -
}

# Standardize GPU list
GPU_LIST=$(normalize_gpu_list "$GPU_LIST")

# Calculate number of GPUs
IFS=',' read -ra GPULIST <<<"$GPU_LIST"
NUM_GPUS=${#GPULIST[@]}

echo "[Info] Using $NUM_GPUS GPUs: $GPU_LIST"

#-----------------------------------------------------------------------------------------------------------------------

# Create necessary directories
mkdir -p "$OUTPUT_PATH/logs"
mkdir -p "$OUTPUT_PATH/data"

# Define output file path
json_output="${OUTPUT_PATH}/data/$(basename "$POLICY_MODEL_PATH")/$(basename "$DATASET_NAME")"

echo "[Info] Using model path: $POLICY_MODEL_PATH"
echo "[Info] Using dataset name: $DATASET_NAME"
echo "[Info] Output will be saved to: $json_output"

source "${CONDA_BASE_PATH}/bin/activate" || {
    echo "Error: Failed to source conda at ${CONDA_BASE_PATH}/bin/activate"
    exit 1
}
conda activate "$CONDA_ENV_NAME" || {
    echo "Error: Conda environment '$CONDA_ENV_NAME' not found or failed to activate"
    exit 1
}

#-----------------------------------------------------------------------------------------------------------------------

echo "[Info] Starting sample generation"
declare -a pids

for IDX in $(seq 0 $((NUM_GPUS - 1))); do
    log_file="${OUTPUT_PATH}/logs/Gen_samples_GPU_${IDX}.log"

    # Print environment variables and settings to log file
    {
        echo "Environment variables and settings for GPU $IDX:"
        declare -p SCRIPT_DIR POLICY_MODEL_PATH DATASET_NAME json_output SAMPLE_NUM SAMPLE_TEMP IDX NUM_GPUS TRUNCATE_LENGTH 2>/dev/null
        echo "----------------------------------------"
    } >>"$log_file"

    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python "${SCRIPT_DIR}/scripts/gen_samples.py" \
        --model_name_or_path "$POLICY_MODEL_PATH" \
        --dataset_name_or_path "$DATASET_NAME" \
        --output_dir "$json_output" \
        --K "$SAMPLE_NUM" \
        --temperature "$SAMPLE_TEMP" \
        --local_index "$IDX" \
        --my_world_size "$NUM_GPUS" \
        --truncate_length "$TRUNCATE_LENGTH" \
        2>&1 | tee -a "$log_file" &

    pids[IDX]=$!
    echo "Process $IDX (PID: ${pids[$IDX]}) started" >>"$log_file"

    sleep 3 # Wait a bit before starting the next process
done

# Wait for all processes and check exit status
for IDX in $(seq 0 $((NUM_GPUS - 1))); do
    wait "${pids[$IDX]}" || echo "Process $IDX (PID: ${pids[$IDX]}) failed with exit code: $?"
done

# Merge generated data
echo "[Info] Merging generated data"
python "${SCRIPT_DIR}/scripts/merge_files.py" \
    --base_path "${json_output}" --output_file "${json_output}_data.jsonl" --num_datasets "$NUM_GPUS"

# Perform verifiable reward labeling on generated results
echo "[Info] Starting verifiable reward labeling"
python "${SCRIPT_DIR}/scripts/verifiable_reward_labeling.py" \
    --dataset_name_or_path "${json_output}_data.jsonl" --output_file "${json_output}_data_v.jsonl"

echo "[Info] Starting process reward labeling"
PRM_input_file="${json_output}_data_v.jsonl"
PRM_output_name="${json_output}_prm"

declare -a pids
for IDX in $(seq 0 $((NUM_GPUS - 1))); do
    log_file="${OUTPUT_PATH}/logs/PRM_labeling_GPU_${IDX}.log"

    # Print environment variables and settings to log file
    {
        echo "Environment variables and settings for GPU $IDX:"
        declare -p SCRIPT_DIR PRM_MODEL PRM_input_file PRM_output_name IDX NUM_GPUS 2>/dev/null
        echo "----------------------------------------"
    } >>"$log_file"

    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python "${SCRIPT_DIR}/scripts/progress_reward_labeling.py" \
        --PRM_model_name_or_path "$PRM_MODEL" \
        --input_file "$PRM_input_file" \
        --output_name "$PRM_output_name" \
        --local_index "$IDX" \
        --my_world_size "$NUM_GPUS" \
        2>&1 | tee -a "$log_file" &

    pids[IDX]=$!
    echo "Process $IDX (PID: ${pids[$IDX]}) started" >>"$log_file"

    sleep 3 # Wait a bit before starting the next process
done

# Wait for all processes and check exit status
for IDX in $(seq 0 $((NUM_GPUS - 1))); do
    wait "${pids[$IDX]}" || echo "Process $IDX (PID: ${pids[$IDX]}) failed with exit code: $?"
done

echo "[Info] Merging PRM labeled data"
python "${SCRIPT_DIR}/scripts/merge_files.py" \
    --base_path "$PRM_output_name" --output_file "${json_output}_prm_data.jsonl" --num_datasets "$NUM_GPUS"

echo "[Info] Generating uni DPO training data"
python "${SCRIPT_DIR}/scripts/get_uni_dpo_data.py" \
    --input_file "${json_output}_prm_data.jsonl" --output_file "${json_output}_training_data.jsonl" \
    --score_margin "$SCORE_MARGIN" --reward_type "$REWARD_TYPE" \
    --pos_strategy "$POS_STRATEGY" --neg_strategy "$NEG_STRATEGY"

echo "[Info] Uni DPO training data generated at: ${json_output}_training_data.jsonl"

echo "[Info] Data generation pipeline completed successfully!"
