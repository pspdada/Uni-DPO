#!/bin/bash
# shellcheck disable=SC1090

set -ex

# Read parameters from environment variables
export CUDA_VISIBLE_DEVICES=$GPU_ID
export DATA_DIR=${DATA_DIR:-"./data"}
export DATA_NAME=${DATA_NAME:-"math500,minerva_math,olympiadbench,aime24,amc23"}
export PROMPT_TYPE=${PROMPT_TYPE:-"qwen25-math-cot"}
export SPLIT=${SPLIT:-"test"}
export NUM_TEST_SAMPLE=${NUM_TEST_SAMPLE:-"-1"}
export SCRIPT_DIR=${SCRIPT_DIR:-$(dirname "$(realpath "$0")")}

TOKENIZERS_PARALLELISM=false \
    python -u "${SCRIPT_DIR}/scripts/math_eval.py" \
    --model_name_or_path "${MODEL_NAME}" \
    --data_name "${DATA_NAME}" \
    --data_dir "${DATA_DIR}" \
    --output_dir "${OUTPUT_DIR}" \
    --split "${SPLIT}" \
    --prompt_type "${PROMPT_TYPE}" \
    --num_test_sample "${NUM_TEST_SAMPLE}" \
    --max_tokens_per_call 3000 \
    --seed 0 \
    --temperature 0 \
    --n_sampling 1 \
    --top_p 1 \
    --start 0 \
    --end -1 \
    --use_vllm \
    --save_outputs
