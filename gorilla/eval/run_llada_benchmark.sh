#!/bin/bash
# LLaDA Benchmark Script for Gorilla API Evaluation
#
# This script runs the LLaDA model on the Gorilla API benchmark and evaluates the results.
#
# Usage:
#   ./run_llada_benchmark.sh [api_name] [steps] [gen_length]
#
# Arguments:
#   api_name: torchhub, huggingface, or tensorhub (default: torchhub)
#   steps: Number of diffusion steps (default: 128)
#   gen_length: Maximum generation length (default: 256)

set -e

# Default parameters
API_NAME=${1:-torchhub}
STEPS=${2:-128}
GEN_LENGTH=${3:-256}
BLOCK_LENGTH=32
TEMPERATURE=0.0

# Paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="${SCRIPT_DIR}/../../data"
EVAL_DATA_DIR="${SCRIPT_DIR}/eval-data"

# Set paths based on API name
if [ "$API_NAME" == "torchhub" ]; then
    QUESTION_FILE="${EVAL_DATA_DIR}/questions/torchhub/questions_torchhub_0_shot.jsonl"
    API_DATASET="${DATA_DIR}/api/torchhub_api.jsonl"
    APIBENCH="${DATA_DIR}/apibench/torchhub_eval.json"
    EVAL_SCRIPT="${SCRIPT_DIR}/eval-scripts/ast_eval_th.py"
elif [ "$API_NAME" == "huggingface" ]; then
    QUESTION_FILE="${EVAL_DATA_DIR}/questions/huggingface/questions_huggingface_0_shot.jsonl"
    API_DATASET="${DATA_DIR}/api/huggingface_api.jsonl"
    APIBENCH="${DATA_DIR}/apibench/huggingface_eval.json"
    EVAL_SCRIPT="${SCRIPT_DIR}/eval-scripts/ast_eval_hf.py"
elif [ "$API_NAME" == "tensorhub" ]; then
    QUESTION_FILE="${EVAL_DATA_DIR}/questions/tensorflowhub/questions_tensorflowhub_0_shot.jsonl"
    API_DATASET="${DATA_DIR}/api/tensorflowhub_api.jsonl"
    APIBENCH="${DATA_DIR}/apibench/tensorflow_eval.json"
    EVAL_SCRIPT="${SCRIPT_DIR}/eval-scripts/ast_eval_tf.py"
else
    echo "Error: Unknown API name: $API_NAME"
    echo "Supported: torchhub, huggingface, tensorhub"
    exit 1
fi

# Output file
OUTPUT_DIR="${EVAL_DATA_DIR}/responses/${API_NAME}"
mkdir -p "$OUTPUT_DIR"
OUTPUT_FILE="${OUTPUT_DIR}/response_${API_NAME}_llada_0_shot.jsonl"

echo "======================================"
echo "LLaDA Benchmark for Gorilla"
echo "======================================"
echo "API Name: $API_NAME"
echo "Steps: $STEPS"
echo "Gen Length: $GEN_LENGTH"
echo "Block Length: $BLOCK_LENGTH"
echo "Temperature: $TEMPERATURE"
echo ""
echo "Question File: $QUESTION_FILE"
echo "Output File: $OUTPUT_FILE"
echo "======================================"

# Step 1: Generate responses using LLaDA
echo ""
echo "[Step 1/2] Generating responses with LLaDA..."
echo ""

python "${SCRIPT_DIR}/get_llm_responses_llada.py" \
    --output_file "$OUTPUT_FILE" \
    --question_data "$QUESTION_FILE" \
    --api_name "$API_NAME" \
    --steps "$STEPS" \
    --gen_length "$GEN_LENGTH" \
    --block_length "$BLOCK_LENGTH" \
    --temperature "$TEMPERATURE"

echo ""
echo "[Step 1/2] Response generation complete!"
echo ""

# Step 2: Evaluate the responses
echo "[Step 2/2] Evaluating responses..."
echo ""

cd "${SCRIPT_DIR}/eval-scripts"
python "$EVAL_SCRIPT" \
    --api_dataset "$API_DATASET" \
    --apibench "$APIBENCH" \
    --llm_responses "$OUTPUT_FILE"

echo ""
echo "======================================"
echo "Benchmark complete!"
echo "======================================"
