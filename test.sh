#!/bin/bash

# SAM3 Anomaly Detection Testing Script
# By default, tests all categories unless specific category is provided
# Examples:
# - ./test.sh                     # Test all categories
# - ./test.sh --category bottle   # Test specific category
# - ./test.sh --visualize         # Enable visualization

# Default parameters
DATASET="mvtec"
DATASET_PATH="~/autodl-tmp/datasets"
MODEL_PATH="~/autodl-tmp/download/sam3"
OUTPUT_DIR="~/autodl-tmp/test_results"
DEVICE="cuda"
BATCH_SIZE=1
CONFIDENCE_THRESHOLD=0.5
IOU_THRESHOLD=0.5
PROMPT_STRATEGY="mixed"
NUM_PROMPTS=8
CATEGORY=""
SEED=122
VISUALIZE=false
CONFIG="configs/config.yaml"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dataset) DATASET="$2"; shift 2 ;;
        --dataset_path) DATASET_PATH="$2"; shift 2 ;;
        --model_path) MODEL_PATH="$2"; shift 2 ;;
        --output_dir) OUTPUT_DIR="$2"; shift 2 ;;
        --device) DEVICE="$2"; shift 2 ;;
        --batch_size) BATCH_SIZE=$2; shift 2 ;;
        --confidence_threshold) CONFIDENCE_THRESHOLD=$2; shift 2 ;;
        --iou_threshold) IOU_THRESHOLD=$2; shift 2 ;;
        --prompt_strategy) PROMPT_STRATEGY="$2"; shift 2 ;;
        --num_prompts) NUM_PROMPTS=$2; shift 2 ;;
        --category) CATEGORY="$2"; shift 2 ;;
        --seed) SEED=$2; shift 2 ;;
        --visualize) VISUALIZE=true; shift 1 ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
done

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Print configuration
echo "=== Test Configuration ==="
echo "Dataset: $DATASET | Path: $DATASET_PATH"
echo "Model Path: $MODEL_PATH"
echo "Output: $OUTPUT_DIR | Device: $DEVICE | Batch Size: $BATCH_SIZE"
echo "Thresholds: Conf $CONFIDENCE_THRESHOLD | IOU $IOU_THRESHOLD"
echo "Prompt: $PROMPT_STRATEGY ($NUM_PROMPTS) | Seed: $SEED"
echo "Category: ${CATEGORY:-All categories}"
echo "Visualization: $VISUALIZE | Config: ${CONFIG:-None}"
echo "======================="

# Build command
CMD="python main.py --mode test --dataset \"$DATASET\" --dataset_path \"$DATASET_PATH\" --model_path \"$MODEL_PATH\" --output_dir \"$OUTPUT_DIR\" --device \"$DEVICE\" --batch_size $BATCH_SIZE --confidence_threshold $CONFIDENCE_THRESHOLD --iou_threshold $IOU_THRESHOLD --prompt_strategy \"$PROMPT_STRATEGY\" --num_prompts $NUM_PROMPTS --seed $SEED $CONFIG"

# Add conditional arguments
[[ -n "$CATEGORY" ]] && CMD="$CMD --category \"$CATEGORY\""
[[ "$VISUALIZE" = true ]] && CMD="$CMD --visualize"

# Execute command
echo "Starting test process..."
eval "$CMD"

echo "\n=== Testing completed ==="
echo "Results saved to: $OUTPUT_DIR"
echo "Metrics location: ${CATEGORY:+"./$OUTPUT_DIR/$CATEGORY" : "$OUTPUT_DIR/category_directories + $OUTPUT_DIR/average_metrics.txt"}"
