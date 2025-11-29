#!/bin/bash

# Testing script for SAM3 Anomaly Detection using main.py
# By default, it will test all categories in the dataset unless a specific category is provided
# Example usage:
# - Test all categories: ./test.sh
# - Test specific category: ./test.sh --category bottle

# Default parameters
DATASET="mvtec"
DATASET_PATH="~/autodl-tmp/datasets"
MODEL_PATH="~/autodl-tmp/download/sam3"
OUTPUT_DIR="./test_results"
DEVICE="cuda"
BATCH_SIZE=1
CONFIDENCE_THRESHOLD=0.5
IOU_THRESHOLD=0.5
PROMPT_STRATEGY="mixed"
NUM_PROMPTS=8
CATEGORY=""
SEED=122
VISUALIZE=false
CONFIG=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --dataset)
            DATASET="$2"
            shift 2
            ;;
        --dataset_path)
            DATASET_PATH="$2"
            shift 2
            ;;
        --model_path)
            MODEL_PATH="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE=$2
            shift 2
            ;;
        --confidence_threshold)
            CONFIDENCE_THRESHOLD=$2
            shift 2
            ;;
        --iou_threshold)
            IOU_THRESHOLD=$2
            shift 2
            ;;
        --prompt_strategy)
            PROMPT_STRATEGY="$2"
            shift 2
            ;;
        --num_prompts)
            NUM_PROMPTS=$2
            shift 2
            ;;
        --category)
            CATEGORY="$2"
            shift 2
            ;;
        --seed)
            SEED=$2
            shift 2
            ;;
        --visualize)
            VISUALIZE=true
            shift 1
            ;;
        *)
            echo "Unknown parameter: $1"
            exit 1
            ;;
    esac
done

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Print configuration
echo "=== Test Configuration ==="
echo "Dataset: $DATASET"
echo "Dataset Path: $DATASET_PATH"
echo "Model Path: $MODEL_PATH"
echo "Output Directory: $OUTPUT_DIR"
echo "Device: $DEVICE"
echo "Batch Size: $BATCH_SIZE"
echo "Confidence Threshold: $CONFIDENCE_THRESHOLD"
echo "IOU Threshold: $IOU_THRESHOLD"
echo "Prompt Strategy: $PROMPT_STRATEGY"
echo "Number of Prompts: $NUM_PROMPTS"
echo "Seed: $SEED"
echo "Category: ${CATEGORY:-All categories}"
echo "Visualization: $VISUALIZE"
echo "Config File: ${CONFIG:-None}"
echo "======================="

# Build visualize argument
VISUALIZE_ARG=""
if [ "$VISUALIZE" = true ]; then
    VISUALIZE_ARG="--visualize"
fi

# Run the testing script using main.py
COMMAND="python main.py \
    --mode test \
    --dataset \"$DATASET\" \
    --dataset_path \"$DATASET_PATH\" \
    --model_path \"$MODEL_PATH\" \
    --output_dir \"$OUTPUT_DIR\" \
    --device \"$DEVICE\" \
    --batch_size $BATCH_SIZE \
    --confidence_threshold $CONFIDENCE_THRESHOLD \
    --iou_threshold $IOU_THRESHOLD \
    --prompt_strategy \"$PROMPT_STRATEGY\" \
    --num_prompts $NUM_PROMPTS \
    --seed $SEED"

# Add category parameter only if it's provided
if [ -n "$CATEGORY" ]; then
    COMMAND="$COMMAND \\
    --category \"$CATEGORY\""
fi

# Add visualize argument if set
if [ -n "$VISUALIZE_ARG" ]; then
    COMMAND="$COMMAND \\
    $VISUALIZE_ARG"
fi

# Add config argument if provided
if [ -n "$CONFIG" ]; then
    COMMAND="$COMMAND \\
    $CONFIG"
fi

# Execute the command
echo "Starting test process..."
eval "$COMMAND"

echo "\n=== Testing completed ==="
echo "- Category-level metrics are saved in respective category directories under $OUTPUT_DIR"
echo "- Overall average metrics are saved in $OUTPUT_DIR/average_metrics.txt (if multiple categories tested)"
