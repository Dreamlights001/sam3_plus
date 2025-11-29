#!/bin/bash

# Testing script for SAM3 Anomaly Detection

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

# Build visualize argument
VISUALIZE_ARG=""
if [ "$VISUALIZE" = true ]; then
    VISUALIZE_ARG="--visualize"
fi

# Run testing
python scripts/test.py \
    --dataset "$DATASET" \
    --dataset_path "$DATASET_PATH" \
    --model_path "$MODEL_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --device "$DEVICE" \
    --batch_size $BATCH_SIZE \
    --confidence_threshold $CONFIDENCE_THRESHOLD \
    --iou_threshold $IOU_THRESHOLD \
    --prompt_strategy "$PROMPT_STRATEGY" \
    --num_prompts $NUM_PROMPTS \
    --category "$CATEGORY" \
    --seed $SEED \
    $VISUALIZE_ARG
