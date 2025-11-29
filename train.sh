#!/bin/bash

# Training script for SAM3 Anomaly Detection using main.py

# Default parameters
DATASET="mvtec"
DATASET_PATH="~/autodl-tmp/datasets"
MODEL_PATH="~/autodl-tmp/download/sam3"
OUTPUT_DIR="./train_results"
DEVICE="cuda"
BATCH_SIZE=1
PROMPT_STRATEGY="mixed"
NUM_PROMPTS=8
CATEGORY=""
SEED=122
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
        --num_epochs)
            NUM_EPOCHS=$2
            shift 2
            ;;
        --lr)
            LR=$2
            shift 2
            ;;
        --weight_decay)
            WEIGHT_DECAY=$2
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
        --config)
            CONFIG="--config $2"
            shift 2
            ;;
        *)
            echo "Unknown parameter: $1"
            exit 1
            ;;
    esac
done

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run the training script using main.py
python main.py \
    --mode train \
    --dataset "$DATASET" \
    --dataset_path "$DATASET_PATH" \
    --model_path "$MODEL_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --device "$DEVICE" \
    --batch_size $BATCH_SIZE \
    --prompt_strategy "$PROMPT_STRATEGY" \
    --num_prompts $NUM_PROMPTS \
    --category "$CATEGORY" \
    --seed $SEED \
    $CONFIG
