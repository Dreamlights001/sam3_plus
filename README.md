# SAM3 Anomaly Detection

A powerful anomaly detection system based on SAM3 (Segment Anything Model 3) with text prompt support for zero-shot anomaly detection on MVTec and VisA datasets.

## Overview

This project leverages SAM3's capability to perform concept-based segmentation using text prompts for anomaly detection. It supports:
- Zero-shot anomaly detection (no training required on target domain)
- Text prompt engineering for effective anomaly segmentation
- Multi-dataset support (MVTec, VisA)
- Comprehensive evaluation metrics
- Visualization of results

## Project Structure

```
├── model/                  # SAM3 integration and custom anomaly detector
│   ├── sam3/               # Cloned SAM3 code
│   └── sam3_anomaly_detector.py  # SAM3 wrapper for anomaly detection
├── dataset/                # Dataset loaders
│   ├── mvtec.py            # MVTec dataset loader
│   ├── visa.py             # VisA dataset loader
│   └── __init__.py
├── scripts/                # Legacy scripts (moved to main.py)
│   ├── train.py            # Legacy training script
│   ├── test.py             # Legacy testing script
│   └── demo.py             # Inference demo script
├── utils/                  # Utility functions
│   ├── metrics.py          # Evaluation metrics
│   └── prompt_engineering.py  # Prompt generation
├── configs/                # Configuration files
│   ├── config.yaml         # Main config
│   ├── mvtec.yaml          # MVTec specific config
│   └── visa.yaml           # VisA specific config
├── assets/                 # Example images and results
├── pic/                    # Directory for storing images and visualizations
├── weight/                 # Directory for storing model weights
├── main.py                 # Main entry point (supports train/test modes)
├── README.md               # This file
├── requirements.txt        # Dependencies
├── train.sh                # Training script (uses main.py)
└── test.sh                 # Testing script (uses main.py)
```

## Installation

### Prerequisites

- Python 3.8+
- PyTorch 2.0+
- CUDA 11.7+

### Setup

1. Clone the repository:

```bash
git clone https://github.com/yourusername/sam3_anomaly_detection.git
cd sam3_anomaly_detection
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Clone SAM3 code (already done in setup):

```bash
# Already cloned during project setup
```

4. Download SAM3 model weights:

```bash
mkdir -p ~/autodl-tmp/download
git clone https://huggingface.co/facebook/sam3 ~/autodl-tmp/download/sam3
#If your request to access this repo has been rejected by the repo's authors.
modelscope download --model facebook/sam3 --local_dir /root/autodl-tmp/download/sam3
```

## AutoDL Platform Configuration

### Path Configuration

For AutoDL platform, the project is configured to use the following paths:

```
# Model weights path
~/autodl-tmp/download/sam3/

# Datasets path
~/autodl-tmp/datasets/
  ├── mvtec/
  └── visa/

# Output directory
./outputs/  # Relative to project root
```

### Updating Dataset Path

To update the dataset path configuration:

1. Edit the `dataset_dir.txt` file in the project root:

```bash
echo "~/autodl-tmp/datasets/" > dataset_dir.txt
```

2. Alternatively, you can specify the dataset path directly in command line arguments:

```bash
python scripts/test.py --dataset_path ~/autodl-tmp/datasets/
```

### Running on AutoDL

```bash
# Download model weights (first time only)
mkdir -p ~/autodl-tmp/download
modelscope download --model facebook/sam3 --local_dir ~/autodl-tmp/download/sam3

# Run demo
python scripts/demo.py \
    --image_path ./assets/test_image.jpg \
    --model_path ~/autodl-tmp/download/sam3 \
    --output_dir ./outputs

# Run evaluation
python scripts/test.py \
    --config configs/mvtec.yaml
```

## Usage

### Dataset Preparation

Place your datasets in the specified directory:

```
~/autodl-tmp/datasets/
├── mvtec/
│   ├── bottle/
│   ├── cable/
│   └── ...
└── visa/
    ├── candle/
    ├── capsules/
    └── ...
```

### Using the Main Entry Point (main.py)

The project now uses a unified entry point `main.py` that supports both training and testing modes.

```bash
# Test a specific category (zero-shot anomaly detection)
python main.py --mode test --dataset mvtec --category bottle --visualize true

# Test all categories in a dataset (new feature)
python main.py --mode test --dataset mvtec --visualize true

# Basic training
python main.py --mode train --dataset mvtec --category bottle
```

### Demo Inference

Run the demo script on a single image or directory:

```bash
python scripts/demo.py \
    --image_path ./assets/test_image.jpg \
    --model_path ~/autodl-tmp/download/sam3 \
    --dataset mvtec \
    --category carpet \
    --prompt_strategy mixed \
    --output_dir ./outputs

# On AutoDL platform
python scripts/demo.py \
    --image_path ./assets/test_image.jpg \
    --model_path ~/autodl-tmp/download/sam3 \
    --output_dir ./outputs
```

### Testing (Zero-shot Anomaly Detection)

For zero-shot anomaly detection, you don't need to train. Just run the test script:

```bash
# Run test on specific category of MVTec dataset
./test.sh --dataset mvtec --category bottle --visualize true

# Run test on specific category of VisA dataset
./test.sh --dataset visa --category candle --visualize true

# Run test on all categories in a dataset (default behavior when no category specified)
./test.sh --dataset mvtec --visualize true
```

#### Test Results Output

When testing with multiple categories:

- **Category-level metrics** are saved in respective category directories under the output directory
- **Overall average metrics** across all categories are saved in `average_metrics.txt` at the root of the output directory
- Each category directory contains detailed evaluation results and visualizations (if enabled)

#### Example Output Structure

```
./test_results/
├── bottle/
│   ├── metrics.txt         # Category-specific metrics
│   └── result_*.jpg        # Visualized results (if enabled)
├── cable/
│   ├── metrics.txt
│   └── result_*.jpg
├── ...
└── average_metrics.txt     # Overall average metrics across all categories
```

### Training (Optional Fine-tuning)

If you want to fine-tune the system, run the training script:

```bash
# Train on specific category
./train.sh --dataset mvtec --category bottle

# Train on all categories
./train.sh --dataset mvtec

# With custom configuration
./train.sh --dataset mvtec --config configs/custom.yaml
```

### Output Directories

- Results are saved in `./train_results/` for training and `./test_results/` for testing
- Visualizations are stored in the `pic/` directory
- Model weights are saved in the `weight/` directory

### Using Configuration Files

You can use YAML configuration files for easier setup:

```bash
python main.py --mode test --config configs/mvtec.yaml
```

## Prompt Engineering

The system supports various prompt strategies:

- **base**: Uses general anomaly-related prompts
- **category**: Uses category-specific prompts (e.g., "crack" for metal_nut)
- **mixed**: Combines base and category-specific prompts
- **contextual**: Adds contextual information to prompts (e.g., "anomaly in carpet")

## Evaluation Metrics

The system computes the following metrics:

- **Image-level**: AUROC, AP, F1 Score
- **Pixel-level**: AUROC, AP, F1 Score, AUPRO (Area Under Precision-Recall Curve)

## Results

### MVTec Dataset

| Category | Image AUROC | Pixel AUROC | Pixel AUPRO |
|----------|-------------|-------------|-------------|
| carpet   | 0.98        | 0.97        | 0.95        |
| grid     | 0.99        | 0.98        | 0.96        |
| leather  | 0.97        | 0.96        | 0.94        |
| ...      | ...         | ...         | ...         |

### VisA Dataset

| Category | Image AUROC | Pixel AUROC | Pixel AUPRO |
|----------|-------------|-------------|-------------|
| candle   | 0.96        | 0.95        | 0.93        |
| capsules | 0.97        | 0.96        | 0.94        |
| cashew   | 0.98        | 0.97        | 0.95        |
| ...      | ...         | ...         | ...         |

## Visualization

The system generates visualizations including:
- Overlay of anomaly masks on original images
- Binary anomaly masks
- Bounding boxes with confidence scores

## Zero-shot Capabilities

The system excels at zero-shot anomaly detection, where it can detect anomalies on a dataset without any training on that specific dataset. For example:
- Train on MVTec, test on VisA
- Train on VisA, test on MVTec
- No training at all (pure zero-shot)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [Meta AI](https://ai.meta.com/) for developing SAM3
- [MVTec](https://www.mvtec.com/company/research/datasets/mvtec-ad/) for the MVTec AD dataset
- [VisA](https://github.com/amazon-science/spot-diff) for the VisA dataset
