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
├── scripts/                # Main scripts
│   ├── train.py            # Training script (optional fine-tuning)
│   ├── test.py             # Testing and evaluation script
│   └── demo.py             # Inference demo script
├── utils/                  # Utility functions
│   ├── metrics.py          # Evaluation metrics
│   └── prompt_engineering.py  # Prompt generation
├── configs/                # Configuration files
│   ├── config.yaml         # Main config
│   ├── mvtec.yaml          # MVTec specific config
│   └── visa.yaml           # VisA specific config
├── assets/                 # Example images and results
├── README.md               # This file
├── requirements.txt        # Dependencies
├── train.sh                # Training script
└── test.sh                 # Testing script
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

### Testing

Evaluate the model on a dataset:

```bash
python scripts/test.py \
    --dataset mvtec \
    --dataset_path ~/autodl-tmp/datasets \
    --model_path ~/autodl-tmp/download/sam3 \
    --output_dir ./test_results \
    --prompt_strategy mixed \
    --visualize

# On AutoDL platform
python scripts/test.py \
    --dataset mvtec \
    --dataset_path ~/autodl-tmp/datasets \
    --model_path ~/autodl-tmp/download/sam3 \
    --output_dir ./test_results \
    --prompt_strategy mixed \
    --visualize
```

### Training (Optional Fine-tuning)

Fine-tune the model on a dataset:

```bash
python scripts/train.py \
    --dataset mvtec \
    --dataset_path ~/autodl-tmp/datasets \
    --model_path ~/autodl-tmp/download/sam3 \
    --output_dir ./train_results \
    --num_epochs 10 \
    --lr 1e-5

# On AutoDL platform
python scripts/train.py \
    --dataset mvtec \
    --dataset_path ~/autodl-tmp/datasets \
    --model_path ~/autodl-tmp/download/sam3 \
    --output_dir ./train_results \
    --num_epochs 10 \
    --lr 1e-5
```

### Using Configuration Files

You can use YAML configuration files for easier setup:

```bash
python scripts/test.py --config configs/mvtec.yaml
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
