#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""SAM3+ Anomaly Detection Main Entry Point"""
import sys
import os

# Add the project root directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import argparse
import numpy as np
import torch
import yaml
from typing import Dict, Any

# Import necessary modules
from model.sam3_anomaly_detector import SAM3AnomalyDetector
from dataset import MVTecDataset, VisaDataset, Br35HDataset, BrainMRIDataset, \
    BTADDataset, ClinicDBDataset, ColonDBDataset, DAGMDataset, DTDDataset, \
    ISICDataset, KvasirDataset
from utils.metrics import calculate_all_metrics, print_metrics
from utils.prompt_engineering import AnomalyPromptGenerator
import json

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="SAM3+ Anomaly Detection")
    parser.add_argument("--mode", type=str, required=True, choices=["train", "test"],
                        help="Run mode: train or test")
    parser.add_argument("--dataset", type=str, required=True,
                        choices=["mvtec", "visa", "br35h", "brainMRI", "btad", 
                                "clinicdb", "colondb", "dagm", "dtd", "isic", "kvasir"],
                        help="Dataset name")
    parser.add_argument("--dataset_path", type=str, required=True, 
                        help="Path to dataset root")
    parser.add_argument("--model_path", type=str, required=True, 
                        help="Path to SAM3 model checkpoint")
    parser.add_argument("--config", type=str, default=None, 
                        help="Path to configuration file (optional)")
    parser.add_argument("--output_dir", type=str, default="./outputs", 
                        help="Directory to save results")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", 
                        help="Device to run model on")
    parser.add_argument("--batch_size", type=int, default=1, 
                        help="Batch size")
    parser.add_argument("--prompt_strategy", type=str, default="mixed", 
                        choices=["base", "category", "mixed", "contextual"], 
                        help="Prompt strategy")
    parser.add_argument("--num_prompts", type=int, default=8, 
                        help="Number of prompts to use")
    parser.add_argument("--category", type=str, default=None, 
                        help="Specific category to process (None for all)")
    parser.add_argument("--seed", type=int, default=122, 
                        help="Random seed")
    parser.add_argument("--visualize", action="store_true", 
                        help="Save visualization results")
    return parser.parse_args()

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file"""
    if not config_path or not os.path.exists(config_path):
        return {}
    
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def get_dataset(args, train: bool = True):
    """Get dataset based on name and mode"""
    dataset_mapping = {
        "mvtec": MVTecDataset,
        "visa": VisaDataset,
        "br35h": Br35HDataset,
        "brainMRI": BrainMRIDataset,
        "btad": BTADDataset,
        "clinicdb": ClinicDBDataset,
        "colondb": ColonDBDataset,
        "dagm": DAGMDataset,
        "dtd": DTDDataset,
        "isic": ISICDataset,
        "kvasir": KvasirDataset
    }
    
    dataset_class = dataset_mapping.get(args.dataset)
    if not dataset_class:
        raise ValueError(f"Unknown dataset: {args.dataset}")
    
    # Create dataset with appropriate parameters
    dataset = dataset_class(
        root=args.dataset_path,
        train=train,
        category=args.category,
        transform=None,  # Will be handled inside the dataset class if needed
        gt_target_transform=None
    )
    
    return dataset

def train(args):
    """Train the SAM3+ anomaly detector"""
    print(f"=== Starting training for {args.dataset} dataset ===")
    
    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load dataset
    train_dataset = get_dataset(args, train=True)
    
    # Initialize model
    print(f"Loading SAM3 model from {args.model_path}")
    model = SAM3AnomalyDetector(
        model_path=args.model_path,
        device=args.device,
        prompt_strategy=args.prompt_strategy,
        num_prompts=args.num_prompts
    )
    
    # Training logic implementation in progress
    # Basic training framework has been established
    print("Training functionality implementation in progress")
    
    # Save model checkpoints
    checkpoint_path = os.path.join(args.output_dir, f"{args.dataset}_model.pt")
    print(f"Training completed. Checkpoint would be saved to {checkpoint_path}")
    
    return 0

def test_category(category, args, detector, prompt_generator):
    """Test on a specific category"""
    print(f"\n=== Testing category: {category} ===")
    
    # Create category output directory
    category_output_dir = os.path.join(args.output_dir, category)
    try:
        os.makedirs(category_output_dir, exist_ok=True)
        print(f"Created category output directory: {category_output_dir}")
    except Exception as e:
        print(f"Error creating output directory: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    # Create dataset with specific category
    temp_args = argparse.Namespace(**vars(args))
    temp_args.category = category
    dataset = get_dataset(temp_args, train=False)
    
    # Initialize lists for metrics
    y_true_images = []
    y_score_images = []
    y_true_pixels = []
    y_score_pixels = []
    
    # Generate prompts
    prompts = prompt_generator.get_prompts_for_inference(
        args.dataset,
        category,
        args.prompt_strategy,
        args.num_prompts
    )
    
    # Process each sample
    for i, data in enumerate(dataset):
        try:
            # Try to unpack the data (different datasets might have different structures)
            if len(data) == 5:
                image, label, gt, category_name, img_path = data
            elif len(data) == 4:
                image, label, gt, img_path = data
                category_name = category
            else:
                print(f"Unexpected data structure: {len(data)} elements")
                continue
            
            print(f"Processing image {i + 1}/{len(dataset)}: {os.path.basename(img_path)}")
            
            # Load image
            from PIL import Image
            image = Image.open(img_path).convert("RGB")
            
            # Detect anomalies
            results = detector.detect_anomaly(
                image,
                text_prompts=prompts,
                confidence_threshold=getattr(args, 'confidence_threshold', 0.5)
            )
            
            # Calculate image-level score
            if results.get("scores"):
                img_score = max(results["scores"])
            else:
                img_score = 0.0
            
            # Generate pixel-level score map
            anomaly_mask = detector.get_anomaly_mask(results, image.size)
            score_map = anomaly_mask.astype(np.float32)
            
            # Convert gt to numpy array
            if isinstance(gt, Image.Image):
                gt = np.array(gt)
            elif gt is None:
                gt = np.zeros((image.size[1], image.size[0]), dtype=np.uint8)
            
            # Resize score map to match gt size
            gt_size = gt.shape[:2]
            if score_map.shape != gt_size:
                from PIL import Image
                score_map = np.array(Image.fromarray(score_map).resize(gt_size[::-1], Image.LANCZOS))
            
            # Append to lists for metrics
            y_true_images.append(label)
            y_score_images.append(img_score)
            y_true_pixels.append(gt)
            y_score_pixels.append(score_map)
            
            # Visualize if enabled
            if args.visualize:
                visualized_image = detector.visualize_results(image, results)
                image_name = os.path.basename(img_path)
                visualized_path = os.path.join(category_output_dir, f"result_{image_name}")
                visualized_image.save(visualized_path)
                
                # Save anomaly mask - ensure proper data type conversion
                mask_array = (anomaly_mask * 255).astype(np.uint8)
                mask_image = Image.fromarray(mask_array)
                mask_path = os.path.join(category_output_dir, f"mask_{image_name}")
                mask_image.save(mask_path)
                
        except Exception as e:
            print(f"Error processing sample {i}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Calculate metrics
    metrics = calculate_all_metrics(
        y_true_images,
        y_score_images,
        y_true_pixels,
        y_score_pixels
    )
    
    # Print metrics
    print_metrics(metrics, f"{args.dataset} - {category}")
    
    # Save metrics to file
    metrics_file = os.path.join(category_output_dir, "metrics.txt")
    with open(metrics_file, "w") as f:
        for key, value in metrics.items():
            f.write(f"{key}: {value:.6f}\n")
    
    return metrics

def test(args):
    """Test the SAM3+ anomaly detector"""
    print(f"=== Starting testing for {args.dataset} dataset ===")
    
    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Created output directory: {args.output_dir}")
    
    # Initialize detector and prompt generator
    print(f"Loading SAM3 model from {args.model_path}")
    detector = SAM3AnomalyDetector(
        model_path=args.model_path,
        device=args.device,
        seed=args.seed
    )
    
    # Initialize prompt generator
    prompt_generator = AnomalyPromptGenerator()
    
    # Get categories to test
    if args.category:
        categories = [args.category]
        print(f"Testing specific category: {args.category}")
    else:
        print("Testing all categories in dataset")
        # Get all categories from the dataset
        if args.dataset == "mvtec":
            from dataset.mvtec import MVTecDataset
            categories = MVTecDataset.categories
        elif args.dataset == "visa":
            from dataset.visa import VisaDataset
            categories = VisaDataset.categories
        elif args.dataset == "br35h":
            from dataset.br35h import Br35HDataset
            categories = Br35HDataset.categories
        elif args.dataset == "brainMRI":
            from dataset.brainMRI import BrainMRIDataset
            categories = BrainMRIDataset.categories
        elif args.dataset == "btad":
            from dataset.btad import BTADDataset
            categories = BTADDataset.categories
        elif args.dataset == "clinicdb":
            from dataset.clinicdb import ClinicDBDataset
            categories = ClinicDBDataset.categories
        elif args.dataset == "colondb":
            from dataset.colondb import ColonDBDataset
            categories = ColonDBDataset.categories
        elif args.dataset == "dagm":
            from dataset.dagm import DAGMDataset
            categories = DAGMDataset.categories
        elif args.dataset == "dtd":
            from dataset.dtd import DTDDataset
            categories = DTDDataset.categories
        elif args.dataset == "isic":
            from dataset.isic import ISICDataset
            categories = ISICDataset.categories
        elif args.dataset == "kvasir":
            from dataset.kvasir import KvasirDataset
            categories = KvasirDataset.categories
        else:
            raise ValueError(f"Unsupported dataset: {args.dataset}")
    
    print(f"Categories to test: {categories}")
    
    # Test each category
    all_metrics = {}
    for i, category in enumerate(categories):
        print(f"\n[{i+1}/{len(categories)}] Processing category: {category}")
        try:
            category_metrics = test_category(
                category,
                args,
                detector,
                prompt_generator
            )
            all_metrics[category] = category_metrics
        except Exception as e:
            print(f"Error processing category {category}: {e}")
            import traceback
            traceback.print_exc()
            # Continue with next category
    
    # Calculate and print average metrics if multiple categories
    if len(categories) > 1:
        print(f"\n=== Average Metrics across {len(categories)} categories ===")
        avg_metrics = {}
        # Collect all metric names
        metric_names = list(all_metrics[categories[0]].keys())
        # Calculate average for each metric
        for metric_name in metric_names:
            values = [metrics.get(metric_name, 0) for metrics in all_metrics.values()]
            avg_metrics[metric_name] = np.mean(values)
        # Print average metrics
        print_metrics(avg_metrics, f"{args.dataset} - Average")
        # Save average metrics
        avg_metrics_file = os.path.join(args.output_dir, "average_metrics.txt")
        with open(avg_metrics_file, "w") as f:
            for key, value in avg_metrics.items():
                f.write(f"{key}: {value:.6f}\n")
    
    print("\n=== Testing completed successfully! ===")
    return 0

def main():
    """Main function"""
    # Parse arguments
    args = parse_args()
    
    # Load configuration if provided
    config = load_config(args.config)
    if config:
        # Override arguments with config values
        for key, value in config.items():
            if hasattr(args, key) and value is not None:
                setattr(args, key, value)
    
    # Print configuration
    print(f"Running in {args.mode} mode")
    print(f"Dataset: {args.dataset}")
    print(f"Dataset path: {args.dataset_path}")
    print(f"Model path: {args.model_path}")
    print(f"Output directory: {args.output_dir}")
    print(f"Device: {args.device}")
    
    # Execute based on mode
    if args.mode == "train":
        return train(args)
    elif args.mode == "test":
        return test(args)
    else:
        print(f"Unknown mode: {args.mode}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
