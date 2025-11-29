"""Test script for SAM3 Anomaly Detection"""
import sys
import os

# Add the project root directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from PIL import Image
from dataset import MVTecDataset, VisaDataset
from model.sam3_anomaly_detector import SAM3AnomalyDetector
from utils.prompt_engineering import AnomalyPromptGenerator
from utils.metrics import calculate_all_metrics, print_metrics


def parse_args():
    parser = argparse.ArgumentParser(description="SAM3 Anomaly Detection Test")
    parser.add_argument("--dataset", type=str, required=True, choices=["mvtec", "visa"], help="Dataset name")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to dataset root")
    parser.add_argument("--model_path", type=str, required=True, help="Path to SAM3 model checkpoint")
    parser.add_argument("--output_dir", type=str, default="test_results", help="Directory to save results")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run model on")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--confidence_threshold", type=float, default=0.5, help="Confidence threshold")
    parser.add_argument("--iou_threshold", type=float, default=0.5, help="IoU threshold for NMS")
    parser.add_argument("--prompt_strategy", type=str, default="mixed", choices=["base", "category", "mixed", "contextual"], help="Prompt strategy")
    parser.add_argument("--num_prompts", type=int, default=8, help="Number of prompts to use")
    parser.add_argument("--category", type=str, default=None, help="Specific category to test (None for all)")
    parser.add_argument("--seed", type=int, default=122, help="Random seed")
    parser.add_argument("--visualize", action="store_true", help="Save visualization results")
    return parser.parse_args()


def get_dataset(dataset_name, dataset_path, category=None):
    """Get dataset based on name"""
    if dataset_name == "mvtec":
        dataset = MVTecDataset(
            root=dataset_path,
            train=False,
            category=category,
            transform=None,
            gt_target_transform=None
        )
    elif dataset_name == "visa":
        dataset = VisaDataset(
            root=dataset_path,
            train=False,
            category=category,
            transform=None,
            gt_target_transform=None
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    return dataset


def test_category(category, dataset_name, dataset_path, detector, prompt_generator, args):
    """Test on a specific category"""
    print(f"\n=== Testing category: {category} ===")
    
    # Create dataset and dataloader
    dataset = get_dataset(dataset_name, dataset_path, category)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    
    # Create category output directory
    category_output_dir = os.path.join(args.output_dir, category)
    os.makedirs(category_output_dir, exist_ok=True)
    
    # Initialize lists for metrics
    y_true_images = []
    y_score_images = []
    y_true_pixels = []
    y_score_pixels = []
    
    # Iterate over test set
    for i, (images, labels, gts, categories, img_paths) in enumerate(dataloader):
        for j in range(len(images)):
            # Get image, label, gt, and path
            img_path = img_paths[j]
            label = labels[j].item()
            gt = gts[j].numpy() if gts[j] is not None else np.zeros((images[j].shape[1], images[j].shape[2]))
            
            print(f"Processing image {i*args.batch_size + j + 1}/{len(dataset)}: {os.path.basename(img_path)}")
            
            # Generate prompts
            prompts = prompt_generator.get_prompts_for_inference(
                dataset_name, 
                category, 
                args.prompt_strategy, 
                args.num_prompts
            )
            
            # Load image
            image = Image.open(img_path).convert("RGB")
            
            # Detect anomalies
            results = detector.detect_anomaly(
                image, 
                text_prompts=prompts,
                confidence_threshold=args.confidence_threshold,
                iou_threshold=args.iou_threshold
            )
            
            # Calculate image-level score
            if results["scores"]:
                img_score = max(results["scores"])
            else:
                img_score = 0.0
            
            # Generate pixel-level score map
            anomaly_mask = detector.get_anomaly_mask(results, image.size)
            score_map = anomaly_mask.astype(np.float32)
            
            # Resize score map to match gt size
            gt_size = gt.shape[:2]
            if score_map.shape != gt_size:
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
                
                # Save anomaly mask
                mask_image = Image.fromarray(anomaly_mask * 255)
                mask_path = os.path.join(category_output_dir, f"mask_{image_name}")
                mask_image.save(mask_path)
    
    # Calculate metrics
    metrics = calculate_all_metrics(
        y_true_images, 
        y_score_images, 
        y_true_pixels, 
        y_score_pixels
    )
    
    # Print metrics
    print_metrics(metrics, f"{dataset_name} - {category}")
    
    # Save metrics to file
    metrics_file = os.path.join(category_output_dir, "metrics.txt")
    with open(metrics_file, "w") as f:
        for key, value in metrics.items():
            f.write(f"{key}: {value:.6f}\n")
    
    return metrics


def main():
    args = parse_args()
    
    # Set seed
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize detector and prompt generator
    detector = SAM3AnomalyDetector(args.model_path, args.device, args.seed)
    prompt_generator = AnomalyPromptGenerator()
    
    # Get all categories if none specified
    if args.category:
        categories = [args.category]
    else:
        if args.dataset == "mvtec":
            categories = MVTecDataset.categories
        elif args.dataset == "visa":
            categories = VisaDataset.categories
    
    # Test each category
    all_metrics = {}
    for category in categories:
        category_metrics = test_category(
            category, 
            args.dataset, 
            args.dataset_path, 
            detector, 
            prompt_generator, 
            args
        )
        all_metrics[category] = category_metrics
    
    # Calculate average metrics across categories
    if len(categories) > 1:
        avg_metrics = {}
        for metric_name in all_metrics[categories[0]].keys():
            avg_value = np.mean([metrics[metric_name] for metrics in all_metrics.values()])
            avg_metrics[metric_name] = avg_value
        
        print(f"\n=== Average Metrics across {len(categories)} categories ===")
        print_metrics(avg_metrics, f"{args.dataset} - Average")
        
        # Save average metrics
        avg_metrics_file = os.path.join(args.output_dir, "average_metrics.txt")
        with open(avg_metrics_file, "w") as f:
            for key, value in avg_metrics.items():
                f.write(f"{key}: {value:.6f}\n")
    
    print("\n=== Test completed successfully! ===")
    print(f"Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
