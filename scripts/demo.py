"""Demo script for SAM3 Anomaly Detection"""
import sys
import os

# Add the project root directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import numpy as np
from PIL import Image

# Skip BPE vocab download since facebook/sam3 is a gated repo
print("📝 Skipping BPE vocab download - will use existing tokenizer files")

from model.sam3_anomaly_detector import SAM3AnomalyDetector
from utils.prompt_engineering import AnomalyPromptGenerator


def parse_args():
    parser = argparse.ArgumentParser(description="SAM3 Anomaly Detection Demo")
    parser.add_argument("--image_path", type=str, required=True, help="Path to image or directory of images")
    parser.add_argument("--model_path", type=str, required=True, help="Path to SAM3 model checkpoint")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Directory to save results")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run model on")
    parser.add_argument("--confidence_threshold", type=float, default=0.5, help="Confidence threshold")
    parser.add_argument("--iou_threshold", type=float, default=0.5, help="IoU threshold for NMS")
    parser.add_argument("--prompt_strategy", type=str, default="mixed", choices=["base", "category", "mixed", "contextual"], help="Prompt strategy")
    parser.add_argument("--dataset_name", "--dataset", type=str, default="mvtec", choices=["mvtec", "visa"], help="Dataset name for prompt generation")
    parser.add_argument("--category", type=str, default="carpet", help="Category for prompt generation")
    parser.add_argument("--num_prompts", type=int, default=8, help="Number of prompts to use")
    parser.add_argument("--seed", type=int, default=122, help="Random seed")
    return parser.parse_args()


def process_single_image(image_path, detector, prompt_generator, args):
    """Process a single image"""
    print(f"Processing image: {image_path}")
    
    # Generate prompts
    prompts = prompt_generator.get_prompts_for_inference(
        args.dataset_name, 
        args.category, 
        args.prompt_strategy, 
        args.num_prompts
    )
    print(f"Using prompts: {prompts}")
    
    # Load image
    image = Image.open(image_path).convert("RGB")
    
    # Detect anomalies
    results = detector.detect_anomaly(
        image, 
        text_prompts=prompts,
        confidence_threshold=args.confidence_threshold,
        iou_threshold=args.iou_threshold
    )
    
    # Visualize results
    visualized_image = detector.visualize_results(image, results)
    
    # Save results
    image_name = os.path.basename(image_path)
    output_path = os.path.join(args.output_dir, f"result_{image_name}")
    visualized_image.save(output_path)
    print(f"Results saved to: {output_path}")
    
    # Save anomaly mask
    if results["masks"]:
        mask = detector.get_anomaly_mask(results, image.size)
        mask_image = Image.fromarray(mask * 255)
        mask_output_path = os.path.join(args.output_dir, f"mask_{image_name}")
        mask_image.save(mask_output_path)
        print(f"Anomaly mask saved to: {mask_output_path}")
    
    return results


def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize detector and prompt generator
    detector = SAM3AnomalyDetector(args.model_path, args.device, args.seed)
    prompt_generator = AnomalyPromptGenerator()
    
    # Process images
    if os.path.isdir(args.image_path):
        # Process all images in directory
        image_files = [f for f in os.listdir(args.image_path) if f.endswith((".jpg", ".jpeg", ".png", ".bmp"))]
        for image_file in image_files:
            image_path = os.path.join(args.image_path, image_file)
            process_single_image(image_path, detector, prompt_generator, args)
    else:
        # Process single image
        process_single_image(args.image_path, detector, prompt_generator, args)
    
    print("\nDemo completed successfully!")


if __name__ == "__main__":
    main()
