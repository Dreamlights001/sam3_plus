"""Train script for SAM3 Anomaly Detection"""
import sys
import os

# Add the project root directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import MVTecDataset, VisaDataset
from model.sam3_anomaly_detector import SAM3AnomalyDetector
from utils.prompt_engineering import AnomalyPromptGenerator


def parse_args():
    parser = argparse.ArgumentParser(description="SAM3 Anomaly Detection Train")
    parser.add_argument("--dataset", type=str, required=True, choices=["mvtec", "visa"], help="Dataset name")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to dataset root")
    parser.add_argument("--model_path", type=str, required=True, help="Path to SAM3 model checkpoint")
    parser.add_argument("--output_dir", type=str, default="train_results", help="Directory to save results")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run model on")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--prompt_strategy", type=str, default="mixed", choices=["base", "category", "mixed", "contextual"], help="Prompt strategy")
    parser.add_argument("--num_prompts", type=int, default=8, help="Number of prompts to use")
    parser.add_argument("--category", type=str, default=None, help="Specific category to train (None for all)")
    parser.add_argument("--seed", type=int, default=122, help="Random seed")
    parser.add_argument("--fewshot", type=int, default=0, help="Number of few-shot samples")
    return parser.parse_args()


def get_dataset(dataset_name, dataset_path, category=None, fewshot=0):
    """Get dataset based on name"""
    if dataset_name == "mvtec":
        dataset = MVTecDataset(
            root=dataset_path,
            train=True,
            category=category,
            fewshot=fewshot,
            transform=None,
            gt_target_transform=None
        )
    elif dataset_name == "visa":
        dataset = VisaDataset(
            root=dataset_path,
            train=True,
            category=category,
            fewshot=fewshot,
            transform=None,
            gt_target_transform=None
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    return dataset


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
    
    # Note: SAM3 is primarily designed for zero-shot inference
    # This script provides a basic framework for optional fine-tuning
    # For full fine-tuning, refer to SAM3's official training script
    print("Note: SAM3 is primarily designed for zero-shot inference.")
    print("This script provides a basic framework for optional fine-tuning.")
    print("For full fine-tuning, refer to SAM3's official training script.")
    
    # Get dataset and dataloader
    dataset = get_dataset(args.dataset, args.dataset_path, args.category, args.fewshot)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    # Initialize optimizer (only fine-tune specific layers)
    # Here we fine-tune the text encoder and vision-language combiner
    params_to_optimize = [
        {"params": detector.model.text_encoder.parameters(), "lr": args.lr},
        {"params": detector.model.vl_combiner.parameters(), "lr": args.lr * 10},
    ]
    
    optimizer = optim.AdamW(params_to_optimize, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs)
    
    # Loss function
    criterion = nn.BCEWithLogitsLoss()
    
    # Training loop
    for epoch in range(args.num_epochs):
        detector.model.train()
        epoch_loss = 0.0
        
        for i, (images, labels, gts, categories, img_paths) in enumerate(dataloader):
            optimizer.zero_grad()
            
            # Process batch
            batch_loss = 0.0
            for j in range(len(images)):
                category = categories[j]
                img_path = img_paths[j]
                
                # Generate prompts
                prompts = prompt_generator.get_prompts_for_inference(
                    args.dataset, 
                    category, 
                    args.prompt_strategy, 
                    args.num_prompts
                )
                
                # Load image
                from PIL import Image
                image = Image.open(img_path).convert("RGB")
                
                # Forward pass (this is a simplified example)
                # In practice, you'd need to implement proper fine-tuning logic
                # based on SAM3's architecture
                detector.predictor.set_image(image)
                
                with torch.set_grad_enabled(True):
                    # This is a placeholder - actual fine-tuning would require
                    # modifying SAM3's forward pass to support training
                    results = detector.predictor.predict(
                        text_queries=prompts,
                        confidence_threshold=0.5,
                        iou_threshold=0.5,
                        return_logits=True
                    )
                
                detector.predictor.reset_image()
                
                # Calculate loss (placeholder)
                # In practice, you'd compute loss between predictions and ground truth
                batch_loss += 0.0  # Placeholder loss
            
            # Backward pass
            batch_loss.backward()
            optimizer.step()
            
            epoch_loss += batch_loss.item()
            
            if (i + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{args.num_epochs}], Step [{i+1}/{len(dataloader)}], Loss: {batch_loss.item():.4f}")
        
        scheduler.step()
        
        print(f"Epoch [{epoch+1}/{args.num_epochs}] completed. Average Loss: {epoch_loss/len(dataloader):.4f}")
        
        # Save checkpoint
        checkpoint_path = os.path.join(args.output_dir, f"checkpoint_epoch_{epoch+1}.pth")
        torch.save({
            'epoch': epoch+1,
            'model_state_dict': detector.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': epoch_loss/len(dataloader),
        }, checkpoint_path)
        
        print(f"Checkpoint saved to: {checkpoint_path}")
    
    print("\n=== Training completed successfully! ===")
    print(f"Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
