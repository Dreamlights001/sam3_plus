"""Test script for SAM3 Anomaly Detection"""
import sys
import os

# Add the project root directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import numpy as np
# 确保numpy正确导入
print(f"numpy version: {np.__version__}")
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
    
    # Create dataset and dataloader with batch_size=1 and custom collate to handle PIL images
    print(f"Creating dataset: {dataset_name} with path: {dataset_path} and category: {category}")
    try:
        dataset = get_dataset(dataset_name, dataset_path, category)
        print(f"Dataset created with {len(dataset)} samples")
        # Use batch_size=1 and disable batching with custom collate_fn
        print("Creating dataloader...")
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=lambda x: x[0])
        print(f"Dataloader created successfully")
        
        # Check if dataloader is empty
        try:
            first_item = next(iter(dataloader))
            print(f"Dataloader contains data, first item type: {type(first_item)}")
            # Reset dataloader
            dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=lambda x: x[0])
        except StopIteration:
            print(f"WARNING: Dataloader for category {category} is empty!")
    except Exception as e:
        print(f"Error creating dataset/dataloader: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    # Initialize lists for metrics
    y_true_images = []
    y_score_images = []
    y_true_pixels = []
    y_score_pixels = []
    
    # Iterate over test set
    print(f"Starting iteration over test set with {len(dataset)} samples...")
    for i, data in enumerate(dataloader):
        print(f"\nData received from dataloader at index {i}: {type(data)}")
        # Try to unpack the data
        try:
            image, label, gt, category_name, img_path = data
            print(f"Successfully unpacked data")
            print(f"Image type: {type(image)}, Label: {label}, GT type: {type(gt)}")
            print(f"Category name: {category_name}, Image path: {img_path}")
            print(f"Processing image {i + 1}/{len(dataset)}: {os.path.basename(img_path)}")
        except Exception as e:
            print(f"Error unpacking data: {e}")
            print(f"Data structure: {data}")
            continue
        
        # Generate prompts
        prompts = prompt_generator.get_prompts_for_inference(
            dataset_name, 
            category, 
            args.prompt_strategy, 
            args.num_prompts
        )
        
        # Load image (using the path from the dataset)
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
        
        # Convert gt to numpy array if it's a PIL Image
        if isinstance(gt, Image.Image):
            gt = np.array(gt)
        elif gt is None:
            gt = np.zeros((image.size[1], image.size[0]), dtype=np.uint8)
        
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
    """Main function to run the SAM3 anomaly detection test."""
    print("=== Starting test script ===")
    
    # 确保numpy在函数内可用
    import numpy as np
    import random
    import torch
    
    args = parse_args()
    print(f"Parsed arguments: {args}")
    
    # Set seed for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    print("Seed set successfully")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Created output directory: {args.output_dir}")
    
    # Initialize detector and prompt generator
    print(f"Initializing detector with model path: {args.model_path} on device: {args.device}")
    try:
        detector = SAM3AnomalyDetector(args.model_path, args.device, args.seed)
        print("Detector initialized successfully")
    except Exception as e:
        print(f"Error initializing detector: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    # 检查检测器是否正常初始化
    print(f"Detector type: {type(detector)}")
    print("Detector attributes:")
    for attr in dir(detector):
        if not attr.startswith('_'):
            print(f"  - {attr}")
    
    # 初始化提示词生成器
    print("Initializing prompt generator...")
    try:
        prompt_generator = AnomalyPromptGenerator()
        print("Prompt generator initialized successfully")
    except Exception as e:
        print(f"Error initializing prompt generator: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    print("Passed detector and prompt generator initialization")
    
    
    # Get all categories if none specified
    print("Getting categories to test...")
    try:
        if args.category:
            categories = [args.category]
            print(f"Testing specific category: {args.category}")
        else:
            print("=== Testing dataset path access ===")
            if not os.path.exists(args.dataset_path):
                print(f"Dataset path does not exist: {args.dataset_path}")
                # Try to list the directory to see what's available
                print("Contents of parent directory:")
                parent_dir = os.path.dirname(args.dataset_path)
                if os.path.exists(parent_dir):
                    for item in os.listdir(parent_dir):
                        print(f"  - {item}")
                raise FileNotFoundError(f"Dataset path does not exist: {args.dataset_path}")
            
            print(f"Dataset path exists: {args.dataset_path}")
            print(f"Contents of dataset path:")
            for item in os.listdir(args.dataset_path):
                print(f"  - {item}")
            print("Passed dataset path access test")
            
            if args.dataset == "mvtec":
                categories = MVTecDataset.categories
                print("Retrieved MVTecDataset categories")
            elif args.dataset == "visa":
                categories = VisaDataset.categories
                print("Retrieved VisaDataset categories")
            else:
                raise ValueError(f"Unsupported dataset: {args.dataset}")
        print(f"Categories to test: {categories}")
        if not categories:
            print("WARNING: No categories found!")
            # Try to list the dataset path to see what's available
            print("Contents of dataset path:")
            if os.path.exists(args.dataset_path):
                for item in os.listdir(args.dataset_path):
                    print(f"  - {item}")
    except Exception as e:
        print(f"Error getting categories: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    # 创建一个简单的测试脚本来验证检测器功能
    print("=== Creating simple test to verify detector functionality ===")
    try:
        # 创建一个随机图像进行测试
        from PIL import Image
        import numpy as np
        print("Creating random test image...")
        test_image = Image.fromarray((np.random.rand(224, 224, 3) * 255).astype(np.uint8))
        print(f"Test image created successfully, size: {test_image.size}")
        test_prompts = ["anomaly", "defect", "damage"]
        print(f"Test prompts: {test_prompts}")
        
        print("=== Starting detect_anomaly function call ===")
        results = detector.detect_anomaly(test_image, test_prompts)
        print("=== detect_anomaly function returned successfully ===")
        print(f"Test successful! Results type: {type(results)}")
        if isinstance(results, dict):
            print(f"Results keys: {results.keys()}")
            # 打印每个键的值类型
            for key, value in results.items():
                print(f"  - {key}: {type(value)}")
                # 如果是列表，打印长度
                if isinstance(value, list):
                    print(f"    Length: {len(value)}")
                # 如果是张量，打印形状
                elif hasattr(value, 'shape'):
                    print(f"    Shape: {value.shape}")
        else:
            print("Results are not a dictionary")
    except Exception as e:
        print(f"Error testing detector functionality: {e}")
        import traceback
        traceback.print_exc()
        # 继续执行，不中断程序
    
    print("Continuing with main test process")
    
    # Test each category
    all_metrics = {}
    print(f"Starting testing on {len(categories)} categories...")
    if not categories:
        print("No categories to process. Exiting...")
        return
    
    # 为了调试，先只测试第一个类别
    print("=== DEBUG MODE: Testing only the first category ===")
    categories = categories[:1]  # 只测试第一个类别
    print(f"Categories after debug filtering: {categories}")
    
    for i, category in enumerate(categories):
        print(f"\n[{i+1}/{len(categories)}] Processing category: {category}")
        try:
            print(f"=== Calling test_category function for category: {category} ===")
            category_metrics = test_category(
                category, 
                args.dataset, 
                args.dataset_path, 
                detector, 
                prompt_generator, 
                args
            )
            print(f"=== test_category function returned successfully ===")
            all_metrics[category] = category_metrics
            print(f"Successfully processed category: {category}")
        except Exception as e:
            print(f"Error processing category {category}: {e}")
            import traceback
            traceback.print_exc()
            # Continue with next category instead of failing completely
    
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
