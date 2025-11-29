"""SAM3 Anomaly Detector"""
import os
import torch
import numpy as np
from PIL import Image
from .sam3.model_builder import build_sam3_image_model as build_sam3_model
from .sam3.model.sam3_image_processor import Sam3Processor


class SAM3AnomalyDetector:
    def __init__(self, model_path, device="cuda", seed=122):
        """
        Initialize SAM3 Anomaly Detector
        
        Args:
            model_path: Path to SAM3 model checkpoint
            device: Device to run model on
            seed: Random seed for reproducibility
        """
        self.device = device
        self.seed = seed
        self.set_seed(seed)
        
        # Load SAM3 model
        self.model = self._load_model(model_path)
        self.processor = Sam3Processor(self.model, device=device)
        
    def set_seed(self, seed):
        """Set random seed for reproducibility"""
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        
    def _load_model(self, model_path):
        """Load SAM3 model from checkpoint"""
        # Check if model_path is a directory, if so, use sam3.pt inside it
        if os.path.isdir(model_path):
            checkpoint_path = os.path.join(model_path, "sam3.pt")
        else:
            checkpoint_path = model_path
        
        model = build_sam3_model(
            checkpoint_path=checkpoint_path,
            device=self.device,
            load_from_HF=False  # Don't load from HF if checkpoint_path is provided
        )
        model.eval()
        return model
    
    def process_image(self, image_path):
        """Process image for SAM3"""
        image = Image.open(image_path).convert("RGB")
        return image
    
    def detect_anomaly(self, image, text_prompts=["anomaly", "defect", "damage", "irregularity"], 
                      confidence_threshold=0.5, iou_threshold=0.5):
        """
        Detect anomalies in image using text prompts
        
        Args:
            image: PIL Image or image path
            text_prompts: List of text prompts for anomaly detection
            confidence_threshold: Confidence threshold for predictions
            iou_threshold: IoU threshold for NMS
            
        Returns:
            dict with masks, boxes, scores, and labels
        """
        if isinstance(image, str):
            image = self.process_image(image)
        
        # Set confidence threshold
        self.processor.set_confidence_threshold(confidence_threshold)
        
        all_masks = []
        all_boxes = []
        all_scores = []
        all_labels = []
        
        # Process each text prompt
        for prompt in text_prompts:
            # Set image and run inference for this prompt
            state = self.processor.set_image(image)
            state = self.processor.set_text_prompt(prompt, state)
            
            # Extract results from state
            if "masks" in state and len(state["masks"]) > 0:
                all_masks.extend(state["masks"].cpu().numpy())
                all_boxes.extend(state["boxes"].cpu().numpy())
                all_scores.extend(state["scores"].cpu().numpy())
                all_labels.extend([prompt] * len(state["scores"]))
        
        results = {
            "masks": all_masks,
            "boxes": all_boxes,
            "scores": all_scores,
            "labels": all_labels
        }
        
        return results
    
    def get_anomaly_mask(self, results, image_size):
        """Generate combined anomaly mask from results"""
        if not results["masks"]:
            return np.zeros(image_size[:2], dtype=np.uint8)
        
        # Combine masks
        combined_mask = np.zeros(image_size[:2], dtype=np.uint8)
        for mask in results["masks"]:
            mask = mask.astype(np.uint8)
            combined_mask = np.maximum(combined_mask, mask)
        
        return combined_mask
    
    def visualize_results(self, image, results, save_path=None):
        """Visualize anomaly detection results"""
        from .sam3.visualization_utils import plot_results
        import matplotlib.pyplot as plt
        
        # Create a figure and axis
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Plot the results
        plot_results(image, results)
        
        # Convert the plot to PIL Image
        fig.canvas.draw()
        visualized = Image.frombytes('RGB', fig.canvas.get_width_height(), fig.canvas.tostring_rgb())
        
        if save_path:
            visualized.save(save_path)
        
        # Close the figure to free memory
        plt.close(fig)
        
        return visualized
