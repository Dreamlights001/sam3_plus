"""Evaluation Metrics for Anomaly Detection"""
import numpy as np
from sklearn.metrics import (roc_auc_score, precision_recall_curve, 
                             average_precision_score, f1_score, auc)


def calculate_image_level_auroc(y_true, y_scores):
    """
    Calculate image-level AUROC
    
    Args:
        y_true: List of ground truth labels (0: normal, 1: anomaly)
        y_scores: List of predicted scores
        
    Returns:
        Image-level AUROC score
    """
    return roc_auc_score(y_true, y_scores)


def calculate_pixel_level_auroc(y_true, y_scores):
    """
    Calculate pixel-level AUROC
    
    Args:
        y_true: List of ground truth masks (2D arrays)
        y_scores: List of predicted masks (2D arrays)
        
    Returns:
        Pixel-level AUROC score
    """
    # Flatten all masks
    y_true_flat = np.concatenate([mask.flatten() for mask in y_true])
    y_scores_flat = np.concatenate([score.flatten() for score in y_scores])
    
    return roc_auc_score(y_true_flat, y_scores_flat)


def calculate_image_level_ap(y_true, y_scores):
    """
    Calculate image-level Average Precision
    
    Args:
        y_true: List of ground truth labels (0: normal, 1: anomaly)
        y_scores: List of predicted scores
        
    Returns:
        Image-level AP score
    """
    return average_precision_score(y_true, y_scores)


def calculate_pixel_level_ap(y_true, y_scores):
    """
    Calculate pixel-level Average Precision
    
    Args:
        y_true: List of ground truth masks (2D arrays)
        y_scores: List of predicted masks (2D arrays)
        
    Returns:
        Pixel-level AP score
    """
    # Flatten all masks
    y_true_flat = np.concatenate([mask.flatten() for mask in y_true])
    y_scores_flat = np.concatenate([score.flatten() for score in y_scores])
    
    return average_precision_score(y_true_flat, y_scores_flat)


def calculate_f1_score(y_true, y_pred, threshold=0.5):
    """
    Calculate F1 Score at given threshold
    
    Args:
        y_true: List of ground truth labels/masks
        y_pred: List of predicted scores/masks
        threshold: Threshold for binarization
        
    Returns:
        F1 Score
    """
    # Check if input is image-level or pixel-level
    if isinstance(y_true[0], (int, float)):
        # Image-level
        y_pred_binary = [1 if score >= threshold else 0 for score in y_pred]
        return f1_score(y_true, y_pred_binary)
    else:
        # Pixel-level
        y_true_flat = np.concatenate([mask.flatten() for mask in y_true])
        y_pred_flat = np.concatenate([score.flatten() for score in y_pred])
        y_pred_binary = (y_pred_flat >= threshold).astype(int)
        return f1_score(y_true_flat, y_pred_binary)


def calculate_pixel_level_aupro(y_true, y_scores):
    """
    Calculate pixel-level Area Under Precision-Recall Curve (AUPRO)
    
    Args:
        y_true: List of ground truth masks (2D arrays)
        y_scores: List of predicted masks (2D arrays)
        
    Returns:
        Pixel-level AUPRO score
    """
    # Flatten all masks
    y_true_flat = np.concatenate([mask.flatten() for mask in y_true])
    y_scores_flat = np.concatenate([score.flatten() for score in y_scores])
    
    # Calculate precision-recall curve
    precision, recall, _ = precision_recall_curve(y_true_flat, y_scores_flat)
    
    # Calculate AUPRO
    aupro = auc(recall, precision)
    
    return aupro


def calculate_all_metrics(y_true_images, y_score_images, y_true_pixels, y_score_pixels, threshold=0.5):
    """
    Calculate all evaluation metrics
    
    Args:
        y_true_images: List of image-level ground truth labels
        y_score_images: List of image-level predicted scores
        y_true_pixels: List of pixel-level ground truth masks
        y_score_pixels: List of pixel-level predicted masks
        threshold: Threshold for F1 score
        
    Returns:
        Dictionary of all metrics
    """
    metrics = {
        "image_level_auroc": calculate_image_level_auroc(y_true_images, y_score_images),
        "image_level_ap": calculate_image_level_ap(y_true_images, y_score_images),
        "pixel_level_auroc": calculate_pixel_level_auroc(y_true_pixels, y_score_pixels),
        "pixel_level_ap": calculate_pixel_level_ap(y_true_pixels, y_score_pixels),
        "pixel_level_aupro": calculate_pixel_level_aupro(y_true_pixels, y_score_pixels),
        "image_level_f1": calculate_f1_score(y_true_images, y_score_images, threshold),
        "pixel_level_f1": calculate_f1_score(y_true_pixels, y_score_pixels, threshold)
    }
    
    return metrics


def print_metrics(metrics, dataset_name="Test Set"):
    """Print metrics in a readable format"""
    print(f"\n=== {dataset_name} Metrics ===")
    print(f"Image-level AUROC: {metrics['image_level_auroc']:.4f}")
    print(f"Image-level AP: {metrics['image_level_ap']:.4f}")
    print(f"Image-level F1 Score: {metrics['image_level_f1']:.4f}")
    print(f"Pixel-level AUROC: {metrics['pixel_level_auroc']:.4f}")
    print(f"Pixel-level AP: {metrics['pixel_level_ap']:.4f}")
    print(f"Pixel-level AUPRO: {metrics['pixel_level_aupro']:.4f}")
    print(f"Pixel-level F1 Score: {metrics['pixel_level_f1']:.4f}")
    print("============================")
