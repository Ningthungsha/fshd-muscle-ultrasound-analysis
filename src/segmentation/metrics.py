import numpy as np
import torch

def compute_segmentation_metrics(pred_mask, gt_mask, threshold=0.5):
    """
    Computes comprehensive clinical segmentation metrics:
    - Dice Similarity Coefficient (DSC)
    - Intersection over Union (IoU / Jaccard Index)
    - Precision (Positive Predictive Value)
    - Recall / Sensitivity (True Positive Rate)
    - Specificity (True Negative Rate)
    - Accuracy
    """
    if isinstance(pred_mask, torch.Tensor):
        pred_mask = pred_mask.detach().cpu().numpy()
    if isinstance(gt_mask, torch.Tensor):
        gt_mask = gt_mask.detach().cpu().numpy()

    # Binarize
    p = (pred_mask >= threshold).astype(bool).flatten()
    g = (gt_mask > 0).astype(bool).flatten()

    tp = np.logical_and(p, g).sum()
    fp = np.logical_and(p, ~g).sum()
    fn = np.logical_and(~p, g).sum()
    tn = np.logical_and(~p, ~g).sum()

    dice = (2.0 * tp) / (2.0 * tp + fp + fn + 1e-7)
    iou = tp / (tp + fp + fn + 1e-7)
    precision = tp / (tp + fp + 1e-7)
    recall = tp / (tp + fn + 1e-7)
    specificity = tn / (tn + fp + 1e-7)
    accuracy = (tp + tn) / (len(p) + 1e-7)

    return {
        'dice': float(dice),
        'iou': float(iou),
        'precision': float(precision),
        'recall': float(recall),
        'specificity': float(specificity),
        'accuracy': float(accuracy),
        'tp': int(tp),
        'fp': int(fp),
        'fn': int(fn),
        'tn': int(tn)
    }

