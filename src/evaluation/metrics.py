import numpy as np
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score,
    recall_score, f1_score, cohen_kappa_score, confusion_matrix,
    roc_auc_score
)

def compute_classification_metrics(y_true, y_pred, y_probs=None, num_classes=4):
    """
    Computes rigorous clinical evaluation metrics for Heckmatt grading:
    - Standard Accuracy
    - Balanced Accuracy (accounts for rare Grade 4)
    - Macro Precision, Recall, F1
    - Quadratic Weighted Cohen's Kappa (penalizes ordinal distance quadratically)
    - Ordinal Mean Absolute Error (MAE)
    - Multi-class ROC-AUC (One-vs-Rest)
    - Confusion Matrix
    """
    y_true = np.array(y_true, dtype=int)
    y_pred = np.array(y_pred, dtype=int)

    acc = accuracy_score(y_true, y_pred)
    bacc = balanced_accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='macro', zero_division=0)
    rec = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    
    # Quadratic Weighted Kappa is the gold-standard metric for medical ordinal grading
    qwk = cohen_kappa_score(y_true, y_pred, weights='quadratic')
    mae = float(np.mean(np.abs(y_true - y_pred)))
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))

    roc_auc = None
    if y_probs is not None:
        try:
            # Handle potential missing classes in y_true for AUC
            roc_auc = roc_auc_score(y_true, y_probs, multi_class='ovr', average='macro')
        except Exception:
            roc_auc = None

    return {
        'accuracy': float(acc),
        'balanced_accuracy': float(bacc),
        'macro_precision': float(prec),
        'macro_recall': float(rec),
        'macro_f1': float(f1),
        'quadratic_weighted_kappa': float(qwk),
        'ordinal_mae': float(mae),
        'roc_auc': float(roc_auc) if roc_auc is not None else 0.0,
        'confusion_matrix': cm.tolist()
    }

