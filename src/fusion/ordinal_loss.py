import torch
import torch.nn as nn
import torch.nn.functional as F

class OrdinalHeckmattLoss(nn.Module):
    """
    Ordinal classification loss for Heckmatt grading.
    Penalizes classification errors proportionally to their ordinal distance
    from the ground-truth grade (e.g. misclassifying Grade 1 as Grade 4 is
    penalized 3x more heavily than misclassifying as Grade 2).
    """
    def __init__(self, num_classes=4, ordinal_weight=0.5, class_weights=None):
        super().__init__()
        self.num_classes = num_classes
        self.ordinal_weight = ordinal_weight
        self.ce_loss = nn.CrossEntropyLoss(weight=class_weights)
        
        # Matrix of ordinal distances: distance_matrix[i, j] = |i - j|
        dist = torch.zeros((num_classes, num_classes), dtype=torch.float32)
        for i in range(num_classes):
            for j in range(num_classes):
                dist[i, j] = abs(i - j)
        self.register_buffer('distance_matrix', dist)

    def forward(self, logits, targets):
        """
        logits: [B, num_classes]
        targets: [B] (0-indexed: 0=Grade 1, 1=Grade 2, 2=Grade 3, 3=Grade 4)
        """
        ce = self.ce_loss(logits, targets)
        
        probs = F.softmax(logits, dim=1) # [B, num_classes]
        # Gather ordinal penalties for each sample
        target_distances = self.distance_matrix[targets] # [B, num_classes]
        ordinal_penalty = (probs * target_distances).sum(dim=1).mean()
        
        return ce + self.ordinal_weight * ordinal_penalty


class ClassWeightedFocalLoss(nn.Module):
    """
    Focal Loss with class weights for handling rare Heckmatt grades (e.g. Grade 4).
    """
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha # Tensor of weights per class
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1.0 - pt) ** self.gamma) * ce_loss
        
        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss
            
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss

