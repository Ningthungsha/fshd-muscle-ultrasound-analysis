import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-5):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        # pred is after sigmoid or logits
        if pred.min() < 0 or pred.max() > 1:
            pred = torch.sigmoid(pred)
        
        pred = pred.view(-1)
        target = target.view(-1)
        
        intersection = (pred * target).sum()
        dice = (2.0 * intersection + self.smooth) / (pred.sum() + target.sum() + self.smooth)
        return 1.0 - dice


class DiceBCELoss(nn.Module):
    """Hybrid Dice + BCE loss for balanced region overlap and boundary precision"""
    def __init__(self, dice_weight=0.5, bce_weight=0.5, smooth=1e-5):
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.dice_loss = DiceLoss(smooth)
        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(self, logits, target):
        loss_dice = self.dice_loss(logits, target)
        loss_bce = self.bce_loss(logits, target)
        return self.dice_weight * loss_dice + self.bce_weight * loss_bce


class FocalDiceLoss(nn.Module):
    """Focal + Dice Loss for handling severe background imbalance in ultrasound scans"""
    def __init__(self, gamma=2.0, alpha=0.25, smooth=1e-5):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.dice_loss = DiceLoss(smooth)

    def forward(self, logits, target):
        prob = torch.sigmoid(logits)
        bce = F.binary_cross_entropy_with_logits(logits, target, reduction='none')
        p_t = prob * target + (1 - prob) * (1 - target)
        focal_loss = (self.alpha * (1 - p_t) ** self.gamma * bce).mean()
        dice_l = self.dice_loss(prob, target)
        return focal_loss + dice_l

