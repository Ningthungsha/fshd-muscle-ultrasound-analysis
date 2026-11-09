import torch
import torch.nn as nn
import torch.nn.functional as F

class GatedAttentionFusionModel(nn.Module):
    """
    Cross-modal gated fusion network combining deep latent embeddings
    with radiomic features via a learned Sigmoid gating mechanism.
    """
    def __init__(self, deep_dim=128, radiomic_dim=48, hidden_dim=64, num_classes=4):
        super().__init__()
        self.deep_dim = deep_dim
        self.radiomic_dim = radiomic_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes

        # Radiomics projection
        self.rad_proj = nn.Sequential(
            nn.Linear(radiomic_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2)
        )

        # Deep representation projection
        self.deep_proj = nn.Sequential(
            nn.Linear(deep_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2)
        )

        # Cross-modal gating network
        self.gate_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid()
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

        # Ordinal regression head (continuous severity)
        self.ordinal_regressor = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1)
        )

    def forward(self, deep_features, radiomic_features):
        """
        deep_features: [B, deep_dim]
        radiomic_features: [B, radiomic_dim]
        """
        h_rad = self.rad_proj(radiomic_features)
        h_deep = self.deep_proj(deep_features)

        # Compute gating weights
        concat = torch.cat([h_deep, h_rad], dim=1)
        g = self.gate_net(concat) # [B, hidden_dim] in [0, 1]

        # Dynamically modulated fusion: gate selects informative balance
        fused = g * h_deep + (1.0 - g) * h_rad

        logits = self.classifier(fused)
        ordinal_score = self.ordinal_regressor(fused)

        return {
            'logits': logits,
            'probabilities': F.softmax(logits, dim=1),
            'ordinal_score': ordinal_score.squeeze(-1) + 1.0, # shift to [1, 4] range
            'gate_weights': g,
            'fused_representation': fused
        }


class EarlyFusionModel(nn.Module):
    """
    Early Fusion Baseline: Direct concatenation of deep representations and radiomics.
    """
    def __init__(self, deep_dim=128, radiomic_dim=48, hidden_dim=96, num_classes=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(deep_dim + radiomic_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 48),
            nn.BatchNorm1d(48),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(48, num_classes)
        )

    def forward(self, deep_features, radiomic_features):
        x = torch.cat([deep_features, radiomic_features], dim=1)
        logits = self.net(x)
        return {
            'logits': logits,
            'probabilities': F.softmax(logits, dim=1)
        }


class DeepOnlyModel(nn.Module):
    """Deep-features-only baseline"""
    def __init__(self, deep_dim=128, hidden_dim=64, num_classes=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(deep_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, deep_features, radiomic_features=None):
        logits = self.net(deep_features)
        return {
            'logits': logits,
            'probabilities': F.softmax(logits, dim=1)
        }


class RadiomicsOnlyModel(nn.Module):
    """Radiomic-features-only baseline"""
    def __init__(self, radiomic_dim=48, hidden_dim=48, num_classes=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(radiomic_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, deep_features=None, radiomic_features=None):
        logits = self.net(radiomic_features)
        return {
            'logits': logits,
            'probabilities': F.softmax(logits, dim=1)
        }


class LateFusionEnsemble(nn.Module):
    """Late Fusion (Ensemble) Model blending deep and radiomic predictions"""
    def __init__(self, deep_model, radiomic_model, weight_deep=0.5):
        super().__init__()
        self.deep_model = deep_model
        self.radiomic_model = radiomic_model
        self.weight_deep = nn.Parameter(torch.tensor([weight_deep], dtype=torch.float32))

    def forward(self, deep_features, radiomic_features):
        out_deep = self.deep_model(deep_features)
        out_rad = self.radiomic_model(None, radiomic_features)

        w = torch.sigmoid(self.weight_deep)
        probs = w * out_deep['probabilities'] + (1.0 - w) * out_rad['probabilities']
        logits = torch.log(probs + 1e-7)

        return {
            'logits': logits,
            'probabilities': probs,
            'deep_weight': w.item()
        }

