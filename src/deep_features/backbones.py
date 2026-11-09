import torch
import torch.nn as nn
import torchvision.models as models

class UltrasoundDeepFeatureExtractor(nn.Module):
    """
    Deep feature representation learner for musculoskeletal ultrasound.
    Extracts high-dimensional spatial and textural representations from muscle ROIs.
    """
    def __init__(self, latent_dim=128, pretrained=False):
        super().__init__()
        # Use resnet18 backbone adapted for 1-channel grayscale ultrasound
        base_model = models.resnet18(weights=None)
        
        # Modify first conv layer to accept 1 channel (grayscale ultrasound)
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = base_model.bn1
        self.relu = base_model.relu
        self.maxpool = base_model.maxpool

        self.layer1 = base_model.layer1
        self.layer2 = base_model.layer2
        self.layer3 = base_model.layer3
        self.layer4 = base_model.layer4

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Projection head to latent embedding
        self.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, latent_dim)
        )

    def extract_conv_features(self, x):
        """Extract spatial feature map from final conv layer for Grad-CAM"""
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

    def forward(self, x):
        conv_feats = self.extract_conv_features(x)
        pooled = self.avgpool(conv_feats)
        flattened = torch.flatten(pooled, 1)
        latent = self.fc(flattened)
        return latent

