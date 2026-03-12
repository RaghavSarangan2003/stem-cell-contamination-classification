import torch.nn as nn
from torchvision import models
from torchvision.models import EfficientNet_B0_Weights

def build_efficientnet_b0(num_classes=3, freeze_backbone=True):
    """
    Build EfficientNet-B0 model for 3-class classification
    """

    model = models.efficientnet_b0(
        weights=EfficientNet_B0_Weights.DEFAULT
    )

    # Freeze backbone
    if freeze_backbone:
        for param in model.features.parameters():
            param.requires_grad = False

    # Replace classifier
    in_features = model.classifier[1].in_features

    model.classifier = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(in_features, num_classes)
    )

    return model

