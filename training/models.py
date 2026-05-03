# training/models.py
import torch
import torch.nn as nn
import torchvision.models as models
import timm

def get_resnet18(num_classes: int, pretrained: bool = True) -> nn.Module:
    model = models.resnet18(weights='IMAGENET1K_V1' if pretrained else None)
    # Замена последнего полносвязного слоя
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model

def get_vit_tiny(num_classes: int, pretrained: bool = True) -> nn.Module:
    """Маленький ViT (Tiny) из библиотеки timm."""
    model = timm.create_model(
        'vit_tiny_patch16_224',
        pretrained=pretrained,
        num_classes=num_classes
    )
    return model