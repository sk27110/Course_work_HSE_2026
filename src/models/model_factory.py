import timm
import torchvision.models as models
import torch

class ModelFactory:
    @staticmethod
    def create(name, num_classes, pretrained=True):
        if name.startswith("resnet"):
            model = getattr(models, name)(pretrained=pretrained)
            model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
            return model
        
        else:
            model = timm.create_model(name, pretrained=pretrained, num_classes=num_classes)
            return model
