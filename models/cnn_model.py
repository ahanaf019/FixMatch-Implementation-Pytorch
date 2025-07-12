import torch
import torch.nn as nn
import torchvision

class CNNModel(nn.Module):
    def __init__(self, num_classes, dropout_rate=0.1):
        super().__init__()
        self.backbone = torchvision.models.resnet18(
            weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1
        )
        self.backbone.fc = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout_rate, inplace=True),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)
