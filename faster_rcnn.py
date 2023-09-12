import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import sys

from skimage.io import imread
from skimage.transform import resize
from torch.nn import Module

class FasterRCNN(Module):
    def __init__(self, device):
        super(FasterRCNN, self).__init__()

        self.device = device

        # Load the resnet model with pretrained weights
        resnet_model = torchvision.models.resnet50()

        # Can mess around with the length of this for accuracy
        required_layers = list(resnet_model.children())[:8]

        self.backbone = nn.Sequential(*required_layers)

        # Unfreeze all parameters, check if already unfrozen
        for param in self.backbone.named_parameters():
            param[1].requires_grad = True
    
    def forward(self, x: torch.Tensor):
        # Predict output on an image to get the new dimensions of the compressed feature map

        x = self.backbone(x)

        return x
