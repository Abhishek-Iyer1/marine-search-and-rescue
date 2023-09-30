import torch
import torchvision
from torchvision import ops
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn

from utils import *
from torchvision.ops import generalized_box_iou_loss

# -------------------- Models -----------------------

class FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        model = torchvision.models.resnet50(pretrained=True)
        req_layers = list(model.children())[:7]
        self.backbone = nn.Sequential(*req_layers)
        for param in self.backbone.named_parameters():
            param[1].requires_grad = True
        
    def forward(self, img_data):
        return self.backbone(img_data)