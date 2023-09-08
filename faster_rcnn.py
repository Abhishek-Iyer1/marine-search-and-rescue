import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import sys

from skimage.io import imread
from skimage.transform import resize


def load_single_image():
    data_path = "data/train"

    # Empty lists and dicts to hold training data and labels
    f_names: dict = {}
    
    # Get all the file names in train dir and seperate extensions and discard them
    for file in os.listdir(path=data_path):
        f_names[int(file.split(".")[0])] = file

    # Sort f_names in numerical order to maintain sequencing
    mapping = sorted(list(f_names.keys()))

    # Display a given image with its annotations
    index = int(sys.argv[1]) - 1
    
    # If input is within the valid range, proceed
    if (index >= 0) and (index <= len(mapping)):
        
        # Load image as an np.array and normalize
        image = imread(os.path.join(data_path, f_names[mapping[index]]))
        image_resized = resize(image, (640, 480))

        # Re-arranging tensor so channels are the first dimension
        image_tensor = torch.from_numpy(image_resized).permute(2,0,1).to(dtype=torch.float32)

        return image_tensor

    else:
        return ValueError (f"Index provided is out of bounds for list dataset with size {len(f_names.keys())}")

def cnn_backbone(image: torch.Tensor):

    # Load the resnet model with pretrained weights
    resnet_model = torchvision.models.resnet50()

    # Can mess around with the length of this for accuracy
    required_layers = list(resnet_model.children())[:8]

    backbone = nn.Sequential(*required_layers)

    # Unfreeze all parameters, check if already unfrozen
    for param in backbone.named_parameters():
        param[1].requires_grad = True

    # Predict output on an image to get the new dimensions of the compressed feature map
    out: np.ndarray = backbone(image)

    print(image.size, out.size)

if __name__ == '__main__':

    image = load_single_image()
    cnn_backbone(image)