import os
import torch
import numpy as np

from skimage.io import imread

# Paths ------------------------------------------------------------
IMG_DIR_PATH = "data/train"
ANNOTATIONS_PATH = "data/annotations/instances_train_swimmer.json"
SAVE_IMAGES_PATH = "custom_data/partition_ids.json"
SAVE_BBOXES_PATH = "custom_data/bboxes_all.json"
SAVE_CLASSES_PATH = "custom_data/categories_all.json"
MODEL_SAVE_PATH = "model_micro.pt"

# Dimensions -------------------------------------------------------

OUT_C = 1024
OUT_H = 14  # Need to change if RESIZE_FACTOR is changed
OUT_W = 24  # Need to change if RESIZE_FACTOR is changed
RESIZE_FACTOR = 10

test_image = np.array(imread(os.path.join(IMG_DIR_PATH, '0.jpg')))
ORIG_IMAGE_SHAPE = test_image.shape
RESIZED_IMAGE_SHAPE = (round(ORIG_IMAGE_SHAPE[0] / RESIZE_FACTOR), round(ORIG_IMAGE_SHAPE[1] / RESIZE_FACTOR))
WIDTH_SCALE_FACTOR = RESIZED_IMAGE_SHAPE[1] // OUT_W
HEIGHT_SCALE_FACTOR = RESIZED_IMAGE_SHAPE[0] // OUT_H

print(WIDTH_SCALE_FACTOR, HEIGHT_SCALE_FACTOR)

# Training Hyperparameters -----------------------------------------
N_CLASSES = 6
LEARNING_RATE = 1e-3
ROI_SIZE = (2, 2)
N_EPOCHS = 100
CONF_THRESH = 0.85
NMS_THRESH = 0.1
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")