import torch

# Paths ------------------------------------------------------------
IMG_DIR_PATH = "data/train"
ANNOTATIONS_PATH = "data/annotations/instances_train_swimmer.json"
SAVE_IMAGES_PATH = "custom_data/partition_ids.json"
SAVE_BBOXES_PATH = "custom_data/bboxes_all.json"
SAVE_CLASSES_PATH = "custom_data/categories_all.json"
MODEL_SAVE_PATH = "model.pt"

# Dimensions -------------------------------------------------------
OUT_C = 2048
OUT_H = 17
OUT_W = 30
RESIZE_FACTOR = 4

# Training Hyperparameters -----------------------------------------
N_CLASSES = 6
LEARNING_RATE = 1e-3
ROI_SIZE = (2, 2)
N_EPOCHS = 200
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")