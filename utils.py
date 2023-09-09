import json
import os
import torch
import cv2
import matplotlib.pyplot as plt

from tqdm import tqdm
from skimage.io import imread
from skimage.transform import resize

def parse_annotations(annotation_path: str, image_dir: str):

    partition = {
        "all": [],
        "train": [],
        "validation": [],
        "test": []
    }

    labels = {
        "all": [],
        "train": [],
        "validation": [],
        "test": []
    }

    # Opening annotations JSON file and loading data
    with open(annotation_path) as json_file:
        annotations: dict = json.load(json_file)

    print("Checking for annotations, collecting file paths...")
    sorted_img_dir = sorted(os.listdir(image_dir), key= lambda y: int(y.split(".")[0]))
    sorted_annotations = sorted(annotations["annotations"], key= lambda y: int(y["image_id"]))

    for file in tqdm(sorted_img_dir):
        partition["all"].append(os.path.join(image_dir, file))
        img_number = file.split(".")[0]

        temp_labels = {}

        for ann in sorted_annotations:
            if int(ann["image_id"]) == int(img_number):
                x1, y1, w, h = ann["bbox"]
                x2 = x1 + w
                y2 = y1 + h

                bbox = torch.Tensor([x1, y1, x2, y2])

                category_id = ann["category_id"]

                groundtruth_bboxes.append(bbox.tolist())
                groundtruth_categories.append(category_id)

        bboxes_all.append(torch.Tensor(groundtruth_bboxes))
        classes_all.append(groundtruth_categories)

    return bboxes_all, classes_all, img_paths