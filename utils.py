import json
import os
import torch

from tqdm import tqdm 

def parse_annotations(annotation_path: str, image_dir: str, resized_img_size: tuple[int, int], original_img_shape: tuple[int, int, int]):
    img_h, img_w = resized_img_size
    orig_img_h, orig_img_w, _ = original_img_shape

    w_scale = img_w/orig_img_w
    h_scale = img_h/orig_img_h

    img_paths = []
    bboxes_all = []
    classes_all = []

    # Opening annotations JSON file and loading data
    with open(annotation_path) as json_file:
        annotations: dict = json.load(json_file)

    print("Checking for annotations, collecting file paths...")
    for file in tqdm(os.listdir(image_dir)):
        image_path = os.path.join(image_dir, file)
        img_paths.append(image_path)
        img_number = file.split(".")[0]

        groundtruth_bboxes = []
        groundtruth_categories = []

        for ann in annotations["annotations"]:
            if ann["image_id"] == img_number:
                x1, y1, w, h = ann["bbox"]
                x2 = x1 + w
                y2 = y1 + h

                bbox = torch.Tensor([x1, y1, x2, y2])
                bbox[[0, 2]] *= w_scale
                bbox[[1, 3]] *= h_scale

                category_id = ann["category_id"]

                groundtruth_bboxes.append(bbox.tolist())
                groundtruth_categories.append(category_id)

        bboxes_all.append(torch.Tensor(groundtruth_bboxes))
        classes_all.append(groundtruth_categories)

    return bboxes_all, classes_all, img_paths