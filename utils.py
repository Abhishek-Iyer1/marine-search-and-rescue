import json
import os
import torch
import cv2
import matplotlib.pyplot as plt

from tqdm import tqdm
from skimage.io import imread
from skimage.transform import resize

def parse_annotations(annotation_path: str, image_dir: str):

    partition: dict = {
        "all": [],
        "train": [],
        "validation": [],
        "test": []
    }

    bboxes_all: dict = {
        "all": [],
        "train": [],
        "validation": [],
        "test": []
    }

    categories_all: dict = {
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

        category_labels = []
        bbox_labels = []

        for ann in sorted_annotations:
            if int(ann["image_id"]) == int(img_number):
                x1, y1, w, h = ann["bbox"]
                x2 = x1 + w
                y2 = y1 + h

                bbox = torch.Tensor([x1, y1, x2, y2])

                category_id = ann["category_id"]

                category_labels.append(int(category_id))
                bbox_labels.append(bbox.tolist())

        bboxes_all["all"].append(bbox_labels)
        categories_all["all"].append(category_labels)

    return partition, bboxes_all, categories_all

def split_ids(partition: dict[str, list], bboxes_all: dict[str, list], categories_all: dict[str, list], split: tuple[float, float, float]) -> dict[str, list]:

    train_index = (0, round(split[0] * len(partition["all"])))
    validation_index = (train_index[1], train_index[1] + round(split[1] * len(partition["all"])))
    test_index = (validation_index[1], len(partition["all"]))

    # Splitting all the file paths by train:val:test split
    partition["train"] = partition["all"][train_index[0] : train_index[1]]
    partition["validation"] = partition["all"][validation_index[0] : validation_index[1]]
    partition["test"] = partition["all"][test_index[0]: test_index[1]]

    bboxes_all["train"] = bboxes_all["all"][train_index[0] : train_index[1]]
    bboxes_all["validation"] = bboxes_all["all"][validation_index[0] : validation_index[1]]
    bboxes_all["test"] = bboxes_all["all"][test_index[0]: test_index[1]]

    categories_all["train"] = categories_all["all"][train_index[0] : train_index[1]]
    categories_all["validation"] = categories_all["all"][validation_index[0] : validation_index[1]]
    categories_all["test"] = categories_all["all"][test_index[0]: test_index[1]]

    return partition, bboxes_all, categories_all

def display(image_ids, bboxes_all, categories_all, index, resized_img_shape, orig_img_shape):
    
    img_h, img_w = resized_img_shape
    orig_img_h, orig_img_w, _ = orig_img_shape

    w_scale = img_w/orig_img_w
    h_scale = img_h/orig_img_h

    colors = {
        1: (0,255,0), # Swimmer 
        2: (255,0,0), # Floater
        3: (51,255,255), # Boat
        4: (255,255,0), # Swimmer on Boat
        5: (255, 102, 255), # Floater on Boat
        6: (255, 153, 51)} # Life Jacket
    
    image = imread(image_ids[index])
    # image = resize(image, img_shape)
    print(len(categories_all[index - 1]), len(bboxes_all[index - 1]))
    for cat, bbox in list(zip(categories_all[index - 1], bboxes_all[index - 1])):
        print(cat, bbox)
        # for bbox in bboxes:
        #     print(bbox)
        # Resize BBOX as per image here
        cv2.rectangle(image, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), colors[cat], thickness=2)
    
    plt.imshow(image)
    plt.axis("on")
    plt.show()

def load_data(save_partition_path, save_bboxes_path, save_categories_path, annotations_path, img_dir_path, resized_img_shape, orig_img_shape, split = (0.7, 0.1, 0.2), force_overwrite=False):

    if (force_overwrite) or (not os.path.exists(save_partition_path)) or (not os.path.exists(save_bboxes_path)) or (not os.path.exists(save_categories_path)):
        print(f"Not found save file path or argument force_overwrite = True. Generating data again...")
        
        partition_ids, bboxes_all, categories_all = parse_annotations(annotation_path=annotations_path, image_dir=img_dir_path)
        
        partition_ids, bboxes_all, categories_all = split_ids(partition_ids, bboxes_all, categories_all, split)

        with open(save_partition_path, "w") as json_partition_obj:
            json.dump(partition_ids, json_partition_obj)

        with open(save_bboxes_path, "w") as json_bbox_obj:
            json.dump(bboxes_all, json_bbox_obj)

        with open(save_categories_path, "w") as json_categories_obj:
            json.dump(categories_all, json_categories_obj)

    else:
        print(f"Found save file path and argument force_overwrite = False. Fetching saved data...")
        
        with open(save_partition_path, "r") as json_partition_obj:
            partition_ids = json.load(json_partition_obj)
        
        with open(save_bboxes_path, "r") as json_bboxes_obj:
            bboxes_all = json.load(json_bboxes_obj)

        with open(save_categories_path, "r") as json_categories_obj:
            categories_all = json.load(json_categories_obj)

    print(f"Length of Data: {len(partition_ids['all'])}, Length of Train: {len(partition_ids['train'])}, Length of Val: {len(partition_ids['validation'])}, Length of Test: {len(partition_ids['test'])}")
    print(f"Length of Bboxes: {len(bboxes_all['all'])}, Length of Train: {len(bboxes_all['train'])}, Length of Val: {len(bboxes_all['validation'])}, Length of Test: {len(bboxes_all['test'])}")
    print(f"Length of Categories: {len(categories_all['all'])}, Length of Train: {len(categories_all['train'])}, Length of Val: {len(categories_all['validation'])}, Length of Test: {len(categories_all['test'])}")

    return partition_ids, bboxes_all, categories_all