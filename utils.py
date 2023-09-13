import json
import os
import torch
import cv2
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches

from tqdm import tqdm
from skimage.io import imread
from skimage.transform import resize
from torch.nn.utils.rnn import pad_sequence
from torch.nn.functional import pad
from torchvision import ops

def parse_annotations(annotation_path: str, image_dir: str):

    partition: dict[str, list] = {
        "all": [],
        "train": [],
        "validation": [],
        "test": []
    }

    bboxes_all: dict[str, list] = {
        "all": [],
        "train": [],
        "validation": [],
        "test": []
    }

    categories_all: dict[str, list] = {
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

                bbox = [x1, y1, x2, y2]

                category_id = ann["category_id"]

                if len(bbox) > 0:
                    category_labels.append([category_id])
                    bbox_labels.append(bbox)
        
        if len(bbox_labels) == 0:
            category_labels.append([0])
            bbox_labels.append([0,0,0,0])

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

# def display(image_ids, bboxes_all, categories_all, index, resized_img_shape, orig_img_shape):
    
#     img_h, img_w = resized_img_shape
#     orig_img_h, orig_img_w, _ = orig_img_shape

#     w_scale = img_w/orig_img_w
#     h_scale = img_h/orig_img_h

#     colors = {
#         1: (0,255,0), # Swimmer 
#         2: (255,0,0), # Floater
#         3: (51,255,255), # Boat
#         4: (255,255,0), # Swimmer on Boat
#         5: (255, 102, 255), # Floater on Boat
#         6: (255, 153, 51)} # Life Jacket
    
#     image = imread(image_ids[index])
#     # image = resize(image, img_shape)
#     # print(len(categories_all[index - 1]), len(bboxes_all[index - 1]))
#     for cat, bbox in list(zip(categories_all[index - 1], bboxes_all[index - 1])):
#         # print(cat, bbox)
#         # for bbox in bboxes:
#         #     print(bbox)
#         # Resize BBOX as per image here
#         cv2.rectangle(image, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), colors[cat], thickness=2)
    
#     plt.imshow(image)
#     plt.axis("on")
#     plt.show()

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

def custom_collate(data) -> tuple[torch.FloatTensor, torch.Tensor, torch.Tensor]:
    img_inputs: list[torch.Tensor] = [d[0] for d in data]
    categories: list[torch.Tensor] = [d[1] for d in data]
    bboxes: list[torch.Tensor] = [d[2] for d in data]

    cat_index = max([cat.size(dim=1) for cat in categories])
    bboxes_index = max([bb.size(dim=1) for bb in bboxes])

    for i in range(0, len(categories)):
        pad_by = (0, cat_index - categories[i].size(dim=1))
        categories[i] = pad(categories[i], pad_by, "constant", 0)

    for i in range(0, len(bboxes)):
        pad_by = (0, bboxes_index - bboxes[i].size(dim=1))
        bboxes[i] = pad(bboxes[i], pad_by, "constant", 0)

    img_batch = pad_sequence(img_inputs, batch_first=True)
    categories_batch = pad_sequence(categories, batch_first=True)
    bboxes_batch = pad_sequence(bboxes, batch_first=True)

    return torch.FloatTensor(img_batch), categories_batch, bboxes_batch

def gen_anc_centers(out_size):
    out_h, out_w = out_size
    
    anc_pts_x = torch.arange(0, out_w) + 0.5
    anc_pts_y = torch.arange(0, out_h) + 0.5
    
    return anc_pts_x, anc_pts_y

def gen_anc_base(anc_pts_x, anc_pts_y, anc_scales, anc_ratios, out_size):
    n_anc_boxes = len(anc_scales) * len(anc_ratios) * 2
    anc_base = torch.zeros(1, anc_pts_x.size(dim=0) \
                              , anc_pts_y.size(dim=0), n_anc_boxes, 4) # shape - [1, Hmap, Wmap, n_anchor_boxes, 4]
    
    for ix, xc in enumerate(anc_pts_x):
        for jx, yc in enumerate(anc_pts_y):
            anc_boxes = torch.zeros((n_anc_boxes, 4))
            c = 0
            for i, scale in enumerate(anc_scales):
                for j, ratio in enumerate(anc_ratios):
                    w = scale
                    h = scale * ratio
                    
                    xmin = xc - w / 2
                    ymin = yc - h / 2
                    xmax = xc + w / 2
                    ymax = yc + h / 2

                    anc_boxes[c, :] = torch.Tensor([xmin, ymin, xmax, ymax])
                    c += 1

                    w = scale * ratio
                    h = scale

                    xmin = xc - w / 2
                    ymin = yc - h / 2
                    xmax = xc + w / 2
                    ymax = yc + h / 2

                    anc_boxes[c, :] = torch.Tensor([xmin, ymin, xmax, ymax])
                    c += 1

            anc_base[:, ix, jx, :] = ops.clip_boxes_to_image(anc_boxes, size=out_size)
            
    return anc_base

def project_bboxes(bboxes, width_scale_factor, height_scale_factor, mode='a2p'):
    assert mode in ['a2p', 'p2a']
    
    batch_size = bboxes.size(dim=0)
    proj_bboxes = bboxes.clone().reshape(batch_size, -1, 4)
    invalid_bbox_mask = (proj_bboxes == -1) # indicating padded bboxes
    
    if mode == 'a2p':
        # activation map to pixel image
        proj_bboxes[:, :, [0, 2]] *= width_scale_factor
        proj_bboxes[:, :, [1, 3]] *= height_scale_factor
    else:
        # pixel image to activation map
        proj_bboxes[:, :, [0, 2]] /= width_scale_factor
        proj_bboxes[:, :, [1, 3]] /= height_scale_factor
        
    proj_bboxes.masked_fill_(invalid_bbox_mask, -1) # fill padded bboxes back with -1
    proj_bboxes.resize_as_(bboxes)
    
    return proj_bboxes

# Display Functions

def display_batch(batch: tuple):
    img_batch, classes_batch, bboxes_batch = batch

    fig = plt.figure(figsize=(12, 8))
    for i in range(0, len(img_batch)):
        fig.add_subplot(1, len(img_batch), i+1)
        img = draw_bb(img_batch[i], classes_batch[i], bboxes_batch[i])
        plt.imshow(img)
        plt.axis("on")
    plt.show()

def draw_bb(img: torch.Tensor, classes: torch.IntTensor, bboxes: torch.IntTensor) -> torch.Tensor:
    colors = {
        0: (0,0,0), # Dummy color for all the padded values
        1: (0,255,0), # Swimmer 
        2: (255,0,0), # Floater
        3: (51,255,255), # Boat
        4: (255,255,0), # Swimmer on Boat
        5: (255, 102, 255), # Floater on Boat
        6: (255, 153, 51)} # Life Jacket
    
    img = np.array(img.permute(1,2,0)).copy()
    for bb, c in list(zip(bboxes.tolist(), classes.tolist())):
        print(bb, c)
        cv2.rectangle(img, (bb[0], bb[1]), (bb[2], bb[3]), colors[c[0]], thickness=2)

    return img

def display_bbox(bboxes, fig, ax, classes=None, in_format='xyxy', color='y', line_width=2):
    if type(bboxes) == np.ndarray:
        bboxes = torch.from_numpy(bboxes)
    if classes:
        assert len(bboxes) == len(classes)
    # convert boxes to xywh format
    bboxes = ops.box_convert(bboxes, in_fmt=in_format, out_fmt='xywh')
    c = 0
    for box in bboxes:
        x, y, w, h = box.numpy()
        # display bounding box
        rect = patches.Rectangle((x, y), w, h, linewidth=line_width, edgecolor=color, facecolor='none')
        ax.add_patch(rect)
        # display category
        if classes:
            if classes[c] == 'pad':
                continue
            ax.text(x + 5, y + 20, classes[c], bbox=dict(facecolor='yellow', alpha=0.5))
        c += 1
        
    return fig, ax

def display_grid(x_points, y_points, fig, ax, special_point=None):
    # plot grid
    for x in x_points:
        for y in y_points:
            ax.scatter(x, y, color="w", marker='+')
            
    # plot a special point we want to emphasize on the grid
    if special_point:
        x, y = special_point
        ax.scatter(x, y, color="red", marker='+')
        
    return fig, ax

def display_img(img_data, fig, axes):
    for i, img in enumerate(img_data):
        if type(img) == torch.Tensor:
            img = img.permute(1, 2, 0).numpy()
        axes[i].imshow(img)
    
    return fig, axes