import json
import os
import torch
import cv2
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches
import config
import math
import torch.nn.functional as F

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
        
        assert len(bbox_labels) == len(category_labels)
        
        if len(bbox_labels) > 0:
            partition["all"].append(os.path.join(image_dir, file))
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
    # print(f"Length of Bboxes: {len(bboxes_all['all'])}, Length of Train: {len(bboxes_all['train'])}, Length of Val: {len(bboxes_all['validation'])}, Length of Test: {len(bboxes_all['test'])}")
    # print(f"Length of Categories: {len(categories_all['all'])}, Length of Train: {len(categories_all['train'])}, Length of Val: {len(categories_all['validation'])}, Length of Test: {len(categories_all['test'])}")

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

def generate_proposals(anchors, offsets):
   
    # change format of the anchor boxes from 'xyxy' to 'cxcywh'
    anchors = ops.box_convert(anchors, in_fmt='xyxy', out_fmt='cxcywh')
    # print(f"Anchors: {anchors.shape}, Offsets: {offsets.shape}")
    # apply offsets to anchors to create proposals
    proposals_ = torch.zeros_like(anchors)
    proposals_[:,0] = anchors[:,0] + offsets[:,0]*anchors[:,2]
    proposals_[:,1] = anchors[:,1] + offsets[:,1]*anchors[:,3]
    proposals_[:,2] = anchors[:,2] * torch.exp(offsets[:,2])
    proposals_[:,3] = anchors[:,3] * torch.exp(offsets[:,3])

    # change format of proposals back from 'cxcywh' to 'xyxy'
    proposals = ops.box_convert(proposals_, in_fmt='cxcywh', out_fmt='xyxy')

    return proposals

def project_bboxes(bboxes: torch.Tensor, width_scale_factor, height_scale_factor, mode='a2p') -> torch.Tensor:
    bboxes = bboxes.to(config.DEVICE)
    assert mode in ['a2p', 'p2a']
    
    batch_size = bboxes.size(dim=0)
    # print(bboxes.shape)
    if batch_size > 0:
        proj_bboxes:torch.FloatTensor = bboxes.clone().reshape(batch_size, -1, 4).float()
        invalid_bbox_mask = (proj_bboxes == 0) # indicating padded bboxes
        
        if mode == 'a2p':
            # activation map to pixel image
            proj_bboxes[:, :, [0, 2]] *= width_scale_factor
            proj_bboxes[:, :, [1, 3]] *= height_scale_factor
        else:
            # pixel image to activation map
            proj_bboxes[:, :, [0, 2]] /= width_scale_factor
            proj_bboxes[:, :, [1, 3]] /= height_scale_factor
            
        proj_bboxes.masked_fill_(invalid_bbox_mask, 0) # fill padded bboxes back with -1
        proj_bboxes.resize_as_(bboxes)
    else:
        proj_bboxes = bboxes.clone().reshape(batch_size, 1, 4).float()

    proj_bboxes = proj_bboxes.to(config.DEVICE)
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
        # print(bb, c)
        cv2.rectangle(img, (bb[0], bb[1]), (bb[2], bb[3]), colors[c[0]], thickness=2)

    return img

def display_bbox(bboxes, fig, ax, classes=None, in_format='xyxy', color='y', line_width=1):
    colors = {
        "pad" : "#000000", # Pad
        "swimmer": "#00FF00", # Swimmer 
        "floater": "#0000FF", # Floater
        "boat": "#FFFF33", # Boat
        "swimmer_on_boat": "#00FFFF", # Swimmer on Boat
        "floater_on_boat": "#FF66FF", # Floater on Boat
        "life_jacket": "#3399FF"} # Life Jacket
    
    if type(bboxes) == np.ndarray:
        bboxes = torch.from_numpy(bboxes)
    if classes:
        # print(len(bboxes), len(classes))
        assert len(bboxes) == len(classes)
    # convert boxes to xywh format
    bboxes = ops.box_convert(bboxes, in_fmt=in_format, out_fmt='xywh')
    c = 0
    for box in bboxes:
        x, y, w, h = box.numpy()
        # display bounding box
        # print(colors[classes[c]])
        # display category
        if classes:
            if classes[c] == 'pad':
                continue
            rect = patches.Rectangle((x, y), w, h, linewidth=line_width, edgecolor=colors[classes[c]], facecolor='none')
            ax.add_patch(rect)
            ax.text(x + 5, y + 20, classes[c], bbox=dict(facecolor=colors[classes[c]], alpha=0.2))
        else:
            rect = patches.Rectangle((x, y), w, h, linewidth=line_width, edgecolor=color, facecolor='none')
            ax.add_patch(rect)
            # ax.text(x + 5, y + 20, , bbox=dict(facecolor="yellow", alpha=0.2))
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

def display_inference(img_batch: list[torch.Tensor], bboxes_inference, classes_inference, gt_bboxes, gt_classes, width_scale_factor, height_scale_factor):
    nrows, ncols = (1, len(img_batch))
    name2idx = {
        'pad': 0, 
        'swimmer': 1, 
        'floater': 2, 
        'boat': 3, 
        'swimmer_on_boat': 4, 
        'floater_on_boat': 5, 
        'life_jacket': 6}
    
    idx2name = {v:k for k, v in name2idx.items()}

    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 8))
    fig, axes = display_img(img_batch.cpu(), fig, axes)
    print(f"GT BBOXES SHAPE: {gt_bboxes.shape, gt_bboxes[0]}")

    # img_bboxes = project_bboxes(bbox_tensor, width_scale_factor, height_scale_factor)
    # gt_bboxes = project_bboxes(gt_bboxes, width_scale_factor, height_scale_factor, 'p2a')

    for i in range(0, len(img_batch)):
        img_bboxes = project_bboxes(bboxes_inference[i], width_scale_factor, height_scale_factor)
        print(img_bboxes[0:7], gt_bboxes[0:7])
        # # print(f"Bboxes Inference: {bboxes_inference[0]}")
        # gt_bboxes = project_bboxes(gt_bboxes[i], width_scale_factor, height_scale_factor, 'p2a')
        # # print(f"Img Bboxes: {bboxes_inference[i][0]}\nGT Bboxes: {gt_bboxes[0]}")
        classes_pred = [idx2name[cls] for cls in classes_inference[i].tolist()]
        gt_classes_pred = [idx2name[int(cls[0])] for cls in gt_classes[i]]
        # # print(gt_classes_pred)
        fig, _ = display_bbox(img_bboxes.cpu(), fig, axes[i], classes=classes_pred, line_width=1)
        fig, _ = display_bbox(gt_bboxes[i].cpu(), fig, axes[i], classes=gt_classes_pred, line_width=3)

    plt.show()

def get_iou_mat(batch_size, anc_boxes_all, gt_bboxes_all):
    
    # flatten anchor boxes
    anc_boxes_flat = anc_boxes_all.reshape(batch_size, -1, 4)
    # get total anchor boxes for a single image
    tot_anc_boxes = anc_boxes_flat.size(dim=1)
    
    # create a placeholder to compute IoUs amongst the boxes
    ious_mat = torch.zeros((batch_size, tot_anc_boxes, gt_bboxes_all.size(dim=1)))

    # compute IoU of the anc boxes with the gt boxes for all the images
    for i in range(batch_size):
        gt_bboxes = gt_bboxes_all[i]
        anc_boxes = anc_boxes_flat[i]
        # gt_bboxes_projected = project_bboxes(gt_bboxes, config.WIDTH_SCALE_FACTOR, config.HEIGHT_SCALE_FACTOR)
        # anc_boxes_projected = project_bboxes(anc_boxes, width_scale_factor=config.WIDTH_SCALE_FACTOR, height_scale_factor=config.HEIGHT_SCALE_FACTOR)
        temp = ops.box_iou(anc_boxes, gt_bboxes)
        ious_mat[i, :] = temp
        # print(f"IOU MAT within for loop: {ious_mat[i, :]}")
        #flatten, sort, and print for highest values first
        sorted_temp = temp.flatten().sort(descending=True)
        # if sorted_temp.values[0] == 0:
            # print(f"Anchor_boxes: {anc_boxes[0:7]}\nGround Truth: {gt_bboxes[0]}")
            # print(f"Projected GT BBoxes: {gt_bboxes},\nOrig shape: {config.ORIG_IMAGE_SHAPE},\nResized shape: {config.RESIZED_IMAGE_SHAPE},\nProjected Anc Boxes: {anc_boxes}")
            # print(f"Sorted Temp: {sorted_temp}")
    return ious_mat

def calc_gt_offsets(pos_anc_coords, gt_bbox_mapping):
    pos_anc_coords = ops.box_convert(pos_anc_coords, in_fmt='xyxy', out_fmt='cxcywh')
    gt_bbox_mapping = ops.box_convert(gt_bbox_mapping, in_fmt='xyxy', out_fmt='cxcywh')

    gt_cx, gt_cy, gt_w, gt_h = gt_bbox_mapping[:, 0], gt_bbox_mapping[:, 1], gt_bbox_mapping[:, 2], gt_bbox_mapping[:, 3]
    anc_cx, anc_cy, anc_w, anc_h = pos_anc_coords[:, 0], pos_anc_coords[:, 1], pos_anc_coords[:, 2], pos_anc_coords[:, 3]

    tx_ = (gt_cx - anc_cx)/anc_w
    ty_ = (gt_cy - anc_cy)/anc_h
    tw_ = torch.log(gt_w / anc_w)
    th_ = torch.log(gt_h / anc_h)

    return torch.stack([tx_, ty_, tw_, th_], dim=-1)


## Long One
def get_req_anchors(anc_boxes_all: torch.Tensor, gt_bboxes_all: torch.Tensor, gt_classes_all: torch.Tensor, pos_thresh=0.7, neg_thresh=0.3):
    '''
    Prepare necessary data required for training
    
    Input
    ------
    anc_boxes_all - torch.Tensor of shape (B, w_amap, h_amap, n_anchor_boxes, 4)
        all anchor boxes for a batch of images
    gt_bboxes_all - torch.Tensor of shape (B, max_objects, 4)
        padded ground truth boxes for a batch of images
    gt_classes_all - torch.Tensor of shape (B, max_objects)
        padded ground truth classes for a batch of images
        
    Returns
    ---------
    positive_anc_ind -  torch.Tensor of shape (n_pos,)
        flattened positive indices for all the images in the batch
    negative_anc_ind - torch.Tensor of shape (n_pos,)
        flattened positive indices for all the images in the batch
    GT_conf_scores - torch.Tensor of shape (n_pos,), IoU scores of +ve anchors
    GT_offsets -  torch.Tensor of shape (n_pos, 4),
        offsets between +ve anchors and their corresponding ground truth boxes
    GT_class_pos - torch.Tensor of shape (n_pos,)
        mapped classes of +ve anchors
    positive_anc_coords - (n_pos, 4) coords of +ve anchors (for visualization)
    negative_anc_coords - (n_pos, 4) coords of -ve anchors (for visualization)
    positive_anc_ind_sep - list of indices to keep track of +ve anchors
    '''
    anc_boxes_all = anc_boxes_all.to(config.DEVICE)
    gt_bboxes_all = gt_bboxes_all.to(config.DEVICE)
    gt_classes_all = gt_classes_all.to(config.DEVICE)
    
    # get the size and shape parameters
    B, w_amap, h_amap, A, _ = anc_boxes_all.shape
    N = gt_bboxes_all.shape[1] # max number of groundtruth bboxes in a batch
    # print(f"B, wmap, hmap, A: {B, w_amap, h_amap, A}")
    
    # get total number of anchor boxes in a single image
    tot_anc_boxes = A * w_amap * h_amap
    
    # get the iou matrix which contains iou of every anchor box
    # against all the groundtruth bboxes in an image
    # print(f"Anc boxes all: {anc_boxes_all}, GT bboxes all: {gt_bboxes_all}")
    iou_mat: torch.Tensor = get_iou_mat(B, anc_boxes_all, gt_bboxes_all).to(config.DEVICE)
    # print(f"iou  mat: {iou_mat}")

    # for every groundtruth bbox in an image, find the iou 
    # with the anchor box which it overlaps the most
    max_iou_per_gt_box, _ = iou_mat.max(dim=1, keepdim=True)
    # print(f"max iou per gt box: {max_iou_per_gt_box}")

    # print(f"iou indices?: {torch.where(iou_mat == max_iou_per_gt_box)}")
    
    # get positive anchor boxes
    
    # condition 1: the anchor box with the max iou for every gt bbox
    positive_anc_mask: torch.Tensor = torch.logical_and(iou_mat == max_iou_per_gt_box, max_iou_per_gt_box > 0)
    # print(f"positive anc mask: {positive_anc_mask}")
    # condition 2: anchor boxes with iou above a threshold with any of the gt bboxes
    positive_anc_mask = torch.logical_or(positive_anc_mask, iou_mat > pos_thresh)
    
    positive_anc_ind_sep = torch.where(positive_anc_mask)[0] # get separate indices in the batch
    # combine all the batches and get the idxs of the +ve anchor boxes
    # print(f"postive anc ind max {positive_anc_ind_sep.max()}")
    positive_anc_mask = positive_anc_mask.flatten(start_dim=0, end_dim=1)
    positive_anc_ind = torch.where(positive_anc_mask)[0]
    # print(f"Positive Anchor Indices: {positive_anc_ind}")
    
    # for every anchor box, get the iou and the idx of the
    # gt bbox it overlaps with the most
    max_iou_per_anc, max_iou_per_anc_ind = iou_mat.max(dim=-1)
    max_iou_per_anc = max_iou_per_anc.flatten(start_dim=0, end_dim=1)
    
    # get iou scores of the +ve anchor boxes
    GT_conf_scores = max_iou_per_anc[positive_anc_ind]
    
    # get gt classes of the +ve anchor boxes
    
    # expand gt classes to map against every anchor box
    # print(f"Classes All  Shape: {anc_boxes_all.shape, gt_classes_all.shape, gt_bboxes_all.shape, B, A, N}")
    # print(f"Classes All: {gt_classes_all}")
    gt_classes_expand: torch.Tensor = gt_classes_all.view(B, 1, N).expand(B, tot_anc_boxes, N)
    # for every anchor box, consider only the class of the gt bbox it overlaps with the most
    # print(f"Gt classes expand: {gt_classes_expand.device}, max iou: {max_iou_per_anc_ind.device}")
    GT_class = torch.gather(gt_classes_expand, -1, max_iou_per_anc_ind.unsqueeze(-1)).squeeze(-1)
    # combine all the batches and get the mapped classes of the +ve anchor boxes
    GT_class = GT_class.flatten(start_dim=0, end_dim=1)
    GT_class_pos = GT_class[positive_anc_ind]
    
    # get gt bbox coordinates of the +ve anchor boxes
    
    # expand all the gt bboxes to map against every anchor box
    gt_bboxes_expand: torch.Tensor = gt_bboxes_all.view(B, 1, N, 4).expand(B, tot_anc_boxes, N, 4)
    # for every anchor box, consider only the coordinates of the gt bbox it overlaps with the most
    GT_bboxes = torch.gather(gt_bboxes_expand, -2, max_iou_per_anc_ind.reshape(B, tot_anc_boxes, 1, 1).repeat(1, 1, 1, 4))
    # combine all the batches and get the mapped gt bbox coordinates of the +ve anchor boxes
    GT_bboxes = GT_bboxes.flatten(start_dim=0, end_dim=2)
    GT_bboxes_pos = GT_bboxes[positive_anc_ind]
    
    # get coordinates of +ve anc boxes
    anc_boxes_flat = anc_boxes_all.flatten(start_dim=0, end_dim=-2) # flatten all the anchor boxes
    positive_anc_coords = anc_boxes_flat[positive_anc_ind]
    
    # calculate gt offsets
    GT_offsets = calc_gt_offsets(positive_anc_coords, GT_bboxes_pos).to(config.DEVICE)
    
    # get -ve anchors
    
    # condition: select the anchor boxes with max iou less than the threshold
    negative_anc_mask = (max_iou_per_anc < neg_thresh)
    negative_anc_ind = torch.where(negative_anc_mask)[0]
    # print(f"Negative Anchor Indices: {negative_anc_ind}")

    # sample -ve samples to match the +ve samples
    negative_anc_ind = negative_anc_ind[torch.randint(0, negative_anc_ind.shape[0], (positive_anc_ind.shape[0],))]
    negative_anc_coords = anc_boxes_flat[negative_anc_ind]
    
    return positive_anc_ind, negative_anc_ind, GT_conf_scores, GT_offsets, GT_class_pos, \
         positive_anc_coords, negative_anc_coords, positive_anc_ind_sep

def calc_giou_loss(proposals_batch: list[torch.Tensor], ground_truth: torch.Tensor):
    giou_batch_loss = 0
    
    for i, sample_proposals in enumerate(proposals_batch):
        
        # print(f"Sample Proposals: {sample_proposals}")
        giou_img_loss = 0

        sample_proposals = project_bboxes(sample_proposals, config.WIDTH_SCALE_FACTOR, config.HEIGHT_SCALE_FACTOR)

        for individual_bbox in sample_proposals:
            closest_bbox, _ = find_closest_gt(individual_bbox, ground_truth[i])
            individual_giou_loss = ops.generalized_box_iou_loss(individual_bbox, closest_bbox)
            giou_img_loss += individual_giou_loss

        giou_batch_loss += giou_img_loss
    return giou_batch_loss


def find_closest_gt(bbox_proposals: torch.Tensor, img_gt_bboxes: torch.Tensor):
    bbox_proposals_center = ((bbox_proposals[0] + bbox_proposals[2]) / 2, (bbox_proposals[1] + bbox_proposals[3]) / 2)
    img_gt_bboxes_centers = [((img_gt_bbox[0] + img_gt_bbox[2])/2, (img_gt_bbox[1] + img_gt_bbox[3])/2) for img_gt_bbox in img_gt_bboxes]
    distances = []
    for img_gt_bbox in img_gt_bboxes_centers:
        distances.append(math.sqrt((img_gt_bbox[0] - bbox_proposals_center[0])**2 + (img_gt_bbox[1] - bbox_proposals_center[1])**2))
    closest_gt_index = distances.index(min(distances))
    return img_gt_bboxes[closest_gt_index], closest_gt_index
