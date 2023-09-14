import os
import torch
import faster_rcnn
import numpy as np
import json
import sys
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from skimage.io import imread

from dataset_loader import MOTDataset
from utils import split_ids, parse_annotations, load_data, custom_collate, display_batch, gen_anc_centers, display_img, display_grid, gen_anc_base, project_bboxes, display_bbox, get_req_anchors
from faster_rcnn import FasterRCNN

def od_pipeline():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    resize_factor = 5
    img_dir_path = "data/train"
    annotations_path = "data/annotations/instances_train_swimmer.json"
    test_image = np.array(imread(os.path.join(img_dir_path, '0.jpg')))
    orig_img_shape = test_image.shape
    resized_img_shape = (round(orig_img_shape[0] / resize_factor), round(orig_img_shape[1] / resize_factor))
    save_partition_path = "custom_data/partition_ids.json"
    save_bboxes_path = "custom_data/bboxes_all.json"
    save_categories_all = "custom_data/categories_all.json"
    
    partition_ids, bboxes_all, categories_all = load_data(
        save_partition_path=save_partition_path,
        save_bboxes_path=save_bboxes_path,
        save_categories_path=save_categories_all,
        annotations_path=annotations_path,
        img_dir_path=img_dir_path,
        resized_img_shape=resized_img_shape,
        orig_img_shape=orig_img_shape,
        split=(0.7, 0.1, 0.2),
        force_overwrite=False
    )

    # display(image_ids=partition_ids["train"], 
    #         bboxes_all=bboxes_all["train"], 
    #         categories_all=categories_all["train"] , 
    #         index=1, #int(sys.argv[1]), 
    #         resized_img_shape=resized_img_shape, orig_img_shape=orig_img_shape)

    mot_custom_dataset = MOTDataset(
        image_ids=partition_ids["train"],
        bboxes=bboxes_all["train"],
        categories = categories_all["train"],
        resized_img_shape=resized_img_shape,
        orig_img_shape=orig_img_shape
    )

    #Use pad sequence if facing any more errors with the return type of the data_loader
    mot_custom_dataloader = DataLoader(mot_custom_dataset, batch_size=2, shuffle=True, collate_fn=custom_collate)

    for img_batch, class_labels_batch, bboxes_labels_batch in mot_custom_dataloader:
        img_data_all: torch.FloatTensor = img_batch
        classes_all: torch.Tensor = class_labels_batch
        bboxes_all: torch.Tensor = bboxes_labels_batch
        break

    # display_batch(batch=(img_data_all, classes_all, bboxes_all))
    model = FasterRCNN(device)
    torch.cuda.empty_cache()
    model.to(device)
    img_data_all = img_data_all.to(device)
    classes_all = classes_all.to(device)
    bboxes_all = bboxes_all.to(device)

    # print(f"Model: {model.device}, Img Data: {img_data_all.device}, Classes: {classes_all.device}, Bboxes: {bboxes_all.device}")

    print(img_data_all.size(), resized_img_shape)
    out = model.forward(img_data_all)
    out_c, out_h, out_w = out.size(dim=1), out.size(dim=2), out.size(dim=3)
    print(out_c, out_h, out_w)

    width_scale_factor = resized_img_shape[1] // out_w
    height_scale_factor = resized_img_shape[0] // out_h

    print(height_scale_factor, width_scale_factor)
    nrows, ncols = (1, 2)
    fig= plt.figure(figsize=(16, 8))

    filters_data =[filters[0].cpu().detach().numpy() for filters in out[:2]]

    for i in range(0, len(filters_data)):
        fig.add_subplot(nrows, ncols, i+1)
        plt.imshow(filters_data[i])
        plt.axis("on")
    
    plt.show()

    img_data_all = img_data_all.cpu()
    classes_all = classes_all.cpu()
    bboxes_all = bboxes_all.cpu()

    anc_x, anc_y = gen_anc_centers((out_h, out_w))
    # project anchor centers onto the original image
    anc_pts_x_proj = anc_x.clone() * width_scale_factor 
    anc_pts_y_proj = anc_y.clone() * height_scale_factor
    
    # Display image with all anchor points
    # fig, axes = plt.subplots(nrows, ncols, figsize=(16, 8))
 
    # fig, axes = display_img(img_data_all, fig, axes)
    # fig, _ = display_grid(anc_pts_x_proj, anc_pts_y_proj, fig, axes[0])
    # fig, _ = display_grid(anc_pts_x_proj, anc_pts_y_proj, fig, axes[1])

    # plt.show()

    # Display all images with anchor points and all boxes for one point
    #NOTE: Think about modifying generation code to make widthwise boxes
    anc_scales = [0.25, 0.5, 1, 2, 3] #2,4,6
    anc_ratios = [0.5, 1, 1.5] #, 0.5, 1, 1.5
    n_anc_boxes = len(anc_scales) * len(anc_ratios) # number of anchor boxes for each anchor point

    anc_base = gen_anc_base(anc_x, anc_y, anc_scales, anc_ratios, (out_h, out_w))
    anc_boxes_all = anc_base.repeat(img_data_all.size(dim=0), 1, 1, 1, 1)

    nrows, ncols = (1, 2)
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 8))

    fig, axes = display_img(img_data_all, fig, axes)

    # project anchor boxes to the image
    anc_boxes_proj = project_bboxes(anc_boxes_all, width_scale_factor, height_scale_factor, mode='a2p')

    # plot anchor boxes around selected anchor points
    sp_1 = [5, 8]
    sp_2 = [12, 9]
    bboxes_1 = anc_boxes_proj[0][sp_1[0], sp_1[1]]
    bboxes_2 = anc_boxes_proj[1][sp_2[0], sp_2[1]]

    fig, _ = display_grid(anc_pts_x_proj, anc_pts_y_proj, fig, axes[0], (anc_pts_x_proj[sp_1[0]], anc_pts_y_proj[sp_1[1]]))
    fig, _ = display_grid(anc_pts_x_proj, anc_pts_y_proj, fig, axes[1], (anc_pts_x_proj[sp_2[0]], anc_pts_y_proj[sp_2[1]]))
    fig, _ = display_bbox(bboxes_1, fig, axes[0])
    fig, _ = display_bbox(bboxes_2, fig, axes[1])
    plt.show()

    #Display all images with anchor points and all boxes for all anchor points
    # nrows, ncols = (1, 2)
    # fig, axes = plt.subplots(nrows, ncols, figsize=(16, 8))

    # fig, axes = display_img(img_data_all, fig, axes)

    # # plot all anchor boxes
    # for x in range(anc_pts_x_proj.size(dim=0)):
    #     for y in range(anc_pts_y_proj.size(dim=0)):
    #         bboxes = anc_boxes_proj[0][x, y]
    #         fig, _ = display_bbox(bboxes, fig, axes[0], line_width=1)
    #         fig, _ = display_bbox(bboxes, fig, axes[1], line_width=1)

    # # plot feature grid
    # fig, _ = display_grid(anc_pts_x_proj, anc_pts_y_proj, fig, axes[0])
    # fig, _ = display_grid(anc_pts_x_proj, anc_pts_y_proj, fig, axes[1])

    # plt.show()

    # IOU OVERLAP -----------------------------------------
    pos_thresh = 0.6
    neg_thresh = 0.3
    # print(f"Bboxes All: {bboxes_all}")
    # project gt bboxes onto the feature map
    gt_bboxes_proj = project_bboxes(bboxes_all, width_scale_factor, height_scale_factor, mode='p2a')
    positive_anc_ind, negative_anc_ind, GT_conf_scores, \
    GT_offsets, GT_class_pos, positive_anc_coords, \
    negative_anc_coords, positive_anc_ind_sep = get_req_anchors(anc_boxes_all, gt_bboxes_proj, classes_all, pos_thresh, neg_thresh)
    # print(positive_anc_coords, negative_anc_coords)

    # project anchor coords to the image space
    pos_anc_proj = project_bboxes(positive_anc_coords, width_scale_factor, height_scale_factor, mode='a2p')
    neg_anc_proj = project_bboxes(negative_anc_coords, width_scale_factor, height_scale_factor, mode='a2p')

    # grab +ve and -ve anchors for each image separately

    anc_idx_1 = torch.where(positive_anc_ind_sep == 0)[0]
    anc_idx_2 = torch.where(positive_anc_ind_sep == 1)[0]

    pos_anc_1 = pos_anc_proj[anc_idx_1]
    pos_anc_2 = pos_anc_proj[anc_idx_2]

    neg_anc_1 = neg_anc_proj[anc_idx_1]
    neg_anc_2 = neg_anc_proj[anc_idx_2]

    nrows, ncols = (1, 2)
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 8))

    fig, axes = display_img(img_data_all, fig, axes)

    # plot groundtruth bboxes
    fig, _ = display_bbox(bboxes_all[0], fig, axes[0])
    fig, _ = display_bbox(bboxes_all[1], fig, axes[1])

    # plot positive anchor boxes
    fig, _ = display_bbox(pos_anc_1, fig, axes[0], color='g')
    fig, _ = display_bbox(pos_anc_2, fig, axes[1], color='g')

    # plot negative anchor boxes
    fig, _ = display_bbox(neg_anc_1, fig, axes[0], color='r')
    fig, _ = display_bbox(neg_anc_2, fig, axes[1], color='r')
    
    plt.show()


if __name__ == '__main__':
    od_pipeline()