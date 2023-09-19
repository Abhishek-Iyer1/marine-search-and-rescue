import os
import numpy as np
import torch
import torch.optim as optim
import config

from torch.utils.data import DataLoader
from skimage.io import imread
from tqdm import tqdm

from model import *
from dataset_loader import MOTDataset
from utils import load_data, custom_collate
from data_loader import generate_dataloader


def execute_training_loop():

    # # Set device to GPU if it exists
    print(f"Device: {config.DEVICE}")

    # Set important parameters and paths

    # Load all image paths, bounding boxes, and respective classes
    partition_ids, bboxes_all, categories_all = load_data(
        save_partition_path=config.SAVE_IMAGES_PATH,
        save_bboxes_path=config.SAVE_BBOXES_PATH,
        save_categories_path=config.SAVE_CLASSES_PATH,
        annotations_path=config.ANNOTATIONS_PATH,
        img_dir_path=config.IMG_DIR_PATH,
        resized_img_shape=config.RESIZED_IMAGE_SHAPE,
        orig_img_shape=config.ORIG_IMAGE_SHAPE,
        split=(0.7, 0.1, 0.2),
        force_overwrite=False
    )
    
    partition_ids["train_lite"] = []
    bboxes_all["train_lite"] = []
    categories_all["train_lite"] = [] 

    # Shortened Image Dataset
    n_examples = len(partition_ids["train"])
    for i in range(1, n_examples):
        if i % 400 == 0:
            partition_ids["train_lite"].append(partition_ids["train"][i])
            bboxes_all["train_lite"].append(bboxes_all["train"][i-1])
            categories_all["train_lite"].append(categories_all["train"][i-1])

    print(len(partition_ids["train_lite"]), len(bboxes_all["train_lite"]), len(categories_all["train_lite"]))
    # Create a train instance of the MOTDataset type in order to train the model
    mot_train_dataset = MOTDataset(
        image_ids=partition_ids["train_lite"],
        bboxes=bboxes_all["train_lite"],
        categories = categories_all["train_lite"],
        resized_img_shape=config.RESIZED_IMAGE_SHAPE,
        orig_img_shape=config.ORIG_IMAGE_SHAPE
    )

    # Create a train instance of the Dataloader using the train MOTDataset for batches and shuffling
    mot_train_dataloader = DataLoader(mot_train_dataset, num_workers= 8, batch_size=16, shuffle=False, collate_fn=custom_collate, pin_memory=True)

    # Size of input image and the output of the backbone
    img_size = (config.RESIZED_IMAGE_SHAPE[0], config.RESIZED_IMAGE_SHAPE[1])
    out_size = (config.OUT_H, config.OUT_W)

    # Create a TwoStageDetector instance to run the training loop on
    detector = TwoStageDetector(img_size, out_size, config.OUT_C, config.N_CLASSES, config.ROI_SIZE).to(config.DEVICE)

    # Run the training loop, update the weights, save them
    loss_list = training_loop(detector, config.LEARNING_RATE, mot_train_dataloader, config.N_EPOCHS, config.DEVICE)
    torch.save(detector.state_dict(), config.MODEL_SAVE_PATH)
    # plt.plot(loss_list)
    # plt.show()

def training_loop(model: TwoStageDetector, learning_rate, train_dataloader, n_epochs, device):

    # Set
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    model.train()
    loss_list = []
    
    for i in tqdm(range(n_epochs)):
        total_loss = 0
        for img_batch, gt_classes_batch, gt_bboxes_batch in tqdm(train_dataloader):
            
            img_batch: torch.Tensor = img_batch.to(device)
            gt_bboxes_batch: torch.Tensor = gt_bboxes_batch.to(device)
            gt_classes_batch: torch.Tensor = gt_classes_batch.to(device)

            # Forward pass
            batch_loss = model.forward(img_batch, gt_bboxes_batch, gt_classes_batch)
            # print(f"GT BBOXES: {gt_bboxes_batch}\nGT CLASSES: {gt_classes_batch}")
            print(batch_loss)

            # Backpropagation
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()
            
            total_loss += batch_loss.item()

            # nrows, ncols = (1, len(img_batch))
            # fig, axes = plt.subplots(nrows, ncols, figsize=(16, 8))
            # fig, _ = display_img(img_batch.cpu(), fig, axes)
            # plt.show()

        print(f"Epoch Loss: {total_loss}")
        # Keep track of loss for every epoch
        loss_list.append(total_loss)
        
    return loss_list

if __name__ == '__main__':
    execute_training_loop()