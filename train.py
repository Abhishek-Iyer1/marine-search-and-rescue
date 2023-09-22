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
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision.ops import generalized_box_iou_loss 

def execute_training_loop():

    # # Set device to GPU if it exists
    print(f"Device: {config.DEVICE}")

    dataloaders = generate_dataloader(batch_size=config.BATCH_SIZE, shrink_factor=2000, generate_val=True, generate_test=True)

    mot_train_dataloader = dataloaders["train"]
    mot_val_dataloader = dataloaders["val"]
    mot_test_dataloader = dataloaders["test"]
    print(f"Length of Dataloaders: {len(mot_train_dataloader), len(mot_val_dataloader), len(mot_test_dataloader)}")
    # Size of input image and the output of the backbone
    img_size = (config.RESIZED_IMAGE_SHAPE[0], config.RESIZED_IMAGE_SHAPE[1])
    out_size = (config.OUT_H, config.OUT_W)

    # Create a TwoStageDetector instance to run the training loop on
    detector = TwoStageDetector(img_size, out_size, config.OUT_C, config.N_CLASSES, config.ROI_SIZE).to(config.DEVICE)

    # Run the training loop, update the weights, save them
    loss_dict = training_loop(detector, config.N_EPOCHS, config.LEARNING_RATE, mot_train_dataloader, mot_val_dataloader, config.MODEL_SAVE_PATH)
    torch.save(detector.state_dict(), config.MODEL_SAVE_PATH)
    
    fig = plt.figure(figsize=(16, 8))
    fig.add_subplot(1, 2, 1)
    plt.plot(loss_dict["train"])
    plt.title("Training Loss")

    fig.add_subplot(1, 2, 2)
    plt.plot(loss_dict["val"])
    plt.title("Validation Loss")

    plt.show()

def training_loop(model: TwoStageDetector,  n_epochs: int, learning_rate: float, train_dataloader: DataLoader, val_dataloader: DataLoader, checkpoint_path: str):

    # Set
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, 'min', 0.5, 7)
    min_valid_loss = np.inf
    model.train()
    loss_list = {
        "train": [],
        "val": []
    }
    
    for i in tqdm(range(n_epochs)):
        
        # Begin training loop ------------------------------------------------------------------------

        # Initialize training loss
        train_loss = 0      
        # Set model to train mode
        model.train()

        for img_batch, gt_classes_batch, gt_bboxes_batch in tqdm(train_dataloader):
            
            img_batch: torch.Tensor = img_batch.to(config.DEVICE)
            gt_bboxes_batch: torch.Tensor = gt_bboxes_batch.to(config.DEVICE)
            gt_classes_batch: torch.Tensor = gt_classes_batch.to(config.DEVICE)

            # Clear the gradients
            optimizer.zero_grad()

            # Forward pass and find loss
            batch_loss = model.forward(img_batch, gt_bboxes_batch, gt_classes_batch)
            # print(batch_loss)
            
            # Backpropagation
            batch_loss.backward()

            # Update weights
            optimizer.step()
            
            train_loss += batch_loss.item()
        
        print(f"Epoch Training Loss: {train_loss}")
        
        #Initialize valid loss
        valid_loss = 0
        # Set model to eval mode
        model.eval()

        for val_images, val_classes, val_bboxes in val_dataloader:
            
            val_images: torch.Tensor = val_images.to(config.DEVICE)
            val_bboxes: torch.Tensor = val_bboxes.to(config.DEVICE)
            val_classes: torch.Tensor = val_classes.to(config.DEVICE)
            
            # Forward Pass
            proposals_final, conf_scores_final, classes_final, cls_scores = model.inference(val_images, config.CONF_THRESH, config.NMS_THRESH)
            # Find the Loss
            loss = model.forward(val_images, val_bboxes, val_classes)

            # Alternate loss
            # print(f"Proposals: {proposals_final[0]}, Val Bboxes: {val_bboxes}, Classes_final : {classes_final}, Val classes: {val_classes}")

            bbox_loss = calc_giou_loss(proposals_final, val_bboxes)
            # class_loss = F.cross_entropy(cls_scores, val_classes.long())

            # print(f"Bbox Loss GIOU: {bbox_loss}, Class Loss Cross Entropy: {class_loss}")
            # IOU LOSS OVER PROPOSALS AND GT BBOXES
            # get_req_anchors()
            # cls_loss = calc_cls_loss(conf_scores_pos, conf_scores_neg, batch_size)
            # reg_loss = calc_bbox_reg_loss(GT_offsets, offsets_pos, batch_size)
            valid_loss = bbox_loss * config.BBOX_WEIGHT + loss * config.CLASS_WEIGHT
            # Calculate Loss
            valid_loss += loss.item()
        
        scheduler.step(valid_loss)

        print(f"Epoch Valid Loss: {valid_loss}")
        
        # Keep track of loss for every epoch
        loss_list["train"].append(train_loss)
        loss_list["val"].append(valid_loss)
        
        # If epoch loss better than previous minimum, update loss, save model
        if min_valid_loss > valid_loss:
            
            # Update the new minimum
            min_valid_loss = valid_loss
            
            # Saving State Dict
            print(f"Saving model to path: {checkpoint_path}")
            torch.save(model.state_dict(), checkpoint_path)

        # End training loop --------------------------------------------------------------------------
    
    return loss_list

if __name__ == '__main__':
    execute_training_loop()