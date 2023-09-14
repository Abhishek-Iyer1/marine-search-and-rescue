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


def execute_training_loop():

    # # Set device to GPU if it exists
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set important parameters and paths
    test_image = np.array(imread(os.path.join(config.IMG_DIR_PATH, '0.jpg')))
    orig_img_shape = test_image.shape
    resized_img_shape = (round(orig_img_shape[0] / config.RESIZE_FACTOR), round(orig_img_shape[1] / config.RESIZE_FACTOR))

    # Load all image paths, bounding boxes, and respective classes
    partition_ids, bboxes_all, categories_all = load_data(
        save_partition_path=config.SAVE_IMAGES_PATH,
        save_bboxes_path=config.SAVE_BBOXES_PATH,
        save_categories_path=config.SAVE_CLASSES_PATH,
        annotations_path=config.ANNOTATIONS_PATH,
        img_dir_path=config.IMG_DIR_PATH,
        resized_img_shape=resized_img_shape,
        orig_img_shape=orig_img_shape,
        split=(0.7, 0.1, 0.2),
        force_overwrite=False
    )

    # Create a train instance of the MOTDataset type in order to train the model
    mot_train_dataset = MOTDataset(
        image_ids=partition_ids["train"],
        bboxes=bboxes_all["train"],
        categories = categories_all["train"],
        resized_img_shape=resized_img_shape,
        orig_img_shape=orig_img_shape
    )

    # Create a train instance of the Dataloader using the train MOTDataset for batches and shuffling
    mot_train_dataloader = DataLoader(mot_train_dataset, batch_size=4, shuffle=True, collate_fn=custom_collate)

    # Size of input image and the output of the backbone
    img_size = (resized_img_shape[0], resized_img_shape[1])
    out_size = (config.OUT_H, config.OUT_W)

    # Create a TwoStageDetector instance to run the training loop on
    detector = TwoStageDetector(img_size, out_size, config.OUT_C, config.N_CLASSES, config.ROI_SIZE).to(config.DEVICE)

    # Run the training loop, update the weights, save them
    loss_list = training_loop(detector, config.LEARNING_RATE, mot_train_dataloader, config.N_EPOCHS, config.DEVICE)
    torch.save(detector.state_dict(), config.MODEL_SAVE_PATH)
    plt.plot(loss_list)
    plt.show()

def training_loop(model, learning_rate, train_dataloader, n_epochs, device):

    # Set
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    model.train()
    loss_list = []
    
    for i in tqdm(range(n_epochs)):
        total_loss = 0
        for img_batch, gt_classes_batch, gt_bboxes_batch in train_dataloader:
            
            img_batch: torch.Tensor = img_batch.to(device)
            # gt_bboxes_batch: torch.Tensor = gt_bboxes_batch.to(device)
            # gt_classes_batch: torch.Tensor = gt_classes_batch.to(device)

            # Forward pass
            loss = model(img_batch, gt_bboxes_batch, gt_classes_batch)
            
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        # Keep track of loss for every epoch
        loss_list.append(total_loss)
        
    return loss_list

if __name__ == '__main__':
    execute_training_loop()