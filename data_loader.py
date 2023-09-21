import config

from torch.utils.data import DataLoader
from dataset_loader import MOTDataset
from utils import load_data, custom_collate
from tqdm import tqdm

def generate_dataloader(batch_size: int, shrink_factor: int = 1, generate_train: bool = True, generate_val: bool = False, generate_test: bool = False, shuffle: bool = True, overwrite_saved_data: bool = False) -> dict[str, DataLoader]: 
    
    dataloaders = {
        "train": None,
        "val": None,
        "test": None
    }

    img_size = (config.RESIZED_IMAGE_SHAPE[0], config.RESIZED_IMAGE_SHAPE[1])
    out_size = (config.OUT_H, config.OUT_W)

    width_scale_factor = config.RESIZED_IMAGE_SHAPE[1] // config.OUT_W
    height_scale_factor = config.RESIZED_IMAGE_SHAPE[0] // config.OUT_H

    partition_ids, bboxes_all, categories_all = load_data(
        save_partition_path=config.SAVE_IMAGES_PATH,
        save_bboxes_path=config.SAVE_BBOXES_PATH,
        save_categories_path=config.SAVE_CLASSES_PATH,
        annotations_path=config.ANNOTATIONS_PATH,
        img_dir_path=config.IMG_DIR_PATH,
        resized_img_shape=config.RESIZED_IMAGE_SHAPE,
        orig_img_shape=config.ORIG_IMAGE_SHAPE,
        split=(0.7, 0.1, 0.2),
        force_overwrite=overwrite_saved_data
    )
    
    if generate_train:
        print("Generating Train Dataloader...")

        partition_ids["train_lite"] = []
        bboxes_all["train_lite"] = []
        categories_all["train_lite"] = []   
        n_examples = len(partition_ids["train"]) 

        for i in tqdm(range(1, n_examples)):
            if i % shrink_factor == 0:
                partition_ids["train_lite"].append(partition_ids["train"][i])
                bboxes_all["train_lite"].append(bboxes_all["train"][i-1])
                categories_all["train_lite"].append(categories_all["train"][i-1])

        mot_train_dataset = MOTDataset(
            image_ids=partition_ids["train_lite"],
            bboxes=bboxes_all["train_lite"],
            categories = categories_all["train_lite"],
            resized_img_shape=config.RESIZED_IMAGE_SHAPE,
            orig_img_shape=config.ORIG_IMAGE_SHAPE
        )

        #Use pad sequence if facing any more errors with the return type of the data_loader
        mot_train_dataloader = DataLoader(mot_train_dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=custom_collate)
        dataloaders["train"] = mot_train_dataloader
        
    if generate_val:
        print("Generating Validation Dataloader...")
        
        partition_ids["val_lite"] = []
        bboxes_all["val_lite"] = []
        categories_all["val_lite"] = []   
        
        n_examples = len(partition_ids["validation"]) 
        
        for i in tqdm(range(1, n_examples)):
            if i % shrink_factor == 0:
                partition_ids["val_lite"].append(partition_ids["validation"][i])
                bboxes_all["val_lite"].append(bboxes_all["validation"][i-1])
                categories_all["val_lite"].append(categories_all["validation"][i-1])

        mot_val_dataset = MOTDataset(
            image_ids=partition_ids["val_lite"],
            bboxes=bboxes_all["val_lite"],
            categories = categories_all["val_lite"],
            resized_img_shape=config.RESIZED_IMAGE_SHAPE,
            orig_img_shape=config.ORIG_IMAGE_SHAPE
        )

        mot_val_dataloader = DataLoader(mot_val_dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=custom_collate)
        dataloaders["val"] = mot_val_dataloader
