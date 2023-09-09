import os
import torch

from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from utils import parse_annotations
from skimage.io import imread
from skimage.transform import resize
from tqdm import tqdm

class MOTDataset(Dataset):
    
    def __init__(self, img_dir_path, annotations_path, resized_img_shape, orig_img_shape):
        
        self.annotation_path = annotations_path
        self.img_dir = img_dir_path
        self.img_shape = resized_img_shape
        self.orig_img_shape = orig_img_shape

        # self.img_data_full, self.bboxes_all, self.classes_all = self.load_data()

    def __len__(self):
        return self.img_data_full.size(dim=0)
    
    def __getitem__(self, index: int):
        # return self.img_data_full[index], self.bboxes_all[index], self.classes_all[index]
        data = self.load_data(index)
    
    def load_data(self, index) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        
        # img_data_full = []

        # bboxes_all, classes_all, img_paths = parse_annotations(
        #     annotation_path=self.annotation_path,
        #     image_dir=self.img_dir,
        #     resized_img_size=self.img_shape,
        #     original_img_shape=self.orig_img_shape)
        
        # print("Loading image data from file paths...")
        # for i, img_path in enumerate(tqdm(img_paths)):
        
        img_path = ids[index]

        # Check if image path is valid, otherwise skip
        if (not img_path) or (not os.path.exists(img_path)):
            continue
            
        # read and resize image
        img = imread(img_path)
        img = resize(img, self.img_shape)
        
        # convert image to torch tensor and reshape it so channels come first
        img_tensor = torch.from_numpy(img).permute(2, 0, 1)
        # img_data_full.append(img_tensor)

        # img_data_stacked = torch.stack(self.img_data_full, dim=0)

        return img_data_stacked.to(dtype=torch.float32), bboxes_all, classes_all