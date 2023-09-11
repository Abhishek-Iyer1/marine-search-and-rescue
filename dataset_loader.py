import os
import torch

from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from utils import parse_annotations
from skimage.io import imread
from skimage.transform import resize
from tqdm import tqdm

class MOTDataset(Dataset):
    
    def __init__(self, image_ids, bboxes, categories, resized_img_shape, orig_img_shape):

        self.img_shape = resized_img_shape
        self.orig_img_shape = orig_img_shape
        self.image_ids = image_ids
        self.bboxes = bboxes
        self.categories =  categories
        # self.img_data_full, self.bboxes_all, self.classes_all = self.load_data()

    def __len__(self) -> int:
        return len(self.image_ids)
    
    def __getitem__(self, index: int) -> tuple[torch.Tensor, dict]:
        # return img_tensor and label
        return self.load_data(index)
    
    def load_data(self, index) -> tuple[torch.Tensor, dict]:
        
        img_path = self.image_ids[index]

        # Check if image path is valid, otherwise skip
        if (img_path) and (os.path.exists(img_path)):
            
            # read and resize image
            img = imread(img_path)
            # img = resize(img, self.img_shape)
        
            # convert image to torch tensor and reshape it so channels come first
            img_tensor = torch.from_numpy(img).permute(2, 0, 1)
            class_label = torch.IntTensor(self.categories[index - 1])
            bbox_label = torch.IntTensor(self.bboxes[index - 1])

        return img_tensor, class_label, bbox_label