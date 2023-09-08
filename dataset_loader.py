import torch

from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from utils import parse_annotations

class MOTDataset(Dataset):
    
    def __init__(self, img_dir_path, annotations_path, resized_img_size, orig_img_size):
        
        self.annotation_path = annotations_path
        self.img_dir = img_dir_path
        self.img_size = resized_img_size
        self.orig_img_size = orig_img_size

        self.img_data_full, self.bboxes_all, self.classes_all = self.load_data()

    def __len__(self):
        return self.img_data_full.size(dim=0)
    
    def __getitem__(self, index: int):
        return self.img_data_full[index], self.bboxes_all[index], self.classes_all[index]
    
    def load_data(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bboxes_all, classes_all, img_paths = parse_annotations(
            annotation_path=self.annotation_path,
            image_dir=self.img_dir,
            resized_img_size=self.img_size,
            original_img_size=self.orig_img_size)
        