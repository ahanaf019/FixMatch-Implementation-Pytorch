import torch
from torch.utils.data import Dataset
from utils.utils import read_image


class SSDataset(Dataset):
    def __init__(self, image_paths, labels, image_size, weak_augments=None, strong_augments=None):
        super().__init__()
        self.image_paths = image_paths
        self.labels = labels
        self.image_size = image_size
        self.weak_transforms = weak_augments
        self.strong_transforms = strong_augments
    

    def __len__(self):
        return len(self.image_paths)
    

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        image = read_image(image_path)


        if self.labels is not None:
            return self.weak_transforms(image), torch.tensor(self.labels[index])
        return  self.weak_transforms(image), self.strong_transforms(image)