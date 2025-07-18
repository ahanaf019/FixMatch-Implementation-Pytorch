import torch
from torch.utils.data import Dataset
from utils.utils import read_image


class SSDataset(Dataset):
    def __init__(self, image_paths, labels, image_size, transforms_1=None, transforms_2=None):
        super().__init__()
        self.image_paths = image_paths
        self.labels = labels
        self.image_size = image_size
        self.transforms_1 = transforms_1
        self.transforms_2 = transforms_2
    

    def __len__(self):
        return len(self.image_paths)
    

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        image = read_image(image_path)


        if self.labels is not None:
            return self.transforms_1(image), torch.tensor(self.labels[index])
        return  self.transforms_1(image), self.transforms_2(image)