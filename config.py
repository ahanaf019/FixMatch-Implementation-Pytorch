import os
from torchvision import transforms

DB_PATH_ROOT = f'{os.environ["HOME"]}/Datasets/STL10'
BATCH_SIZE = 128
IMAGE_SIZE = 96
WARMUP_EPOCHS = 5
LEARNING_RATE = 1e-4
NUM_EPOCHS = 20

NUM_CLASSES = 10
NUM_UNLABELED = 15000
THRESHOLD = 0.90
LAMBDA_U = 1.0


weak_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomResizedCrop(size=(IMAGE_SIZE, IMAGE_SIZE), scale=(0.8, 1.0), interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize(
        mean =[0.4914 , 0.4822 , 0.4465] , 
        std=[0.2470 , 0.2435 , 0.2616]
    ),
])


strong_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomResizedCrop(size=(IMAGE_SIZE, IMAGE_SIZE), scale=(0.8, 1.0), interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.ColorJitter(
        brightness=0.4,
        contrast=0.4,
        saturation=0.4,
        hue=0.1
    ),
    transforms.RandomGrayscale(p=0.2),
    transforms.ToTensor(),
    transforms.Normalize(
        mean =[0.4914 , 0.4822 , 0.4465] , 
        std=[0.2470 , 0.2435 , 0.2616]
    ),
])


val_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean =[0.4914 , 0.4822 , 0.4465] , 
        std=[0.2470 , 0.2435 , 0.2616]
    ),
])
