import os
import torch
from torchvision import transforms

DB_PATH_ROOT = f'{os.environ["HOME"]}/Datasets/STL10'
BATCH_SIZE = 128
IMAGE_SIZE = 96
WARMUP_STEPS = 2000
LEARNING_RATE = 3e-3
NUM_TRAINING_STEPS = 2**16


NUM_CLASSES = 10
NUM_UNLABELED = 100000
THRESHOLD = 0.90
LAMBDA_U = 1.0


def custom_lr_schedule(step):
    multiplier = torch.cos(
        torch.tensor((7 * torch.pi * step) / (16 * NUM_TRAINING_STEPS))
    )
    return LEARNING_RATE * multiplier

# K = 2**20
# K = total training steps, k = Current training step
# LR Schedule: INIT_LR * cos((7 * pi * k) / (16 * K))


# weak_transforms = transforms.Compose([
#     transforms.ToPILImage(),
#     transforms.RandomHorizontalFlip(p=0.5),
#     transforms.RandomResizedCrop(size=(IMAGE_SIZE, IMAGE_SIZE), scale=(0.8, 1.0), interpolation=transforms.InterpolationMode.BICUBIC),
#     transforms.RandomAffine(degrees=30, translate=(0.125, 0.125), scale=(1, 1), shear=None),
#     transforms.ToTensor(),
#     transforms.Normalize(
#         mean =[0.4914 , 0.4822 , 0.4465] , 
#         std=[0.2470 , 0.2435 , 0.2616]
#     ),
# ])


# strong_transforms = transforms.Compose([
#     transforms.ToPILImage(),
#     transforms.RandomHorizontalFlip(p=0.5),
#     transforms.RandomResizedCrop(size=(IMAGE_SIZE, IMAGE_SIZE), scale=(0.8, 1.0), interpolation=transforms.InterpolationMode.BICUBIC),
#     transforms.ColorJitter(
#         brightness=0.4,
#         contrast=0.4,
#         saturation=0.4,
#         hue=0.1
#     ),
#     transforms.RandAugment(num_ops=3, magnitude=7),
#     transforms.ToTensor(),
#     transforms.RandomErasing(p=0.4),
#     transforms.Normalize(
#         mean =[0.4914 , 0.4822 , 0.4465] , 
#         std=[0.2470 , 0.2435 , 0.2616]
#     ),
# ])

weak_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomCrop(96, padding=12),  # STL-10 is 96x96
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

strong_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomCrop(96, padding=12),
    transforms.RandAugment(num_ops=2, magnitude=10),  # May need to tune magnitude
    transforms.ToTensor(),
    transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0.5),  # CutOut
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


val_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize(
        mean =[0.4914 , 0.4822 , 0.4465] , 
        std=[0.2470 , 0.2435 , 0.2616]
    ),
])
