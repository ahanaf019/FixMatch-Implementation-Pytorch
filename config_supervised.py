import os
import torch
from torchvision import transforms

DB_PATH_ROOT = f'{os.environ["HOME"]}/Datasets/CIFAR-10'
BATCH_SIZE = 128
IMAGE_SIZE = 32
WARMUP_EPOCHS = 5
LEARNING_RATE = 1e-2
NUM_EPOCHS = 200
MILESTONES = [60, 120, 160]
GAMMA = 0.2

NUM_CLASSES = 10
# NUM_UNLABELED = 100000
# THRESHOLD = 0.90
# LAMBDA_U = 1.0

IMAGE_CHANNEL_MEANS = [0.4914, 0.4822, 0.4465]
IMAGE_CHANNEL_STD = [0.2023, 0.1994, 0.2010]


def custom_lr_schedule(step):
    multiplier = torch.cos(
        torch.tensor((7 * torch.pi * step) / (16 * NUM_EPOCHS))
    )
    return LEARNING_RATE * multiplier

# K = 2**20
# K = total training steps, k = Current training step
# LR Schedule: INIT_LR * cos((7 * pi * k) / (16 * K))


weak_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomCrop(size=IMAGE_SIZE, padding=4, padding_mode='reflect'),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGE_CHANNEL_MEANS, std=IMAGE_CHANNEL_STD)
])

strong_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomResizedCrop(size=IMAGE_SIZE, scale=(0.8, 1)),
    transforms.RandAugment(num_ops=3, magnitude=7),  # May need to tune magnitude
    # transforms.RandomAffine(degrees=30, translate=(0.15, 0.15), scale=(0.2, 0.2), shear=(0.1, 0.1)),
    transforms.ToTensor(),
    transforms.RandomErasing(p=0.3, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0.5),  # CutOut
    transforms.Normalize(mean=IMAGE_CHANNEL_MEANS, std=IMAGE_CHANNEL_STD)
])


val_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize(mean =IMAGE_CHANNEL_MEANS, std=IMAGE_CHANNEL_STD),
])
