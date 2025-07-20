import os

import torch.optim.sgd
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

from glob import glob
import shutil
from tqdm.auto import tqdm
from itertools import cycle

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
from torchinfo import summary
from sklearn.model_selection import train_test_split
from torchmetrics.classification import MulticlassAccuracy

from config import *
from datasets.semisupervised_dataset import SSDataset
from models.cnn_model import CNNModel
from models.wide_resnet import WideResNet
from trainer.fixmatch_trainer import FixMatchTrainer
from trainer.supervised_trainer import SupervisedTrainer
from utils.utils import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def main():
    labeled_images = sorted(glob(f'{DB_PATH_ROOT}/test/*/*'))
    labels = [int(x.split('/')[-2]) for x in labeled_images]

    val_db = SSDataset(
        image_paths=labeled_images,
        labels=labels, 
        image_size=IMAGE_SIZE,
        transforms_1=val_transforms,
        transforms_2=val_transforms
    )
    
    test_loader = DataLoader(val_db, batch_size=BATCH_SIZE, shuffle=False, num_workers=os.cpu_count())
    print(next(iter(test_loader))[0].shape)
    print(IMAGE_SIZE)

    model = WideResNet(in_channels=3, base_filters=16, k=4, N=6, num_classes=NUM_CLASSES, dropout_rate=0.1).to(device)
    summary(model, input_size=[1, 3, IMAGE_SIZE, IMAGE_SIZE])

    optim = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9, weight_decay=5e-4, nesterov=True)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optim, milestones=MILESTONES, gamma=GAMMA)
    model, _ = load_state('./checkpoints/WideResNet_96.pt', model, None)
    # trainer = FixMatchTrainer(model, optim, loss_fn, NUM_CLASSES, device)
    trainer = SupervisedTrainer(model, optim, loss_fn, NUM_CLASSES, device)
    print(trainer.validation_epoch(test_loader))


def init_weights_he(m):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

if __name__ == '__main__':
    main()