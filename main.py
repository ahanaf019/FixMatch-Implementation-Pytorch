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
from trainer.fixmatch_trainer import FixMatchTrainer

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def main():
    labeled_images = sorted(glob(f'{DB_PATH_ROOT}/train/*/*'))
    unlabeled_images = sorted(glob(f'{DB_PATH_ROOT}/unlabeled/*')[:NUM_UNLABELED])
    print(len(labeled_images), len(unlabeled_images))
    labels = [int(x.split('/')[-2]) for x in labeled_images]
    images_train, images_val, labels_train, labels_val = train_test_split(labeled_images, labels, test_size=0.1, random_state=224)

    train_db = SSDataset(
        image_paths=images_train,
        labels=labels_train, 
        image_size=IMAGE_SIZE,
        weak_augments=weak_transforms,
        strong_augments=strong_transforms
    )

    val_db = SSDataset(
        image_paths=images_val,
        labels=labels_val, 
        image_size=IMAGE_SIZE,
        weak_augments=val_transforms,
        strong_augments=val_transforms
    )

    unlabeled_db = SSDataset(
        image_paths=unlabeled_images,
        labels=None,
        image_size=IMAGE_SIZE, 
        weak_augments=weak_transforms,
        strong_augments=strong_transforms
    )
    
    train_loader = DataLoader(train_db, batch_size=BATCH_SIZE, shuffle=True, num_workers=os.cpu_count())
    val_loader = DataLoader(val_db, batch_size=BATCH_SIZE, shuffle=False, num_workers=os.cpu_count())
    unlabeled_loader = DataLoader(unlabeled_db, batch_size=BATCH_SIZE, shuffle=True, num_workers=os.cpu_count())

    model = CNNModel(dropout_rate=0.40, num_classes=10).to(device)
    # model.apply(init_weights_he)
    summary(model, input_size=[1, 3, 96, 96])

    optim = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9, weight_decay=5e-4, nesterov=True)
    # optim = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=5e-4)
    loss_fn = nn.CrossEntropyLoss()
    
    # lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=custom_lr_schedule)
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optim, start_factor=0.01, end_factor=1.0, total_iters=WARMUP_STEPS)
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=NUM_TRAINING_STEPS - WARMUP_STEPS, eta_min=0)
    # Combine both schedulers sequentially
    lr_scheduler = torch.optim.lr_scheduler.SequentialLR(optim, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[WARMUP_STEPS])
    
    trainer = FixMatchTrainer(model, optim, loss_fn, NUM_CLASSES, device)
    trainer.fit(train_loader, unlabeled_loader, val_loader, NUM_TRAINING_STEPS, THRESHOLD, LAMBDA_U, lr_scheduler)


def init_weights_he(m):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

if __name__ == '__main__':
    main()