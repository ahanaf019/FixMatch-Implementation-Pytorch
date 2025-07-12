import os
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
        strong_augments=strong_transforms
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


    model = CNNModel(num_classes=NUM_CLASSES, dropout_rate=0.25).to(device)
    summary(model, input_size=[1, 3, 96, 96])

    optim = torch.optim.AdamW(model.parameters(), lr=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    
    trainer = FixMatchTrainer(model, optim, loss_fn, NUM_CLASSES, device)
    trainer.fit(train_loader, unlabeled_loader, val_loader, NUM_EPOCHS, WARMUP_EPOCHS, THRESHOLD, LAMBDA_U)
    


if __name__ == '__main__':
    main()