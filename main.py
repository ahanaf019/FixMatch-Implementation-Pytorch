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
        weak_augments=weak_transforms
    )

    val_db = SSDataset(
        image_paths=images_val,
        labels=labels_val, 
        image_size=IMAGE_SIZE,
        weak_augments=val_transforms
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
    acc_metric = MulticlassAccuracy(num_classes=NUM_CLASSES).to(device)
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer=optim, schedulers=[
            torch.optim.lr_scheduler.LinearLR(
                optim, start_factor=0.001, end_factor=1, total_iters=WARMUP_EPOCHS 
            ),
            torch.optim.lr_scheduler.CosineAnnealingLR(
                optim, T_max=NUM_EPOCHS - WARMUP_EPOCHS
            )
        ], milestones=[WARMUP_EPOCHS])
    
    for epoch in range(NUM_EPOCHS):
        model.train()
        losses = []
        val_losses = []

        for (labeled_images, labels), (weak_img, strong_img) in tqdm(zip(cycle(train_loader), unlabeled_loader)):
            labeled_images, labels = labeled_images.to(device), labels.to(device)

            weak_img, strong_img = weak_img.to(device), strong_img.to(device)


            labelled_preds = model(labeled_images)
            loss_labelled = loss_fn(labelled_preds, labels)

            # with torch.no_grad():
            #     weak_preds = model(weak_img)
            #     pseudo_labels = torch.softmax(weak_preds, dim=1)
            #     max_probs, targets_u = pseudo_labels.max(dim=1)
            
            # mask = max_probs > THRESHOLD

            # strong_preds = model(strong_img)
            # loss_unlabeled_all = F.cross_entropy(strong_preds, targets_u, reduction='none')
            # loss_unlabeled = (loss_unlabeled_all* mask).mean()

            loss = loss_labelled #+ LAMBDA_U * loss_unlabeled
            losses.append(loss.item())

            optim.zero_grad()
            loss.backward()
            optim.step()


        acc_metric.reset()
        model.eval()
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)

            with torch.inference_mode():
                labelled_preds = model(images)
                # print(labels.shape, labelled_preds.shape)
                loss_labelled = loss_fn(labelled_preds, labels)

            acc_metric.update(labelled_preds, labels)
            val_losses.append(loss_labelled.item())
        scheduler.step()
        print(f" Epoch [{ epoch +1}/{ NUM_EPOCHS }] , Loss : {np.mean(losses):0.4f} | val_loss: {np.mean(val_losses):0.4f} | Accuracy: {acc_metric.compute().cpu().item():0.4f}")


if __name__ == '__main__':
    main()