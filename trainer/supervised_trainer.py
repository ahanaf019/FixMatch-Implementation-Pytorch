import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from torch.utils.data import DataLoader
from torchmetrics.classification import MulticlassAccuracy
from itertools import cycle
from tqdm.auto import tqdm
from typing import Callable

from utils.utils import *

class SupervisedTrainer():
    def __init__(
            self,
            model: nn.Module,
            optim: torch.optim.Optimizer,
            loss_fn: Callable,
            num_classes: int,
            device: str='cuda'


        ):
        self.model = model
        self.optim = optim
        self.loss_fn = loss_fn
        self.num_classes = num_classes
        self.device = device
        self.scaler = torch.amp.GradScaler(device=device)
        self.acc_metric = MulticlassAccuracy(num_classes=self.num_classes).to(self.device)

    

    def fit(self, train_loader, val_loader, num_epochs, scheduler: torch.optim.lr_scheduler.LRScheduler):
        writer = SummaryWriter(log_dir=f"tensorboard/{self.model.__class__.__name__}/supervised")

        # train_loader = infinite_iterator(train_loader)
        val_losses = [np.inf]
        for epoch in range(num_epochs):
            loss = self.train_epoch(train_loader)
            writer.add_scalar("Train/train_loss", loss, epoch)
            writer.add_scalar("LearningRate", scheduler.get_last_lr()[-1], epoch)
            scheduler.step()

            val_loss, val_acc = self.validation_epoch(val_loader)
            writer.add_scalar("Validation/Loss", val_loss, epoch)
            writer.add_scalar("Validation/Accuracy", val_acc, epoch)
            print(f" Epoch [{epoch +1}/{num_epochs}] , Loss : {loss:0.4f} | val_loss: {np.mean(val_loss):0.4f} | Accuracy: {val_acc:0.4f} | lr: {scheduler.get_last_lr()[-1]:1.4e}")
            torch.cuda.empty_cache()

            if val_loss < min(val_losses):
                print(f'val_loss improved from {np.min(val_losses):0.4f} to {val_loss:0.4f}')
                save_state(
                    f'checkpoints/{self.model.__class__.__name__}.pt', 
                    self.model, 
                    self.optim,
                    info={
                        'epoch': epoch,
                        'val_loss': val_loss,
                })
            val_losses.append(val_loss)
        save_state(
                    f'checkpoints/{self.model.__class__.__name__}.pt', 
                    self.model, 
                    self.optim,
                    info={
                        'epoch': epoch,
                        'val_loss': val_loss,
                })


    def train_epoch(self, train_loader):
        self.model.train()
        for images, labels in tqdm(train_loader):
            images, labels = images.to(self.device), labels.to(self.device)
            self.optim.zero_grad()

            with torch.autocast(self.device, dtype=torch.float16):
                labelled_preds = self.model(images)
                loss = self.loss_fn(labelled_preds, labels)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optim)
            self.scaler.update()
        return loss.item()

    def validation_epoch(self, val_loader):
        losses = []
        self.acc_metric.reset()
        self.model.eval()
        with torch.inference_mode():
            for images, labels in tqdm(val_loader):
                images, labels = images.to(self.device), labels.to(self.device)
                # print(images.shape)

                preds = self.model(images)
                loss = self.loss_fn(preds, labels)

                self.acc_metric.update(preds, labels)
                losses.append(loss.item())
        return np.mean(losses), self.acc_metric.compute().cpu().item()