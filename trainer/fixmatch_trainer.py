import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from torchmetrics.classification import MulticlassAccuracy
from itertools import cycle
from tqdm.auto import tqdm
from typing import Callable


class FixMatchTrainer():
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
    

    def fit(self, train_loader, unlabeled_loader, val_loader, num_epochs, num_warmup_epochs, confidence_threshold, lambda_u):
        self.acc_metric = MulticlassAccuracy(num_classes=self.num_classes).to(self.device)
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer=self.optim, 
            schedulers=[
                torch.optim.lr_scheduler.LinearLR(
                    self.optim, start_factor=0.001, end_factor=1, total_iters=num_warmup_epochs 
                ),
                torch.optim.lr_scheduler.CosineAnnealingLR(
                    self.optim, T_max=num_epochs - num_warmup_epochs
                )
            ], milestones=[num_warmup_epochs])
        

        for epoch in range(num_epochs):
            loss = self.train_epoch(train_loader, unlabeled_loader, confidence_threshold, lambda_u)
            val_loss, val_acc = self.validation_epoch(val_loader)
            scheduler.step()
            print(f" Epoch [{epoch +1}/{num_epochs}] , Loss : {loss:0.4f} | val_loss: {np.mean(val_loss):0.4f} | Accuracy: {val_acc:0.4f}")


    def train_epoch(self, train_loader, unlabeled_loader, confidence_threshold, lambda_u):
        self.model.train()
        losses = []
        for (labeled_images, labels), (weak_img, strong_img) in tqdm(zip(cycle(train_loader), unlabeled_loader)):
            labeled_images, labels = labeled_images.to(self.device), labels.to(self.device)
            weak_img, strong_img = weak_img.to(self.device), strong_img.to(self.device)

            labelled_preds = self.model(labeled_images)
            loss_labelled = self.loss_fn(labelled_preds, labels)

            with torch.no_grad():
                weak_preds = self.model(weak_img)
                pseudo_labels = torch.softmax(weak_preds, dim=1)
                max_probs, targets_u = pseudo_labels.max(dim=1)
            
            mask = max_probs > confidence_threshold

            strong_preds = self.model(strong_img)
            loss_unlabeled_all = F.cross_entropy(strong_preds, targets_u, reduction='none')
            loss_unlabeled = (loss_unlabeled_all* mask).mean()

            loss = loss_labelled + lambda_u * loss_unlabeled
            losses.append(loss.item())

            self.optim.zero_grad()
            loss.backward()
            self.optim.step()
        return np.mean(losses)


    def validation_epoch(self, val_loader):
        losses = []
        self.acc_metric.reset()
        self.model.eval()
        for images, labels in val_loader:
            images, labels = images.to(self.device), labels.to(self.device)

            with torch.inference_mode():
                preds = self.model(images)
                loss = self.loss_fn(preds, labels)

            self.acc_metric.update(preds, labels)
            losses.append(loss.item())
        return np.mean(losses), self.acc_metric.compute().cpu().item()