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
        self.scaler = torch.amp.GradScaler(device=device)
    

    def fit(self, labeled_loader, unlabeled_loader, val_loader, num_total_steps, confidence_threshold, lambda_u, scheduler: torch.optim.lr_scheduler.LRScheduler):
        self.acc_metric = MulticlassAccuracy(num_classes=self.num_classes).to(self.device)
        writer = SummaryWriter(log_dir=f"tensorboard/{self.model.__class__.__name__}/experiment")

        labeled_iter = infinite_iterator(labeled_loader)
        unlabeled_iter = infinite_iterator(unlabeled_loader)

        for step in range(num_total_steps):
            labeled_images, labels = next(labeled_iter)
            weak_img, strong_img = next(unlabeled_iter)

            labeled_images, labels = labeled_images.to(self.device), labels.to(self.device)
            weak_img, strong_img = weak_img.to(self.device), strong_img.to(self.device)
            loss, loss_labelled, loss_unlabeled, percent_over_threshold = self.train_step(labeled_images, labels, weak_img, strong_img, confidence_threshold, lambda_u)
            if step % 100 == 0:
                writer.add_scalar("Train/total_loss", loss, step)
                writer.add_scalar("Train/labelled_loss", loss_labelled, step)
                writer.add_scalar("Train/unlabelled_loss", loss_unlabeled, step)
                writer.add_scalar("Train/Unlabeled_Over_Thres", percent_over_threshold, step)
                writer.add_scalar("LearningRate", scheduler.get_last_lr()[-1], step)
            scheduler.step()

            if step % 500 == 0:
                val_loss, val_acc = self.validation_epoch(val_loader)
                writer.add_scalar("Validation/Loss", val_loss, step)
                writer.add_scalar("Validation/Accuracy", val_acc, step)
                print(f" Epoch [{step +1}/{num_total_steps}] , Loss : {loss:0.4f} | val_loss: {np.mean(val_loss):0.4f} | Accuracy: {val_acc:0.4f} | lr: {scheduler.get_last_lr()[-1]:1.4e}")
                torch.cuda.empty_cache()
                # print(f" Epoch [{step +1}/{num_total_steps}], lr: {scheduler.get_last_lr()[-1]:1.4e}")


    def train_step(self, labeled_images, labels, weak_img, strong_img, confidence_threshold, lambda_u):
        self.model.train()
        self.optim.zero_grad()

        with torch.autocast(self.device, dtype=torch.float16):
            labelled_preds = self.model(labeled_images)
            loss_labelled = self.loss_fn(labelled_preds, labels)

        self.model.eval()
        with torch.no_grad():
            weak_preds = self.model(weak_img)
            pseudo_labels = torch.softmax(weak_preds, dim=1)
            max_probs, targets_u = pseudo_labels.max(dim=1)
        
        mask = max_probs > confidence_threshold
        percent_over_threshold = mask.sum() / mask.shape[0]

        self.model.train()
        with torch.autocast(self.device, dtype=torch.float16):
            strong_preds = self.model(strong_img)
            loss_unlabeled_all = F.cross_entropy(strong_preds, targets_u, reduction='none')
            loss_unlabeled = (loss_unlabeled_all* mask).mean()
            loss = loss_labelled + lambda_u * loss_unlabeled
        
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optim)
        self.scaler.update()
        return loss.item(), loss_labelled.item(), loss_unlabeled.item(), percent_over_threshold


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