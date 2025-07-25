import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_ema import ExponentialMovingAverage
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
        self.ema = ExponentialMovingAverage(model.parameters(), 0.999)
    

    def fit(self, labeled_loader, unlabeled_loader, val_loader, num_total_steps, u_ratio, confidence_threshold, lambda_u, scheduler: torch.optim.lr_scheduler.LRScheduler):
        self.acc_metric = MulticlassAccuracy(num_classes=self.num_classes).to(self.device)
        writer = SummaryWriter(log_dir=f"tensorboard/{self.model.__class__.__name__}/fixmatch")

        labeled_iter = infinite_iterator(labeled_loader)
        unlabeled_iter = infinite_iterator(unlabeled_loader)
        val_losses = [np.inf]

        for step in tqdm(range(num_total_steps)):
            self.model.train()
            self.optim.zero_grad(set_to_none=True)
            loss_unlabeled_total = 0.0
            percent_over_thresholds = []

            # ==== Labeled Loss (1 batch) ====
            labeled_images, labels = next(labeled_iter)
            labeled_images, labels = labeled_images.to(self.device), labels.to(self.device)
            with torch.autocast(self.device, dtype=torch.float16):
                labeled_preds = self.model(labeled_images)
                loss_labeled = self.loss_fn(labeled_preds, labels)
            self.scaler.scale(loss_labeled).backward()

            # ==== Unlabeled Gradient Accumulation ====
            for _ in range(u_ratio):
                weak_img, strong_img = next(unlabeled_iter)
                weak_img, strong_img = weak_img.to(self.device), strong_img.to(self.device)

                # EMA teacher predictions (no gradients)
                self.ema.store()
                self.model.eval()
                self.ema.copy_to(self.model.parameters())
                with torch.no_grad():
                    weak_preds = self.model(weak_img)
                    pseudo_labels = torch.softmax(weak_preds, dim=1)
                    max_probs, targets_u = pseudo_labels.max(dim=1)
                self.ema.restore()

                mask = max_probs > confidence_threshold
                percent_over_thresholds.append(mask.float().mean().item())

                # Student training on strong augmented images
                self.model.train()
                with torch.autocast(self.device, dtype=torch.float16):
                    strong_preds = self.model(strong_img)
                    loss_unlabeled_all = F.cross_entropy(strong_preds, targets_u, reduction='none')
                    loss_unlabeled = (loss_unlabeled_all * mask).sum() / mask.shape[0]

                    # Accumulate scaled loss
                    loss_unlabeled_scaled = lambda_u * loss_unlabeled / u_ratio
                    # print(loss_unlabeled_scaled, loss_unlabeled)
                self.scaler.scale(loss_unlabeled_scaled).backward()
                loss_unlabeled_total += loss_unlabeled_scaled.detach()

            loss = (loss_unlabeled_total + loss_labeled).detach().item()

            # ==== Optimizer step ====
            self.scaler.step(self.optim)
            self.scaler.update()
            self.optim.zero_grad(set_to_none=True)
            self.ema.update()
            scheduler.step()
            
            if step % 50 == 0:
                writer.add_scalar("Train/total_loss", loss, step)
                writer.add_scalar("Train/labelled_loss", loss_labeled, step)
                writer.add_scalar("Train/unlabelled_loss", loss_unlabeled_total, step)
                writer.add_scalar("Train/Unlabeled_Over_Thres", np.mean(percent_over_thresholds), step)
                writer.add_scalar("LearningRate", scheduler.get_last_lr()[-1], step)

            if step % 250 == 0:
                val_loss, val_acc = self.validation_epoch(val_loader)
                writer.add_scalar("Validation/Loss", val_loss, step)
                writer.add_scalar("Validation/Accuracy", val_acc, step)
                print(f" Step [{step +1}/{num_total_steps}] , Loss : {loss:0.4f} | val_loss: {np.mean(val_loss):0.4f} | Accuracy: {val_acc:0.4f} | Err_Rate: {(1 - val_acc):0.4f} | lr: {scheduler.get_last_lr()[-1]:1.4e}")
                torch.cuda.empty_cache()
                if val_loss < min(val_losses):
                    print(f'val_loss improved from {np.min(val_losses):0.4f} to {val_loss:0.4f}')
                    save_state(
                        f'checkpoints/fixmatch/{self.model.__class__.__name__}.pt', 
                        self.model, 
                        self.optim,
                        info={
                            'step': step,
                            'val_loss': val_loss,
                    })
                val_losses.append(val_loss)


    def validation_epoch(self, val_loader):
        losses = []
        self.acc_metric.reset()
        self.model.eval()
        self.ema.store()
        self.ema.copy_to(self.model.parameters())
        for images, labels in val_loader:
            images, labels = images.to(self.device), labels.to(self.device)

            with torch.inference_mode():
                preds = self.model(images)
                loss = self.loss_fn(preds, labels)

            self.acc_metric.update(preds, labels)
            losses.append(loss.item())
        self.ema.restore()
        return np.mean(losses), self.acc_metric.compute().cpu().item()