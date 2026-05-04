# training/trainer.py
import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Optional, Dict, Any
from .logger import LocalLogger, CometLogger

def set_seed(seed: int):
    """Обеспечивает воспроизводимость экспериментов."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
        device: torch.device,
        experiment_name: str,
        num_epochs: int = 50,
        log_dir: str = "logs",
        comet_logger: Optional[CometLogger] = None,
        seed: int = 42,
        save_best: bool = True,
        log_batch_loss: bool = True,
        config: Optional[Dict[str, Any]] = None   # новый параметр
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.experiment_name = experiment_name
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.num_epochs = num_epochs
        self.log_batch_loss = log_batch_loss
        self.config = config

        # Логгеры
        self.local_logger = LocalLogger(log_dir, experiment_name)
        self.comet_logger = comet_logger

        # Папка для чекпоинтов
        self.checkpoint_dir = os.path.join(log_dir, experiment_name, "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        self.best_val_acc = 0.0
        self.save_best = save_best

        set_seed(seed)

    def train_epoch(self, epoch: int) -> tuple:
        self.model.train()
        running_loss = 0.0
        correct1 = 0
        correct5 = 0
        total = 0

        for batch_idx, (inputs, targets) in enumerate(self.train_loader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.topk(5, dim=1)
            total += targets.size(0)
            correct1 += (predicted[:, 0] == targets).sum().item()
            correct5 += (predicted == targets.view(-1, 1)).sum().item()

            if self.log_batch_loss and self.comet_logger is not None:
                self.comet_logger.log_batch_loss("train", batch_idx, epoch, loss.item())

        train_loss = running_loss / total
        train_acc = correct1 / total
        train_top5 = correct5 / total
        return train_loss, train_acc, train_top5

    @torch.no_grad()
    def validate_epoch(self, epoch: int) -> tuple:
        self.model.eval()
        running_loss = 0.0
        correct1 = 0
        correct5 = 0
        total = 0

        for batch_idx, (inputs, targets) in enumerate(self.val_loader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.topk(5, dim=1)
            total += targets.size(0)
            correct1 += (predicted[:, 0] == targets).sum().item()
            correct5 += (predicted == targets.view(-1, 1)).sum().item()

            if self.log_batch_loss and self.comet_logger is not None:
                self.comet_logger.log_batch_loss("val", batch_idx, epoch, loss.item())

        val_loss = running_loss / total
        val_acc = correct1 / total
        val_top5 = correct5 / total
        return val_loss, val_acc, val_top5

    def fit(self):
        # Логируем конфиг
        if self.config is not None:
            self.local_logger.log_config(self.config)
        self.local_logger.log_message("Training started.")
        print(f"Training started for {self.num_epochs} epochs on device {self.device}")

        for epoch in range(1, self.num_epochs + 1):
            train_loss, train_acc, train_top5 = self.train_epoch(epoch)
            val_loss, val_acc, val_top5 = self.validate_epoch(epoch)

            self.local_logger.log_epoch(
                epoch, train_loss, train_acc, train_top5,
                val_loss, val_acc, val_top5
            )

            if self.comet_logger is not None:
                self.comet_logger.log_epoch_metrics(
                    epoch, train_loss, train_acc, train_top5,
                    val_loss, val_acc, val_top5
                )

            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

            if self.save_best and val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                checkpoint_path = os.path.join(self.checkpoint_dir, f"{self.experiment_name}_best.pth")
                torch.save(self.model.state_dict(), checkpoint_path)
                self.local_logger.log_message(f"Epoch {epoch}: new best model saved (val_acc {val_acc:.4f})")
                print(f"Epoch {epoch}: new best model saved with val_acc {val_acc:.4f}")

            print(
                f"Epoch [{epoch}/{self.num_epochs}] "
                f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} Top5: {train_top5:.4f} | "
                f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} Top5: {val_top5:.4f}"
            )

        if self.comet_logger is not None:
            self.comet_logger.end()
        self.local_logger.log_message("Training finished.")
        print("Training finished.")