# training/logger.py
import os
import csv
from comet_ml import Experiment

class LocalLogger:
    """Локальное сохранение метрик эпох в CSV-файл."""
    def __init__(self, log_dir: str, experiment_name: str):
        self.log_dir = os.path.join(log_dir, experiment_name)
        os.makedirs(self.log_dir, exist_ok=True)
        self.metrics_path = os.path.join(self.log_dir, "metrics.csv")
        self._init_csv()

    def _init_csv(self):
        with open(self.metrics_path, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "train_loss", "train_acc", "train_top5",
                             "val_loss", "val_acc", "val_top5"])

    def log_epoch(self, epoch: int,
                  train_loss: float, train_acc: float, train_top5: float,
                  val_loss: float, val_acc: float, val_top5: float):
        with open(self.metrics_path, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, train_loss, train_acc, train_top5,
                             val_loss, val_acc, val_top5])


class CometLogger:
    """Логгер для Comet ML: параметры эксперимента, метрики по батчам и эпохам."""
    def __init__(self, project_name: str, experiment_name: str, api_key: str = None,
                 workspace: str = None, disabled: bool = False):
        self.disabled = disabled
        if not disabled:
            self.experiment = Experiment(
                api_key=api_key,
                project_name=project_name,
                workspace=workspace,
            )
            self.experiment.set_name(experiment_name)
        else:
            self.experiment = None

    def log_parameters(self, params: dict):
        if self.experiment is not None:
            self.experiment.log_parameters(params)

    def log_metric(self, name: str, value, step: int = None):
        if self.experiment is not None:
            self.experiment.log_metric(name, value, step=step)

    def log_epoch_metrics(self, epoch: int, train_loss: float, train_acc: float,
                          train_top5: float, val_loss: float, val_acc: float,
                          val_top5: float):
        if self.experiment is not None:
            self.experiment.log_metrics({
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "train_top5_accuracy": train_top5,
                "val_loss": val_loss,
                "val_accuracy": val_acc,
                "val_top5_accuracy": val_top5,
            }, epoch=epoch)

    def log_batch_loss(self, phase: str, batch_idx: int, epoch: int, loss: float):
        """Потери по каждому батчу (train / val)"""
        if self.experiment is not None:
            step = epoch * 10000 + batch_idx  # условный глобальный шаг
            self.experiment.log_metric(f"{phase}_batch_loss", loss, step=step)

    def end(self):
        if self.experiment is not None:
            self.experiment.end()