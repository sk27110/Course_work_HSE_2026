# training/logger.py
import os
import csv
from datetime import datetime
from typing import Dict, Any

class LocalLogger:
    def __init__(self, log_dir: str, experiment_name: str):
        self.experiment_dir = os.path.join(log_dir, experiment_name)
        os.makedirs(self.experiment_dir, exist_ok=True)

        # Имена файлов с префиксом эксперимента
        self.metrics_path = os.path.join(self.experiment_dir, f"{experiment_name}_metrics.csv")
        self.log_file = os.path.join(self.experiment_dir, f"{experiment_name}.log")

        self._init_csv()
        self._init_log_file()

    def _init_csv(self):
        with open(self.metrics_path, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "train_loss", "train_acc", "train_top5",
                             "val_loss", "val_acc", "val_top5"])

    def _init_log_file(self):
        with open(self.log_file, 'w') as f:
            f.write(f"Experiment started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 50 + "\n")

    def log_config(self, config: Dict[str, Any]):
        with open(self.log_file, 'a') as f:
            f.write("Configuration:\n")
            for key, value in config.items():
                f.write(f"  {key}: {value}\n")
            f.write("-" * 50 + "\n")

    def log_message(self, message: str):
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        with open(self.log_file, 'a') as f:
            f.write(f"[{timestamp}] {message}\n")

    def log_epoch(self, epoch: int,
                  train_loss: float, train_acc: float, train_top5: float,
                  val_loss: float, val_acc: float, val_top5: float):
        # CSV
        with open(self.metrics_path, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, train_loss, train_acc, train_top5,
                             val_loss, val_acc, val_top5])
        # Текстовый лог
        msg = (f"Epoch {epoch:3d} | "
               f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} Top5: {train_top5:.4f} | "
               f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} Top5: {val_top5:.4f}")
        self.log_message(msg)

# CometLogger – без изменений (приводится для полноты)
class CometLogger:
    def __init__(self, project_name: str, experiment_name: str, api_key: str = None,
                 workspace: str = None, disabled: bool = False):
        self.disabled = disabled
        if not disabled:
            from comet_ml import Experiment
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

    def log_epoch_metrics(self, epoch, train_loss, train_acc, train_top5,
                          val_loss, val_acc, val_top5):
        if self.experiment is not None:
            self.experiment.log_metrics({
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "train_top5_accuracy": train_top5,
                "val_loss": val_loss,
                "val_accuracy": val_acc,
                "val_top5_accuracy": val_top5,
            }, epoch=epoch)

    def log_batch_loss(self, phase, batch_idx, epoch, loss):
        if self.experiment is not None:
            step = epoch * 10000 + batch_idx
            self.experiment.log_metric(f"{phase}_batch_loss", loss, step=step)

    def end(self):
        if self.experiment is not None:
            self.experiment.end()