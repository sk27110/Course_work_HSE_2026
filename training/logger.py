# training/logger.py

import os
import csv
from datetime import datetime
from typing import Dict, Any, Optional


class LocalLogger:
    def __init__(self, log_dir: str, experiment_name: str):
        self.experiment_dir = os.path.join(log_dir, experiment_name)
        os.makedirs(self.experiment_dir, exist_ok=True)

        self.metrics_path = os.path.join(
            self.experiment_dir,
            f"{experiment_name}_metrics.csv"
        )
        self.log_file = os.path.join(
            self.experiment_dir,
            f"{experiment_name}.log"
        )

        self._init_csv()
        self._init_log_file()

    def _init_csv(self):
        with open(self.metrics_path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "epoch",
                "train_loss",
                "train_acc",
                "train_top5",
                "val_loss",
                "val_acc",
                "val_top5"
            ])

    def _init_log_file(self):
        with open(self.log_file, "w", encoding="utf-8") as f:
            f.write(
                f"Experiment started: "
                f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            )
            f.write("=" * 50 + "\n")

    def log_config(self, config: Dict[str, Any]):
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write("Configuration:\n")
            for key, value in config.items():
                f.write(f"  {key}: {value}\n")
            f.write("-" * 50 + "\n")

    def log_message(self, message: str):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(f"[{timestamp}] {message}\n")

    def log_epoch(
        self,
        epoch: int,
        train_loss: float,
        train_acc: float,
        train_top5: float,
        val_loss: float,
        val_acc: float,
        val_top5: float
    ):
        with open(self.metrics_path, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch,
                train_loss,
                train_acc,
                train_top5,
                val_loss,
                val_acc,
                val_top5
            ])

        msg = (
            f"Epoch {epoch:3d} | "
            f"Train Loss: {train_loss:.4f} "
            f"Acc: {train_acc:.4f} "
            f"Top5: {train_top5:.4f} | "
            f"Val Loss: {val_loss:.4f} "
            f"Acc: {val_acc:.4f} "
            f"Top5: {val_top5:.4f}"
        )

        self.log_message(msg)


class CometLogger:
    def __init__(
        self,
        project_name: str,
        experiment_name: str,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
        disabled: bool = False
    ):
        self.disabled = disabled
        self.experiment = None

        if self.disabled:
            print("[CometLogger] Disabled.")
            return

        from comet_ml import Experiment

        self.experiment = Experiment(
            api_key=api_key,
            project_name=project_name,
            workspace=workspace,
        )

        self.experiment.set_name(experiment_name)

        print("[CometLogger] Experiment created.")
        print(f"[CometLogger] project_name={project_name}")
        print(f"[CometLogger] experiment_name={experiment_name}")
        print(f"[CometLogger] workspace={workspace}")

    def log_parameters(self, params: dict):
        if self.experiment is None:
            return

        clean_params = {}

        for key, value in params.items():
            if isinstance(value, (str, int, float, bool)) or value is None:
                clean_params[key] = value
            else:
                clean_params[key] = str(value)

        self.experiment.log_parameters(clean_params)

    def log_metric(self, name: str, value, step: Optional[int] = None):
        if self.experiment is None:
            return

        self.experiment.log_metric(
            name,
            value,
            step=step
        )

    def log_epoch_metrics(
        self,
        epoch: int,
        train_loss: float,
        train_acc: float,
        train_top5: float,
        val_loss: float,
        val_acc: float,
        val_top5: float
    ):
        if self.experiment is None:
            return

        metrics = {
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "train_top5_accuracy": train_top5,
            "val_loss": val_loss,
            "val_accuracy": val_acc,
            "val_top5_accuracy": val_top5,
        }

        # ВАЖНО:
        # step нужен для нормального построения графиков.
        self.experiment.log_metrics(
            metrics,
            step=epoch,
            epoch=epoch
        )

    def log_batch_loss(
        self,
        phase: str,
        batch_idx: int,
        epoch: int,
        loss: float
    ):
        if self.experiment is None:
            return

        # epoch у тебя начинается с 1.
        # batch_idx начинается с 0.
        # 10000 — просто большой множитель, чтобы step не пересекался между эпохами.
        step = (epoch - 1) * 10000 + batch_idx

        self.experiment.log_metric(
            f"{phase}_batch_loss",
            loss,
            step=step,
            epoch=epoch
        )

    def end(self):
        if self.experiment is None:
            return

        self.experiment.end()
