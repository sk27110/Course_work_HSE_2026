# run_experiment.py

import argparse
import os
import random
import shutil
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

from data_utils.dataset_factory import create_datasets
from training.logger import CometLogger
from training.models import get_resnet18, get_vit_tiny
from training.trainer import Trainer, set_seed


def load_config(config_path: str) -> Dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def create_model(model_cfg: Dict, num_classes: int) -> nn.Module:
    model_name = model_cfg["name"].lower()
    pretrained = bool(model_cfg.get("pretrained", False))

    if model_name == "resnet18":
        return get_resnet18(
            num_classes=num_classes,
            pretrained=pretrained,
        )

    if model_name == "vit_tiny":
        return get_vit_tiny(
            num_classes=num_classes,
            pretrained=pretrained,
        )

    raise ValueError(f"Unknown model: {model_name}")


def create_optimizer(model: nn.Module, train_cfg: Dict) -> torch.optim.Optimizer:
    opt_name = train_cfg["optimizer"].lower()
    lr = float(train_cfg["learning_rate"])
    wd = float(train_cfg.get("weight_decay", 0.0))

    if opt_name == "sgd":
        momentum = float(train_cfg.get("momentum", 0.9))
        return torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=momentum,
            weight_decay=wd,
        )

    if opt_name == "adam":
        return torch.optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=wd,
        )

    if opt_name == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=wd,
        )

    raise ValueError(f"Unsupported optimizer: {opt_name}")


def create_scheduler(
    optimizer: torch.optim.Optimizer,
    train_cfg: Dict,
):
    sched_cfg = train_cfg.get("scheduler", {})
    sched_type = sched_cfg.get("type", "none").lower()

    if sched_type == "step_lr":
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=int(sched_cfg.get("step_size", 30)),
            gamma=float(sched_cfg.get("gamma", 0.1)),
        )

    if sched_type == "reduce_on_plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            patience=int(sched_cfg.get("patience", 5)),
            factor=float(sched_cfg.get("factor", 0.5)),
        )

    if sched_type == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=int(train_cfg["num_epochs"]),
        )

    if sched_type == "none":
        return None

    raise ValueError(f"Unsupported scheduler: {sched_type}")


def seed_worker(worker_id: int):
    """
    Makes dataloader workers deterministic enough for common experiments.
    """

    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def create_dataloaders(
    train_dataset,
    val_dataset,
    data_cfg: Dict,
    device: torch.device,
    seed: int,
):
    batch_size = int(data_cfg["batch_size"])
    num_workers = int(data_cfg.get("num_workers", 0))
    pin_memory = device.type == "cuda"

    generator = torch.Generator()
    generator.manual_seed(seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=seed_worker,
        generator=generator,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=seed_worker,
        generator=generator,
    )

    return train_loader, val_loader


def resolve_device(requested_device: str) -> torch.device:
    requested_device = requested_device.lower()

    if requested_device == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")

    if requested_device == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def create_comet_logger(
    log_cfg: Dict,
    experiment_name: str,
) -> CometLogger:
    comet_cfg = log_cfg.get("comet", {})

    api_key = comet_cfg.get("api_key", None)

    api_key_env = comet_cfg.get("api_key_env", None)
    if api_key_env is not None:
        api_key = os.getenv(api_key_env)

    return CometLogger(
        project_name=comet_cfg.get("project_name"),
        experiment_name=experiment_name,
        api_key=api_key,
        workspace=comet_cfg.get("workspace"),
        disabled=bool(comet_cfg.get("disabled", False)),
    )


def log_experiment_parameters(
    comet_logger: CometLogger,
    config: Dict,
    train_dataset,
    val_dataset,
    train_loader,
    val_loader,
):
    data_cfg = config["data"]
    model_cfg = config["model"]
    train_cfg = config["training"]
    scheduler_cfg = train_cfg.get("scheduler", {"type": "none"})
    few_shot_cfg = data_cfg.get("few_shot", {})

    train_batches_per_epoch = len(train_loader)
    val_batches_per_epoch = len(val_loader)
    total_optimizer_steps = train_batches_per_epoch * int(train_cfg["num_epochs"])

    comet_logger.log_parameters({
        "experiment_name": config["experiment_name"],
        "seed": config["seed"],

        "dataset_name": data_cfg.get("dataset_name", "flowers102"),
        "num_classes": data_cfg["num_classes"],

        "train_dataset_type": data_cfg.get("train_dataset_type", "original"),
        "train_dataset_size": len(train_dataset),
        "val_dataset_size": len(val_dataset),

        "few_shot_enabled": few_shot_cfg.get("enabled", False),
        "few_shot_k": few_shot_cfg.get("k", None),
        "few_shot_seed": few_shot_cfg.get("seed", config["seed"]),
        "few_shot_indices_path": few_shot_cfg.get("indices_path", None),

        "aug_root": data_cfg.get("aug_root", None),
        "aug_index_path": data_cfg.get("aug_index_path", None),
        "num_aug_per_image": data_cfg.get("num_aug_per_image", None),
        "alpha": data_cfg.get("alpha", None),

        "batch_size": data_cfg["batch_size"],
        "num_workers": data_cfg.get("num_workers", 0),
        "train_batches_per_epoch": train_batches_per_epoch,
        "val_batches_per_epoch": val_batches_per_epoch,
        "total_optimizer_steps": total_optimizer_steps,

        "model": model_cfg["name"],
        "pretrained": model_cfg.get("pretrained", False),

        "optimizer": train_cfg["optimizer"],
        "learning_rate": train_cfg["learning_rate"],
        "momentum": train_cfg.get("momentum", None),
        "weight_decay": train_cfg.get("weight_decay", 0),

        "scheduler": scheduler_cfg.get("type", "none"),
        "scheduler_params": str(scheduler_cfg),
        "num_epochs": train_cfg["num_epochs"],
    })


def print_experiment_summary(
    config: Dict,
    device: torch.device,
    train_dataset,
    val_dataset,
    train_loader,
    val_loader,
):
    data_cfg = config["data"]
    few_shot_cfg = data_cfg.get("few_shot", {})

    print("=" * 90)
    print(f"Experiment: {config['experiment_name']}")
    print(f"Device: {device}")
    print(f"Dataset: {data_cfg.get('dataset_name', 'flowers102')}")
    print(f"Train dataset type: {data_cfg.get('train_dataset_type', 'original')}")
    print(f"Few-shot enabled: {few_shot_cfg.get('enabled', False)}")
    print(f"Few-shot k: {few_shot_cfg.get('k', None)}")
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Val dataset size: {len(val_dataset)}")
    print(f"Train batches per epoch: {len(train_loader)}")
    print(f"Val batches per epoch: {len(val_loader)}")
    print("=" * 90)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config file.",
    )

    args = parser.parse_args()

    config = load_config(args.config)

    experiment_name = config["experiment_name"]
    seed = int(config["seed"])

    set_seed(seed)

    device = resolve_device(config.get("device", "cuda"))

    data_cfg = config["data"]
    model_cfg = config["model"]
    train_cfg = config["training"]
    log_cfg = config["logging"]

    num_classes = int(data_cfg["num_classes"])

    train_dataset, val_dataset = create_datasets(
        data_cfg=data_cfg,
        seed=seed,
    )

    train_loader, val_loader = create_dataloaders(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        data_cfg=data_cfg,
        device=device,
        seed=seed,
    )

    model = create_model(
        model_cfg=model_cfg,
        num_classes=num_classes,
    )

    criterion = nn.CrossEntropyLoss()

    optimizer = create_optimizer(
        model=model,
        train_cfg=train_cfg,
    )

    scheduler = create_scheduler(
        optimizer=optimizer,
        train_cfg=train_cfg,
    )

    comet_logger = create_comet_logger(
        log_cfg=log_cfg,
        experiment_name=experiment_name,
    )

    print_experiment_summary(
        config=config,
        device=device,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        train_loader=train_loader,
        val_loader=val_loader,
    )

    log_experiment_parameters(
        comet_logger=comet_logger,
        config=config,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        train_loader=train_loader,
        val_loader=val_loader,
    )

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        experiment_name=experiment_name,
        num_epochs=int(train_cfg["num_epochs"]),
        log_dir=log_cfg["log_dir"],
        comet_logger=comet_logger,
        seed=seed,
        log_batch_loss=bool(log_cfg.get("log_batch_loss", False)),
        config=config,
    )

    os.makedirs(trainer.local_logger.experiment_dir, exist_ok=True)

    shutil.copy(
        args.config,
        os.path.join(
            trainer.local_logger.experiment_dir,
            f"{experiment_name}_config.yaml",
        ),
    )

    trainer.fit()


if __name__ == "__main__":
    main()
