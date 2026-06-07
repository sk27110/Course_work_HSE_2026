# run_experiment.py

import os
import yaml
import argparse
import shutil

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from training.models import get_resnet18, get_vit_tiny
from training.trainer import Trainer, set_seed
from training.logger import CometLogger

from local_datasets.oxford_flower102 import OxfordFlowers102
from local_datasets.dtd import DTD

from training.dataset import MixedAugDataset

from utils.transforms import train_transform, val_test_transform


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def create_optimizer(model: nn.Module, train_cfg: dict) -> torch.optim.Optimizer:
    opt_name = train_cfg["optimizer"]
    lr = train_cfg["learning_rate"]
    wd = train_cfg.get("weight_decay", 0)

    if opt_name == "sgd":
        momentum = train_cfg.get("momentum", 0.9)
        return torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=momentum,
            weight_decay=wd
        )

    elif opt_name == "adam":
        return torch.optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=wd
        )

    elif opt_name == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=wd
        )

    else:
        raise ValueError(f"Unsupported optimizer: {opt_name}")


def create_scheduler(optimizer: torch.optim.Optimizer, train_cfg: dict):
    sched_cfg = train_cfg.get("scheduler", {})
    sched_type = sched_cfg.get("type", "none")

    if sched_type == "step_lr":
        step_size = sched_cfg.get("step_size", 30)
        gamma = sched_cfg.get("gamma", 0.1)

        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=step_size,
            gamma=gamma
        )

    elif sched_type == "reduce_on_plateau":
        patience = sched_cfg.get("patience", 5)
        factor = sched_cfg.get("factor", 0.5)

        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            patience=patience,
            factor=factor
        )

    elif sched_type == "cosine":
        num_epochs = train_cfg["num_epochs"]

        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=num_epochs
        )

    elif sched_type == "none":
        return None

    else:
        raise ValueError(f"Unsupported scheduler: {sched_type}")


def create_model(model_cfg: dict, num_classes: int) -> nn.Module:
    model_name = model_cfg["name"]
    pretrained = model_cfg.get("pretrained", False)

    if model_name == "resnet18":
        return get_resnet18(
            num_classes=num_classes,
            pretrained=pretrained
        )

    elif model_name == "vit_tiny":
        return get_vit_tiny(
            num_classes=num_classes,
            pretrained=pretrained
        )

    else:
        raise ValueError(f"Unknown model: {model_name}")



def create_base_dataset(data_cfg: dict):
    dataset_name = data_cfg.get("dataset_name", "flowers102")
    root = data_cfg["root"]
    download = data_cfg.get("download", True)

    if dataset_name == "flowers102":
        dataset = OxfordFlowers102(
            root=root,
            transform=None,
            download=download
        )

    elif dataset_name == "dtd":
        dataset = DTD(
            root=root,
            transform=None,
            download=download,
            partition=data_cfg.get("partition", 1)
        )

    else:
        raise ValueError(f"Unknown dataset_name: {dataset_name}")

    return dataset



def create_datasets(data_cfg: dict):
    """
    Создает train_dataset и val_dataset.

    Поддерживаемые датасеты:
    - flowers102
    - dtd

    Поддерживаемые режимы:
    - original
    - mixed_aug
    """

    train_dataset_type = data_cfg.get("train_dataset_type", "original")

    base_dataset = create_base_dataset(data_cfg)

    base_dataset.set_transforms(
        train_transform=train_transform,
        val_transform=val_test_transform,
        test_transform=val_test_transform
    )

    val_dataset = base_dataset.val_dataset

    if train_dataset_type == "original":
        train_dataset = base_dataset.train_dataset

    elif train_dataset_type == "mixed_aug":
        aug_root = data_cfg["aug_root"]
        aug_index_path = data_cfg["aug_index_path"]
        alpha = data_cfg.get("alpha", 0.5)

        train_dataset = MixedAugDataset(
            root=aug_root,
            index_path=aug_index_path,
            transform=train_transform,
            alpha=alpha
        )

        original_train_len = len(base_dataset.train_dataset)
        mixed_aug_len = len(train_dataset)

        if mixed_aug_len != original_train_len:
            raise ValueError(
                "MixedAugDataset length must match original train dataset length "
                "for fair comparison. "
                f"Got mixed_aug_len={mixed_aug_len}, "
                f"original_train_len={original_train_len}."
            )

    else:
        raise ValueError(f"Unknown train_dataset_type: {train_dataset_type}")

    return train_dataset, val_dataset



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config file"
    )

    args = parser.parse_args()

    config = load_config(args.config)

    experiment_name = config["experiment_name"]
    seed = config["seed"]

    # Важно: seed должен быть установлен ДО создания модели, датасетов и DataLoader.
    set_seed(seed)

    requested_device = config.get("device", "cuda")
    if requested_device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    data_cfg = config["data"]
    num_classes = data_cfg["num_classes"]
    batch_size = data_cfg["batch_size"]
    num_workers = data_cfg.get("num_workers", 0)
    pin_memory = True if device.type == "cuda" else False

    model_cfg = config["model"]
    model = create_model(model_cfg, num_classes)

    train_dataset, val_dataset = create_datasets(data_cfg)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    train_cfg = config["training"]

    criterion = nn.CrossEntropyLoss()
    optimizer = create_optimizer(model, train_cfg)
    scheduler = create_scheduler(optimizer, train_cfg)

    log_cfg = config["logging"]
    comet_cfg = log_cfg.get("comet", {})

    comet_logger = CometLogger(
        project_name=comet_cfg.get("project_name"),
        experiment_name=experiment_name,
        api_key=comet_cfg.get("api_key"),
        workspace=comet_cfg.get("workspace"),
        disabled=comet_cfg.get("disabled", False)
    )

    scheduler_cfg = train_cfg.get("scheduler", {"type": "none"})

    train_dataset_size = len(train_dataset)
    val_dataset_size = len(val_dataset)
    train_batches_per_epoch = len(train_loader)
    val_batches_per_epoch = len(val_loader)
    total_optimizer_steps = train_batches_per_epoch * train_cfg["num_epochs"]

    print("=" * 80)
    print(f"Experiment: {experiment_name}")
    print(f"Device: {device}")
    print(f"Train dataset type: {data_cfg.get('train_dataset_type', 'original')}")
    print(f"Train dataset size: {train_dataset_size}")
    print(f"Val dataset size: {val_dataset_size}")
    print(f"Train batches per epoch: {train_batches_per_epoch}")
    print(f"Total optimizer steps: {total_optimizer_steps}")
    print("=" * 80)

    comet_logger.log_parameters({
        "dataset_name": data_cfg.get("dataset_name", "flowers102"),
        "partition": data_cfg.get("partition", None),

        "model": model_cfg["name"],
        "pretrained": model_cfg.get("pretrained", False),
        "num_classes": num_classes,

        "train_dataset_type": data_cfg.get("train_dataset_type", "original"),
        "num_aug_per_image": data_cfg.get("num_aug_per_image", 0),
        "alpha": data_cfg.get("alpha", None),
        "aug_root": data_cfg.get("aug_root", None),
        "aug_index_path": data_cfg.get("aug_index_path", None),

        "train_dataset_size": train_dataset_size,
        "val_dataset_size": val_dataset_size,
        "train_batches_per_epoch": train_batches_per_epoch,
        "val_batches_per_epoch": val_batches_per_epoch,
        "total_optimizer_steps": total_optimizer_steps,

        "batch_size": batch_size,
        "optimizer": train_cfg["optimizer"],
        "learning_rate": train_cfg["learning_rate"],
        "momentum": train_cfg.get("momentum", None),
        "weight_decay": train_cfg.get("weight_decay", 0),

        "scheduler": scheduler_cfg.get("type", "none"),
        "scheduler_params": str(scheduler_cfg),

        "seed": seed,
        "experiment_name": experiment_name,
    })


    if data_cfg.get("train_dataset_type", "original") == "mixed_aug":
        comet_logger.log_parameters({
            "aug_root": data_cfg.get("aug_root"),
            "aug_index_path": data_cfg.get("aug_index_path"),
            "alpha": data_cfg.get("alpha", 0.5),
            "include_original_in_aug_pool": data_cfg.get(
                "include_original_in_aug_pool",
                False
            )
        })

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        experiment_name=experiment_name,
        num_epochs=train_cfg["num_epochs"],
        log_dir=log_cfg["log_dir"],
        comet_logger=comet_logger,
        seed=seed,
        log_batch_loss=log_cfg.get("log_batch_loss", False),
        config=config
    )

    # Сохраняем копию конфига в папку эксперимента.
    shutil.copy(
        args.config,
        os.path.join(
            trainer.local_logger.experiment_dir,
            f"{experiment_name}_config.yaml"
        )
    )

    trainer.fit()


if __name__ == "__main__":
    main()
