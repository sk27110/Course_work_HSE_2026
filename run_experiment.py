# run_experiment.py
import os
import yaml
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from training.models import get_resnet18, get_vit_tiny
from training.trainer import Trainer
from training.logger import CometLogger
from local_datasets.oxford_flower102 import OxfordFlowers102
from utils.transforms import train_transform, val_test_transform


def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def create_optimizer(model: nn.Module, train_cfg: dict) -> torch.optim.Optimizer:
    opt_name = train_cfg['optimizer']
    lr = train_cfg['learning_rate']
    wd = train_cfg.get('weight_decay', 0)

    if opt_name == 'sgd':
        momentum = train_cfg.get('momentum', 0.9)
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=wd)
    elif opt_name == 'adam':
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    else:
        raise ValueError(f"Unsupported optimizer: {opt_name}")


def create_scheduler(optimizer: torch.optim.Optimizer, train_cfg: dict):
    sched_cfg = train_cfg.get('scheduler', {})
    sched_type = sched_cfg.get('type', 'none')

    if sched_type == 'step_lr':
        step_size = sched_cfg.get('step_size', 30)
        gamma = sched_cfg.get('gamma', 0.1)
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    elif sched_type == 'reduce_on_plateau':
        patience = sched_cfg.get('patience', 5)
        factor = sched_cfg.get('factor', 0.5)
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', patience=patience, factor=factor
        )
    elif sched_type == 'none':
        return None
    else:
        raise ValueError(f"Unsupported scheduler: {sched_type}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to YAML config file')
    args = parser.parse_args()

    config = load_config(args.config)

    experiment_name = config['experiment_name']
    seed = config['seed']
    device = torch.device(config['device'] if torch.cuda.is_available() else "cpu")

    data_cfg = config['data']
    root = data_cfg['root']
    num_classes = data_cfg['num_classes']
    batch_size = data_cfg['batch_size']
    num_workers = data_cfg['num_workers']

    model_cfg = config['model']
    if model_cfg['name'] == 'resnet18':
        model = get_resnet18(num_classes=num_classes, pretrained=model_cfg['pretrained'])
    elif model_cfg['name'] == 'vit_tiny':
        model = get_vit_tiny(num_classes=num_classes, pretrained=model_cfg['pretrained'])
    else:
        raise ValueError(f"Unknown model: {model_cfg['name']}")

    # Датасет
    flowers = OxfordFlowers102(root=root, transform=None, download=True)
    flowers.set_transforms(
        train_transform=train_transform,
        val_transform=val_test_transform,
        test_transform=val_test_transform
    )
    train_loader = DataLoader(flowers.train_dataset, batch_size=batch_size,
                              shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(flowers.val_dataset, batch_size=batch_size,
                            shuffle=False, num_workers=num_workers, pin_memory=True)

    train_cfg = config['training']
    criterion = nn.CrossEntropyLoss()
    optimizer = create_optimizer(model, train_cfg)
    scheduler = create_scheduler(optimizer, train_cfg)

    log_cfg = config['logging']
    comet_logger = CometLogger(
        project_name=log_cfg['comet']['project_name'],
        experiment_name=experiment_name,
        api_key=log_cfg['comet'].get('api_key'),
        workspace=log_cfg['comet'].get('workspace'),
        disabled=log_cfg['comet'].get('disabled', False)
    )
    comet_logger.log_parameters({
        "model": model_cfg['name'],
        "pretrained": model_cfg['pretrained'],
        "num_classes": num_classes,
        "batch_size": batch_size,
        "optimizer": train_cfg['optimizer'],
        "learning_rate": train_cfg['learning_rate'],
        "momentum": train_cfg.get('momentum', None),
        "weight_decay": train_cfg.get('weight_decay', 0),
        "scheduler": train_cfg['scheduler']['type'],
        "scheduler_params": str(train_cfg['scheduler']),
        "seed": seed,
        "experiment_name": experiment_name
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
        num_epochs=train_cfg['num_epochs'],
        log_dir=log_cfg['log_dir'],
        comet_logger=comet_logger,
        seed=seed,
        log_batch_loss=log_cfg.get('log_batch_loss', False),
        config=config
    )

    # Сохраняем копию конфига в папку эксперимента
    import shutil
    shutil.copy(args.config, os.path.join(trainer.local_logger.experiment_dir, f"{experiment_name}_config.yaml"))

    trainer.fit()


if __name__ == "__main__":
    main()