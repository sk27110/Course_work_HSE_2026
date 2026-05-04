import argparse
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
from datetime import datetime

from training.models import get_resnet18, get_vit_tiny
from local_datasets.oxford_flower102 import OxfordFlowers102
from utils.transforms import val_test_transform   # используем как для валидации
from training.trainer import set_seed


def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def build_model_from_config(config: dict, num_classes: int, device: torch.device) -> nn.Module:
    model_cfg = config['model']
    if model_cfg['name'] == 'resnet18':
        model = get_resnet18(num_classes=num_classes, pretrained=False)  # веса загрузим из чекпоинта
    elif model_cfg['name'] == 'vit_tiny':
        model = get_vit_tiny(num_classes=num_classes, pretrained=False)
    else:
        raise ValueError(f"Unknown model: {model_cfg['name']}")
    return model.to(device)


def find_best_checkpoint(experiment_dir: str, experiment_name: str) -> str:
    """Ищет файл чекпоинта с именем <experiment_name>_best.pth."""
    checkpoint_path = os.path.join(experiment_dir, "checkpoints", f"{experiment_name}_best.pth")
    if not os.path.exists(checkpoint_path):
        # Попробуем старое имя 'best_model.pth' на случай, если не переименовали
        fallback = os.path.join(experiment_dir, "checkpoints", "best_model.pth")
        if os.path.exists(fallback):
            return fallback
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path} or {fallback}")
    return checkpoint_path


def evaluate(model: nn.Module, test_loader: DataLoader, criterion: nn.Module, device: torch.device):
    model.eval()
    running_loss = 0.0
    correct1 = 0
    correct5 = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.topk(5, dim=1)
            total += targets.size(0)
            correct1 += (predicted[:, 0] == targets).sum().item()
            correct5 += (predicted == targets.view(-1, 1)).sum().item()

    test_loss = running_loss / total
    test_acc = correct1 / total
    test_top5 = correct5 / total
    return test_loss, test_acc, test_top5


def main():
    parser = argparse.ArgumentParser(description="Test a trained model on the test set.")
    parser.add_argument('--config', type=str, required=True, help='Path to the experiment YAML config')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to checkpoint (.pth). If not provided, auto-detect from config/logs.')
    args = parser.parse_args()

    config = load_config(args.config)

    experiment_name = config['experiment_name']
    seed = config.get('seed', 42)
    set_seed(seed)

    device = torch.device(config['device'] if torch.cuda.is_available() else "cpu")
    data_cfg = config['data']
    root = data_cfg['root']
    num_classes = data_cfg['num_classes']
    batch_size = data_cfg['batch_size']
    num_workers = data_cfg.get('num_workers', 4)

    # Загружаем тестовый датасет
    flowers = OxfordFlowers102(root=root, transform=None, download=True)
    flowers.set_transforms(test_transform=val_test_transform)  # только тестовая трансформация
    test_loader = DataLoader(flowers.test_dataset, batch_size=batch_size,
                             shuffle=False, num_workers=num_workers, pin_memory=True)

    # Модель
    model = build_model_from_config(config, num_classes, device)

    # Определяем чекпоинт
    if args.checkpoint:
        checkpoint_path = args.checkpoint
    else:
        log_dir = config['logging']['log_dir']
        experiment_dir = os.path.join(log_dir, experiment_name)
        checkpoint_path = find_best_checkpoint(experiment_dir, experiment_name)
    print(f"Loading checkpoint from: {checkpoint_path}")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    criterion = nn.CrossEntropyLoss()

    test_loss, test_acc, test_top5 = evaluate(model, test_loader, criterion, device)

    # Вывод результатов
    print("\n" + "="*50)
    print(f"Test Results for experiment: {experiment_name}")
    print(f"Test Loss      : {test_loss:.4f}")
    print(f"Test Accuracy  : {test_acc:.4f}")
    print(f"Test Top-5 Acc : {test_top5:.4f}")
    print("="*50 + "\n")

    # Сохраняем результаты в файл в папке эксперимента
    log_dir = config['logging']['log_dir']
    experiment_dir = os.path.join(log_dir, experiment_name)
    os.makedirs(experiment_dir, exist_ok=True)
    results_file = os.path.join(experiment_dir, "test_results.txt")
    with open(results_file, 'w') as f:
        f.write(f"Experiment: {experiment_name}\n")
        f.write(f"Checkpoint: {checkpoint_path}\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Test Loss: {test_loss:.4f}\n")
        f.write(f"Test Accuracy: {test_acc:.4f}\n")
        f.write(f"Test Top-5 Accuracy: {test_top5:.4f}\n")
    print(f"Results saved to {results_file}")

if __name__ == "__main__":
    main()