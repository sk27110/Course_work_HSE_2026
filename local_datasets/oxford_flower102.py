# local_datasets/oxford_flower102.py
from torchvision import datasets
from typing import Optional

class OxfordFlowers102:
    def __init__(self, root: str = './data', transform=None, download=True):
        self.root = root
        self.download = download
        self.num_classes = 102

        self.train_dataset = datasets.Flowers102(
            root=root, split='train', transform=transform, download=download
        )
        self.val_dataset = datasets.Flowers102(
            root=root, split='val', transform=transform, download=download
        )
        self.test_dataset = datasets.Flowers102(
            root=root, split='test', transform=transform, download=download
        )

    def set_transforms(self, train_transform=None, val_transform=None, test_transform=None):
        """Позволяет задать разные трансформации для train/val/test после инициализации."""
        if train_transform is not None:
            self.train_dataset.transform = train_transform
        if val_transform is not None:
            self.val_dataset.transform = val_transform
        if test_transform is not None:
            self.test_dataset.transform = test_transform