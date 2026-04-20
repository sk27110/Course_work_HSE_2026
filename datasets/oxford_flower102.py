from torchvision import datasets
from typing import Optional, List, Tuple

class OxfordFlowers102:
    """
    Удобная обёртка над torchvision Flowers102 с train / val / test.
    """
    def __init__(self, root: str = './data', transform=None, download=True):
        self.root = root
        self.transform = transform
        self.download = download

        self.train_dataset = datasets.Flowers102(
            root=root, split='train', transform=transform, download=download
        )
        self.val_dataset = datasets.Flowers102(
            root=root, split='val', transform=transform, download=download
        )
        self.test_dataset = datasets.Flowers102(
            root=root, split='test', transform=transform, download=download
        )

        self.num_classes = 102
