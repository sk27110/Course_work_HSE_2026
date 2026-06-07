# local_datasets/cifar100.py

from torchvision import datasets


class CIFAR100Dataset:
    def __init__(
        self,
        root: str = "./data/cifar100",
        transform=None,
        download: bool = True
    ):
        self.root = root
        self.download = download
        self.num_classes = 100

        self.train_dataset = datasets.CIFAR100(
            root=root,
            train=True,
            transform=transform,
            download=download
        )

        # У CIFAR100 нет отдельного val split.
        # Поэтому test split используем как val_dataset.
        self.val_dataset = datasets.CIFAR100(
            root=root,
            train=False,
            transform=transform,
            download=download
        )

        self.test_dataset = self.val_dataset

    def set_transforms(
        self,
        train_transform=None,
        val_transform=None,
        test_transform=None
    ):
        if train_transform is not None:
            self.train_dataset.transform = train_transform

        if val_transform is not None:
            self.val_dataset.transform = val_transform

        if test_transform is not None:
            self.test_dataset.transform = test_transform
