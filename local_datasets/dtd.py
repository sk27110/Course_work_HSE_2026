# local_datasets/dtd.py

from torchvision import datasets


class DTD:
    """
    Describable Textures Dataset.

    В torchvision уже есть готовые split'ы:
    - train
    - val
    - test

    По умолчанию используется partition=1.
    """

    def __init__(
        self,
        root: str = "./data",
        transform=None,
        download: bool = True,
        partition: int = 1
    ):
        self.root = root
        self.download = download
        self.num_classes = 47
        self.partition = partition

        self.train_dataset = datasets.DTD(
            root=root,
            split="train",
            partition=partition,
            transform=transform,
            download=download
        )

        self.val_dataset = datasets.DTD(
            root=root,
            split="val",
            partition=partition,
            transform=transform,
            download=download
        )

        self.test_dataset = datasets.DTD(
            root=root,
            split="test",
            partition=partition,
            transform=transform,
            download=download
        )

        self.class_names = list(getattr(self.train_dataset, "classes", []))
        self.train_dataset.class_names = self.class_names
        self.val_dataset.class_names = self.class_names
        self.test_dataset.class_names = self.class_names

    def set_transforms(
        self,
        train_transform=None,
        val_transform=None,
        test_transform=None
    ):
        """
        Позволяет задать разные трансформации для train/val/test после инициализации.
        """

        if train_transform is not None:
            self.train_dataset.transform = train_transform

        if val_transform is not None:
            self.val_dataset.transform = val_transform

        if test_transform is not None:
            self.test_dataset.transform = test_transform
