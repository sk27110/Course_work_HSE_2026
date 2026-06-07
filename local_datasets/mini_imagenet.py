# local_datasets/mini_imagenet.py

import os
import zipfile
import tarfile
from typing import Optional

from torchvision import datasets
from torchvision.datasets.utils import download_url


class MiniImageNet:
    """
    Датасет miniImageNet в формате ImageFolder.

    Ожидаемая структура после скачивания/распаковки:

    root/
        train/
            class_1/
                img_001.jpg
                ...
            class_2/
                ...
        val/
            class_1/
                ...
        test/
            class_1/
                ...

    Пример:
        data/miniimagenet/
            train/
            val/
            test/
    """

    def __init__(
        self,
        root: str = "./data/miniimagenet",
        transform=None,
        download: bool = False,
        url: Optional[str] = None,
        archive_name: Optional[str] = None,
        remove_archive: bool = False
    ):
        self.root = root
        self.download = download
        self.url = url
        self.archive_name = archive_name
        self.remove_archive = remove_archive

        self.train_dir = os.path.join(root, "train")
        self.val_dir = os.path.join(root, "val")
        self.test_dir = os.path.join(root, "test")

        if download:
            self.download_and_extract()

        self._check_dirs()

        self.train_dataset = datasets.ImageFolder(
            root=self.train_dir,
            transform=transform
        )

        self.val_dataset = datasets.ImageFolder(
            root=self.val_dir,
            transform=transform
        )

        self.test_dataset = datasets.ImageFolder(
            root=self.test_dir,
            transform=transform
        )

        self.num_classes = len(self.train_dataset.classes)

        self._check_class_consistency()

    def _is_prepared(self) -> bool:
        """
        Проверяет, существует ли уже готовая структура train/val/test.
        """
        return (
            os.path.isdir(self.train_dir)
            and os.path.isdir(self.val_dir)
            and os.path.isdir(self.test_dir)
        )

    def download_and_extract(self):
        """
        Скачивает и распаковывает архив miniImageNet.

        Важно:
        url должен вести на прямую ссылку на архив:
        .zip, .tar, .tar.gz или .tgz
        """

        if self._is_prepared():
            print(f"miniImageNet already exists at: {self.root}")
            return

        if self.url is None:
            raise ValueError(
                "download=True, но url не указан.\n"
                "Добавь в конфиг:\n"
                "data:\n"
                "  download: true\n"
                "  url: DIRECT_URL_TO_MINIIMAGENET_ARCHIVE\n"
            )

        os.makedirs(self.root, exist_ok=True)

        if self.archive_name is None:
            self.archive_name = os.path.basename(self.url.split("?")[0])

            if self.archive_name == "":
                self.archive_name = "miniimagenet_archive"

        archive_path = os.path.join(self.root, self.archive_name)

        if not os.path.isfile(archive_path):
            print(f"Downloading miniImageNet from:\n{self.url}")
            download_url(
                url=self.url,
                root=self.root,
                filename=self.archive_name
            )
        else:
            print(f"Archive already exists: {archive_path}")

        print(f"Extracting archive: {archive_path}")
        self._extract_archive(archive_path, self.root)

        if self.remove_archive:
            print(f"Removing archive: {archive_path}")
            os.remove(archive_path)

        if not self._is_prepared():
            raise RuntimeError(
                "Архив был скачан и распакован, но структура train/val/test "
                "не найдена.\n\n"
                f"Ожидалось:\n"
                f"{self.root}/train/class_name/*.jpg\n"
                f"{self.root}/val/class_name/*.jpg\n"
                f"{self.root}/test/class_name/*.jpg\n\n"
                "Возможно, внутри архива есть дополнительная вложенная папка "
                "или структура отличается от ImageFolder."
            )

    def _extract_archive(self, archive_path: str, extract_to: str):
        """
        Распаковка .zip, .tar, .tar.gz, .tgz.
        """

        lower_path = archive_path.lower()

        if lower_path.endswith(".zip"):
            with zipfile.ZipFile(archive_path, "r") as zip_ref:
                zip_ref.extractall(extract_to)

        elif (
            lower_path.endswith(".tar")
            or lower_path.endswith(".tar.gz")
            or lower_path.endswith(".tgz")
        ):
            with tarfile.open(archive_path, "r:*") as tar_ref:
                tar_ref.extractall(extract_to)

        else:
            raise ValueError(
                f"Unsupported archive format: {archive_path}\n"
                "Поддерживаются только .zip, .tar, .tar.gz, .tgz"
            )

    def _check_dirs(self):
        required_dirs = [
            self.train_dir,
            self.val_dir,
            self.test_dir
        ]

        for directory in required_dirs:
            if not os.path.isdir(directory):
                raise FileNotFoundError(
                    f"Directory not found: {directory}\n\n"
                    f"Expected miniImageNet structure:\n"
                    f"{self.root}/train/class_name/*.jpg\n"
                    f"{self.root}/val/class_name/*.jpg\n"
                    f"{self.root}/test/class_name/*.jpg\n\n"
                    f"If you want automatic download, set in config:\n"
                    f"download: true\n"
                    f"url: YOUR_DIRECT_ARCHIVE_URL"
                )

    def _check_class_consistency(self):
        train_classes = self.train_dataset.classes
        val_classes = self.val_dataset.classes
        test_classes = self.test_dataset.classes

        if train_classes != val_classes:
            raise ValueError(
                "Class folders in train and val are different.\n"
                f"Train classes count: {len(train_classes)}\n"
                f"Val classes count: {len(val_classes)}\n"
                f"Train classes: {train_classes}\n"
                f"Val classes: {val_classes}"
            )

        if train_classes != test_classes:
            raise ValueError(
                "Class folders in train and test are different.\n"
                f"Train classes count: {len(train_classes)}\n"
                f"Test classes count: {len(test_classes)}\n"
                f"Train classes: {train_classes}\n"
                f"Test classes: {test_classes}"
            )

    def set_transforms(
        self,
        train_transform=None,
        val_transform=None,
        test_transform=None
    ):
        """
        Позволяет задать разные трансформации для train/val/test
        после инициализации.
        """

        if train_transform is not None:
            self.train_dataset.transform = train_transform

        if val_transform is not None:
            self.val_dataset.transform = val_transform

        if test_transform is not None:
            self.test_dataset.transform = test_transform
