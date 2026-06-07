# local_datasets/mini_imagenet.py

from typing import Callable, Optional

from datasets import load_dataset


class _HFDatasetWrapper:
    def __init__(
        self,
        hf_dataset,
        transform: Optional[Callable] = None,
        class_names: Optional[list[str]] = None,
    ):
        self.hf_dataset = hf_dataset
        self._transform = transform
        self.class_names = class_names

        self._targets = [
            int(self.hf_dataset[i]["label"])
            for i in range(len(self.hf_dataset))
        ]

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        item = self.hf_dataset[idx]

        image = item["image"]
        label = int(item["label"])

        if image.mode != "RGB":
            image = image.convert("RGB")

        if self._transform is not None:
            image = self._transform(image)

        return image, label

    @property
    def targets(self):
        return self._targets

    def set_transform(self, transform: Optional[Callable]):
        self._transform = transform


class MiniImageNet:
    def __init__(
        self,
        root: str = "./data",
        transform: Optional[Callable] = None,
        download: bool = True,
        **kwargs,
    ):
        ds = load_dataset("timm/mini-imagenet")

        self.root = root
        self.download = download
        self.num_classes = 100

        # HuggingFace ClassLabel usually stores class names here.
        label_feature = ds["train"].features["label"]

        if hasattr(label_feature, "names") and label_feature.names is not None:
            self.class_names = [str(x).replace("_", " ") for x in label_feature.names]
        else:
            self.class_names = [f"class {i}" for i in range(self.num_classes)]

        self.train_dataset = _HFDatasetWrapper(
            ds["train"],
            transform=transform,
            class_names=self.class_names,
        )

        self.val_dataset = _HFDatasetWrapper(
            ds["validation"],
            transform=transform,
            class_names=self.class_names,
        )

        self.test_dataset = _HFDatasetWrapper(
            ds["test"],
            transform=transform,
            class_names=self.class_names,
        )

    def set_transforms(
        self,
        train_transform: Optional[Callable] = None,
        val_transform: Optional[Callable] = None,
        test_transform: Optional[Callable] = None,
    ):
        if train_transform is not None:
            self.train_dataset.set_transform(train_transform)

        if val_transform is not None:
            self.val_dataset.set_transform(val_transform)

        if test_transform is not None:
            self.test_dataset.set_transform(test_transform)
