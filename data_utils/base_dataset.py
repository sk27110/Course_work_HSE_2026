# data_utils/base_dataset.py

"""
Base dataset loading utilities.

This module is intentionally responsible only for loading dataset wrappers and
configuring split transforms. It does not know anything about few-shot sampling,
offline augmentation, or classifier training experiments.
"""

from typing import Dict, Optional

from local_datasets.dtd import DTD
from local_datasets.mini_imagenet import MiniImageNet
from local_datasets.oxford_flower102 import OxfordFlowers102


SUPPORTED_DATASETS = {
    "flowers102": OxfordFlowers102,
    "oxfordflowers102": OxfordFlowers102,
    "oxford_flowers102": OxfordFlowers102,
    "miniimagenet": MiniImageNet,
    "mini_imagenet": MiniImageNet,
    "mini-imagenet": MiniImageNet,
    "dtd": DTD,
    "describable_textures": DTD,
}


def create_base_dataset(data_cfg: Dict):
    """
    Load a dataset wrapper from config without applying project-specific logic.

    Returned object is expected to expose train_dataset, val_dataset and
    optionally test_dataset. Transforms are left unset by default so callers can
    decide whether they need raw PIL images or tensors.
    """

    dataset_name = data_cfg.get("dataset_name", "flowers102").lower()
    root = data_cfg.get("root", "./data")
    download = data_cfg.get("download", True)

    if dataset_name not in SUPPORTED_DATASETS:
        supported = ", ".join(sorted(SUPPORTED_DATASETS.keys()))
        raise ValueError(
            f"Unknown dataset_name: {dataset_name}. Supported datasets: {supported}"
        )

    dataset_cls = SUPPORTED_DATASETS[dataset_name]

    kwargs = {
        "root": root,
        "transform": None,
        "download": download,
    }

    if dataset_name in {"dtd", "describable_textures"}:
        kwargs["partition"] = int(data_cfg.get("partition", 1))

    return dataset_cls(**kwargs)


def set_split_transforms(
    base_dataset,
    train_transform=None,
    val_transform=None,
    test_transform=None,
):
    """
    Apply transforms to an already loaded base dataset wrapper.
    """

    if not hasattr(base_dataset, "set_transforms"):
        raise ValueError("Base dataset wrapper must implement set_transforms(...).")

    base_dataset.set_transforms(
        train_transform=train_transform,
        val_transform=val_transform,
        test_transform=test_transform,
    )


def get_split(base_dataset, split: str):
    """
    Return a named split from a base dataset wrapper.
    """

    split = split.lower()
    attr_name = f"{split}_dataset"

    if not hasattr(base_dataset, attr_name):
        raise ValueError(f"Dataset does not expose split '{split}'.")

    return getattr(base_dataset, attr_name)


def infer_num_classes(base_dataset) -> Optional[int]:
    """
    Return num_classes if the dataset wrapper exposes it.
    """

    if hasattr(base_dataset, "num_classes"):
        return int(base_dataset.num_classes)

    if hasattr(base_dataset, "class_names") and base_dataset.class_names is not None:
        return len(base_dataset.class_names)

    return None
