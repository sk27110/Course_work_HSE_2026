# data_utils/dataset_factory.py

from typing import Dict, Tuple

from torch.utils.data import Dataset

from local_datasets.oxford_flower102 import OxfordFlowers102
from local_datasets.mini_imagenet import MiniImageNet

from data_utils.few_shot import build_few_shot_subset
from data_utils.mixed_aug_dataset import MixedAugDataset

from utils.transforms import train_transform, val_test_transform


def create_base_dataset(data_cfg: Dict):
    dataset_name = data_cfg.get("dataset_name", "flowers102").lower()
    root = data_cfg.get("root", "./data")
    download = data_cfg.get("download", True)

    if dataset_name in ["flowers102", "oxfordflowers102", "oxford_flowers102"]:
        return OxfordFlowers102(
            root=root,
            transform=None,
            download=download,
        )

    if dataset_name in ["miniimagenet", "mini_imagenet", "mini-imagenet"]:
        return MiniImageNet(
            root=root,
            transform=None,
            download=download,
        )

    raise ValueError(f"Unknown dataset_name: {dataset_name}")


def apply_few_shot_if_needed(
    full_train_dataset: Dataset,
    data_cfg: Dict,
    seed: int,
) -> Dataset:
    few_shot_cfg = data_cfg.get("few_shot", {})
    enabled = bool(few_shot_cfg.get("enabled", False))

    if not enabled:
        return full_train_dataset

    k = int(few_shot_cfg["k"])
    few_shot_seed = int(few_shot_cfg.get("seed", seed))

    indices_path = few_shot_cfg.get("indices_path", None)
    save_indices_path = few_shot_cfg.get("save_indices_path", None)

    few_shot_dataset = build_few_shot_subset(
        dataset=full_train_dataset,
        k=k,
        seed=few_shot_seed,
        indices_path=indices_path,
        save_indices_path=save_indices_path,
    )

    return few_shot_dataset


def create_datasets(data_cfg: Dict, seed: int) -> Tuple[Dataset, Dataset]:
    """
    Main dataset creation logic.

    Important behavior:
    - validation is always full val split;
    - few-shot is applied only to train split;
    - mixed_aug length must match effective train length;
    - for mixed_aug, augmentations must be generated from the same effective train dataset.
    """

    train_dataset_type = data_cfg.get("train_dataset_type", "original").lower()

    base_dataset = create_base_dataset(data_cfg)

    full_train_dataset = base_dataset.train_dataset
    val_dataset = base_dataset.val_dataset

    effective_train_dataset = apply_few_shot_if_needed(
        full_train_dataset=full_train_dataset,
        data_cfg=data_cfg,
        seed=seed,
    )

    expected_train_len = len(effective_train_dataset)

    # Set transforms after few-shot wrapping.
    # If effective_train_dataset is SubsetDataset, it points to full_train_dataset,
    # so assigning transform to base_dataset.train_dataset is correct.
    base_dataset.set_transforms(
        train_transform=train_transform,
        val_transform=val_test_transform,
        test_transform=val_test_transform,
    )

    if train_dataset_type == "original":
        train_dataset = effective_train_dataset

    elif train_dataset_type == "mixed_aug":
        train_dataset = MixedAugDataset(
            root=data_cfg["aug_root"],
            index_path=data_cfg["aug_index_path"],
            transform=train_transform,
            alpha=float(data_cfg.get("alpha", 0.5)),
        )

        if len(train_dataset) != expected_train_len:
            raise ValueError(
                "MixedAugDataset length must match effective train dataset length. "
                "For few-shot experiments, generate augmentations from the same few-shot subset. "
                f"Got mixed_aug_len={len(train_dataset)}, "
                f"expected_train_len={expected_train_len}."
            )

    else:
        raise ValueError(f"Unknown train_dataset_type: {train_dataset_type}")

    return train_dataset, val_dataset


def create_raw_train_dataset_for_generation(data_cfg: Dict, seed: int) -> Dataset:
    """
    Used by offline augmentation generation.

    Returns train dataset without torchvision tensor transforms.
    If few-shot is enabled, returns exactly the same few-shot subset logic.
    """

    base_dataset = create_base_dataset(data_cfg)
    full_train_dataset = base_dataset.train_dataset

    effective_train_dataset = apply_few_shot_if_needed(
        full_train_dataset=full_train_dataset,
        data_cfg=data_cfg,
        seed=seed,
    )

    return effective_train_dataset
