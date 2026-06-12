# data_utils/experiment_datasets.py

"""
Dataset construction for classifier experiments.

This module turns an already defined source dataset or an already saved offline
augmented dataset into train/validation datasets for classifier training. It does
not create few-shot indices itself beyond delegating to few_shot_dataset, and it
does not run augmentation methods.
"""

from typing import Dict, Tuple

from torch.utils.data import Dataset

from data_utils.base_dataset import create_base_dataset, get_split, set_split_transforms
from data_utils.few_shot_dataset import create_source_train_dataset
from data_utils.mixed_aug_dataset import MixedAugDataset
from utils.transforms import train_transform, val_test_transform


def create_validation_dataset(data_cfg: Dict) -> Dataset:
    """
    Create the validation split used by training/evaluation.
    """

    base_dataset = create_base_dataset(data_cfg)
    set_split_transforms(
        base_dataset=base_dataset,
        train_transform=None,
        val_transform=val_test_transform,
        test_transform=val_test_transform,
    )
    return get_split(base_dataset, "val")


def create_original_train_dataset(data_cfg: Dict, seed: int) -> Dataset:
    """
    Create train data directly from the source full/few-shot dataset.
    """

    return create_source_train_dataset(
        data_cfg=data_cfg,
        seed=seed,
        train_transform=train_transform,
    )


def create_augmented_train_dataset(data_cfg: Dict, seed: int) -> Dataset:
    """
    Load a previously generated offline augmented dataset for classifier training.

    The generated index length is validated against the corresponding source
    train dataset length. This catches accidental training on augmentations that
    were generated from another few-shot seed, another k, or a full dataset.
    """

    if "aug_root" not in data_cfg:
        raise ValueError("data.aug_root is required for augmented train datasets.")

    if "aug_index_path" not in data_cfg:
        raise ValueError("data.aug_index_path is required for augmented train datasets.")

    reference_source_dataset = create_source_train_dataset(
        data_cfg=data_cfg,
        seed=seed,
        train_transform=None,
    )

    train_dataset = MixedAugDataset(
        root=data_cfg["aug_root"],
        index_path=data_cfg["aug_index_path"],
        transform=train_transform,
        alpha=float(data_cfg.get("alpha", 0.5)),
    )

    expected_train_len = len(reference_source_dataset)
    if len(train_dataset) != expected_train_len:
        raise ValueError(
            "Augmented dataset length must match the source train dataset length. "
            "Generate augmentations from the same source/few-shot dataset that this "
            "experiment config references. "
            f"Got augmented_len={len(train_dataset)}, expected_train_len={expected_train_len}."
        )

    return train_dataset


def create_experiment_datasets(data_cfg: Dict, seed: int) -> Tuple[Dataset, Dataset]:
    """
    Create train and validation datasets for classifier experiments.

    Supported train_dataset_type values:
    - original: train on the original full/few-shot source dataset;
    - mixed_aug / augmented / offline_aug: train on a saved offline augmented dataset.
    """

    train_dataset_type = data_cfg.get("train_dataset_type", "original").lower()

    if train_dataset_type == "original":
        train_dataset = create_original_train_dataset(
            data_cfg=data_cfg,
            seed=seed,
        )
    elif train_dataset_type in {"mixed_aug", "augmented", "offline_aug"}:
        train_dataset = create_augmented_train_dataset(
            data_cfg=data_cfg,
            seed=seed,
        )
    else:
        raise ValueError(f"Unknown train_dataset_type: {train_dataset_type}")

    val_dataset = create_validation_dataset(data_cfg)
    return train_dataset, val_dataset
