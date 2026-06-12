# data_utils/few_shot_dataset.py

"""
Few-shot dataset construction utilities.

This module owns the procedure of creating/loading few-shot subsets. It does not
perform augmentation and does not construct dataloaders, models, optimizers or
training experiments.
"""

from typing import Dict

from torch.utils.data import Dataset

from data_utils.base_dataset import create_base_dataset, get_split, set_split_transforms
from data_utils.few_shot import build_few_shot_subset


def apply_few_shot_if_needed(
    full_train_dataset: Dataset,
    data_cfg: Dict,
    seed: int,
) -> Dataset:
    """
    Apply the few-shot config to a train split.

    If few_shot.enabled is false, the original train split is returned unchanged.
    If indices_path is provided, fixed indices are loaded. Otherwise, a new
    reproducible k-shot subset is sampled and can optionally be saved.
    """

    few_shot_cfg = data_cfg.get("few_shot", {})
    enabled = bool(few_shot_cfg.get("enabled", False))

    if not enabled:
        return full_train_dataset

    k = int(few_shot_cfg["k"])
    few_shot_seed = int(few_shot_cfg.get("seed", seed))

    indices_path = few_shot_cfg.get("indices_path", None)
    save_indices_path = few_shot_cfg.get("save_indices_path", None)

    return build_few_shot_subset(
        dataset=full_train_dataset,
        k=k,
        seed=few_shot_seed,
        indices_path=indices_path,
        save_indices_path=save_indices_path,
    )


def create_source_train_dataset(
    data_cfg: Dict,
    seed: int,
    train_transform=None,
) -> Dataset:
    """
    Create the source train dataset used by later pipeline stages.

    This is the only public constructor needed by augmentation generation:
    it returns either a full train split or a few-shot train subset, still without
    performing any augmentation.
    """

    base_dataset = create_base_dataset(data_cfg)

    set_split_transforms(
        base_dataset=base_dataset,
        train_transform=train_transform,
        val_transform=None,
        test_transform=None,
    )

    full_train_dataset = get_split(base_dataset, "train")

    return apply_few_shot_if_needed(
        full_train_dataset=full_train_dataset,
        data_cfg=data_cfg,
        seed=seed,
    )


def create_raw_source_train_dataset(data_cfg: Dict, seed: int) -> Dataset:
    """
    Create source train data as raw PIL images.

    This function is the preferred input for offline augmentation generators.
    """

    return create_source_train_dataset(
        data_cfg=data_cfg,
        seed=seed,
        train_transform=None,
    )
