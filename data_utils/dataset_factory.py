# data_utils/dataset_factory.py

"""
Backward-compatible dataset factory facade.

The project now separates responsibilities into:
- data_utils.base_dataset: loading base dataset wrappers;
- data_utils.few_shot_dataset: creating/loading full or few-shot source datasets;
- data_utils.experiment_datasets: creating train/val datasets for classifier runs.

Existing notebooks/configs can still import the historical function names from
this module, but new code should use the narrower modules directly.
"""

from typing import Dict, Tuple

from torch.utils.data import Dataset

from data_utils.base_dataset import create_base_dataset
from data_utils.experiment_datasets import create_experiment_datasets
from data_utils.few_shot_dataset import (
    apply_few_shot_if_needed,
    create_raw_source_train_dataset,
)


def create_datasets(data_cfg: Dict, seed: int) -> Tuple[Dataset, Dataset]:
    """
    Backward-compatible alias for classifier experiment dataset creation.
    """

    return create_experiment_datasets(data_cfg=data_cfg, seed=seed)


def create_raw_train_dataset_for_generation(data_cfg: Dict, seed: int) -> Dataset:
    """
    Backward-compatible alias for raw full/few-shot source train dataset creation.
    """

    return create_raw_source_train_dataset(data_cfg=data_cfg, seed=seed)
