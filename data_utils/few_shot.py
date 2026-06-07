# data_utils/few_shot.py

import json
import random
from typing import Dict, List, Optional

import torch
from torch.utils.data import Dataset


class SubsetDataset(Dataset):
    """
    Dataset wrapper that selects a subset by indices.

    Keeps:
    - source indices;
    - targets;
    - class_names if base dataset has them.
    """

    def __init__(self, dataset: Dataset, indices: List[int]):
        self.dataset = dataset
        self.indices = [int(i) for i in indices]

        if hasattr(dataset, "class_names"):
            self.class_names = dataset.class_names
        else:
            self.class_names = None

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        source_idx = self.indices[idx]
        return self.dataset[source_idx]

    @property
    def targets(self) -> List[int]:
        base_targets = get_dataset_targets(self.dataset)
        return [int(base_targets[i]) for i in self.indices]

    def save_indices(self, path: str):
        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.indices, f, indent=2)

    @classmethod
    def from_indices_file(cls, dataset: Dataset, path: str):
        with open(path, "r", encoding="utf-8") as f:
            indices = json.load(f)
        return cls(dataset=dataset, indices=indices)



def get_dataset_targets(dataset: Dataset) -> List[int]:
    """
    Universal target extractor.

    For stable few-shot experiments, every dataset wrapper should ideally expose .targets.
    """

    if hasattr(dataset, "targets"):
        targets = dataset.targets
        return [int(x) for x in targets]

    if hasattr(dataset, "_labels"):
        targets = dataset._labels
        return [int(x) for x in targets]

    if hasattr(dataset, "labels"):
        targets = dataset.labels
        return [int(x) for x in targets]

    raise ValueError(
        "Cannot infer targets for dataset. "
        "Please add a .targets property to the dataset wrapper."
    )


class FewShotSampler:
    """
    Reproducible k-shot sampler.

    Samples exactly k images per class from the given dataset.
    """

    def __init__(self, seed: int = 42):
        self.seed = int(seed)

    def sample_indices(self, targets: List[int], k: int) -> List[int]:
        if k <= 0:
            raise ValueError(f"k must be positive, got k={k}")

        rng = random.Random(self.seed)

        class_to_indices: Dict[int, List[int]] = {}

        for idx, label in enumerate(targets):
            label = int(label)
            class_to_indices.setdefault(label, []).append(idx)

        selected_indices: List[int] = []

        for label in sorted(class_to_indices.keys()):
            indices = class_to_indices[label]

            if len(indices) < k:
                raise ValueError(
                    f"Class {label} has only {len(indices)} samples, "
                    f"but k={k} was requested."
                )

            chosen = rng.sample(indices, k)
            selected_indices.extend(chosen)

        rng.shuffle(selected_indices)
        return selected_indices

    def sample(self, dataset: Dataset, targets: List[int], k: int) -> SubsetDataset:
        indices = self.sample_indices(targets=targets, k=k)
        return SubsetDataset(dataset=dataset, indices=indices)


def build_few_shot_subset(
    dataset: Dataset,
    k: int,
    seed: int,
    indices_path: Optional[str] = None,
    save_indices_path: Optional[str] = None,
) -> SubsetDataset:
    """
    Builds a few-shot subset.

    If indices_path is provided, loads fixed indices from json.
    Otherwise samples new indices using k and seed.
    """

    if indices_path is not None:
        subset = SubsetDataset.from_indices_file(dataset=dataset, path=indices_path)
    else:
        targets = get_dataset_targets(dataset)
        sampler = FewShotSampler(seed=seed)
        subset = sampler.sample(dataset=dataset, targets=targets, k=k)

    if save_indices_path is not None:
        subset.save_indices(save_indices_path)

    return subset
