import random
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import List, Dict


class SubsetDataset(Dataset):
    """
    Generic dataset wrapper that selects a subset by indices.
    """

    def __init__(self, dataset: Dataset, indices: List[int]):
        self.dataset = dataset
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]


class FewShotSampler:
    """
    Universal k-shot sampler for any dataset.

    Requires targets (labels) to be provided.
    """

    def __init__(self, seed: int = 42):
        self.seed = seed

    def _set_seed(self):
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

    def sample(
        self,
        dataset: Dataset,
        targets: List[int],
        k: int
    ) -> Dataset:
        """
        Args:
            dataset: any PyTorch dataset
            targets: list of labels (len == len(dataset))
            k: number of samples per class

        Returns:
            SubsetDataset
        """

        if len(targets) != len(dataset):
            raise ValueError("targets length must match dataset length")

        self._set_seed()

        class_to_indices: Dict[int, List[int]] = {}

        # собираем индексы по классам
        for idx, label in enumerate(targets):
            class_to_indices.setdefault(label, []).append(idx)

        selected_indices = []

        for label, indices in class_to_indices.items():
            if len(indices) < k:
                raise ValueError(
                    f"Class {label} has only {len(indices)} samples (< {k})"
                )

            chosen = random.sample(indices, k)
            selected_indices.extend(chosen)

        random.shuffle(selected_indices)

        return SubsetDataset(dataset, selected_indices)