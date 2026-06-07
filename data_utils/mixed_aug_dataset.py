# data_utils/mixed_aug_dataset.py

import json
import os
import random
from typing import Any, Dict, List

from PIL import Image
from torch.utils.data import Dataset


class MixedAugDataset(Dataset):
    """
    Dataset for offline-generated augmentations.

    Index format:
    [
      {
        "source_idx": 123,
        "orig": "5/123_orig.jpg",
        "augs": ["5/123_aug_0.jpg", "5/123_aug_1.jpg"],
        "label": 5
      },
      ...
    ]

    alpha:
      probability of taking synthetic augmentation instead of saved original.
    """

    def __init__(
        self,
        root: str,
        index_path: str,
        transform=None,
        alpha: float = 0.5,
    ):
        self.root = root
        self.index_path = index_path
        self.transform = transform
        self.alpha = float(alpha)

        if not 0.0 <= self.alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {self.alpha}")

        if not os.path.exists(index_path):
            raise FileNotFoundError(f"Augmentation index not found: {index_path}")

        with open(index_path, "r", encoding="utf-8") as f:
            self.index: List[Dict[str, Any]] = json.load(f)

        if not isinstance(self.index, list):
            raise ValueError("Augmentation index must be a list of entries.")

        self._validate_index()

    def _validate_index(self):
        required_keys = {"orig", "augs", "label"}

        for i, entry in enumerate(self.index):
            missing = required_keys - set(entry.keys())
            if missing:
                raise ValueError(
                    f"Entry {i} in {self.index_path} misses keys: {missing}"
                )

            if not isinstance(entry["augs"], list):
                raise ValueError(f"Entry {i}: 'augs' must be a list.")

    def __len__(self):
        return len(self.index)

    @property
    def targets(self):
        return [int(entry["label"]) for entry in self.index]

    def __getitem__(self, idx):
        entry = self.index[idx]

        use_aug = random.random() < self.alpha and len(entry["augs"]) > 0

        if use_aug:
            rel_path = random.choice(entry["augs"])
        else:
            rel_path = entry["orig"]

        full_path = os.path.join(self.root, rel_path)

        if not os.path.exists(full_path):
            raise FileNotFoundError(f"Image file not found: {full_path}")

        image = Image.open(full_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        label = int(entry["label"])

        return image, label
