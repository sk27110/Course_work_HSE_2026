# data_utils/offline_aug_generator.py

import json
import os
from typing import Any

from tqdm import tqdm


class OfflineAugmentedDatasetGenerator:
    """
    Generates offline augmentations and writes dataset_index.json.

    Expected dataset item:
      PIL.Image, int label

    provider interface:
      provider.prepare(dataset)
      provider.get_method(label).augment(image)
    """

    def __init__(
        self,
        provider: Any,
        save_dir: str,
        num_aug: int = 1,
        max_tries: int = 10,
        save_original: bool = True,
    ):
        self.provider = provider
        self.save_dir = save_dir
        self.num_aug = int(num_aug)
        self.max_tries = int(max_tries)
        self.save_original = bool(save_original)

        if self.num_aug < 0:
            raise ValueError(f"num_aug must be non-negative, got {self.num_aug}")

        if self.max_tries <= 0:
            raise ValueError(f"max_tries must be positive, got {self.max_tries}")

    def _get_source_idx(self, dataset, idx: int) -> int:
        if hasattr(dataset, "indices"):
            return int(dataset.indices[idx])
        return int(idx)

    def generate(self, dataset):
        os.makedirs(self.save_dir, exist_ok=True)

        if hasattr(self.provider, "prepare"):
            self.provider.prepare(dataset)

        index = []

        for idx in tqdm(range(len(dataset)), desc="Generating augmented data"):
            img, label = dataset[idx]
            label = int(label)

            source_idx = self._get_source_idx(dataset, idx)

            class_dir = os.path.join(self.save_dir, str(label))
            os.makedirs(class_dir, exist_ok=True)

            orig_name = f"{source_idx}_orig.jpg"
            orig_path = os.path.join(class_dir, orig_name)

            if self.save_original:
                img.save(orig_path)

            aug_names = []
            count = 0
            tries = 0

            while count < self.num_aug and tries < self.max_tries:
                method = self.provider.get_method(label)
                aug = method.augment(img)

                if aug is not None:
                    aug_name = f"{source_idx}_aug_{count}.jpg"
                    aug_path = os.path.join(class_dir, aug_name)

                    aug.save(aug_path)

                    aug_names.append(aug_name)
                    count += 1

                tries += 1

            rel_class_dir = str(label)

            entry = {
                "source_idx": source_idx,
                "orig": os.path.join(rel_class_dir, orig_name),
                "augs": [
                    os.path.join(rel_class_dir, aug_name)
                    for aug_name in aug_names
                ],
                "label": label,
            }

            index.append(entry)

        index_path = os.path.join(self.save_dir, "dataset_index.json")

        with open(index_path, "w", encoding="utf-8") as f:
            json.dump(index, f, indent=2)

        print(f"Saved index to {index_path}")
        print(f"Dataset size: {len(index)}")
