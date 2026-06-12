# training/dataset.py

"""
Backward-compatible training dataset aliases.

Dataset implementations were moved out of the training package so classifier
training stays independent from dataset/augmentation construction details.
Use data_utils.mixed_aug_dataset.MixedAugDataset in new code.
"""

from data_utils.mixed_aug_dataset import MixedAugDataset

__all__ = ["MixedAugDataset"]
