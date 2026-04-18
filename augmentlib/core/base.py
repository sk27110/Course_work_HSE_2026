from abc import ABC, abstractmethod
from typing import Optional
from PIL import Image
from torch.utils.data import Dataset


class BaseAugmentationMethod(ABC):
    """
    Base interface for all augmentation methods.
    """

    def prepare(self, dataset: Dataset):
        """
        Optional precomputation step.
        """
        pass

    @abstractmethod
    def augment(self, image: Image.Image) -> Optional[Image.Image]:
        """
        Apply augmentation.

        Returns:
            Augmented image or None (if rejected)
        """
        pass