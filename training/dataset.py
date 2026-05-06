import os
import json
import random
from PIL import Image
from torch.utils.data import Dataset


class MixedAugDataset(Dataset):
    def __init__(self, root, index_path, transform=None, alpha=0.5):
        """
        root: путь к save_dir, где лежат папки классов и index.json
        index_path: путь к JSON-индексу (или можно передать сам список)
        transform: torchvision-трансформации
        alpha: вероятность использовать синтетическое изображение
        """
        self.root = root
        self.transform = transform
        self.alpha = alpha

        with open(index_path, "r") as f:
            self.index = json.load(f)   # список словарей [{"orig": ..., "augs": ..., "label": ...}, ...]

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        entry = self.index[idx]

        # Решаем, что брать: синтетику или оригинал
        if random.random() < self.alpha and entry["augs"]:
            # Берём случайный аугмент
            img_rel_path = random.choice(entry["augs"])
        else:
            # Берём оригинал
            img_rel_path = entry["orig"]

        full_path = os.path.join(self.root, img_rel_path)
        img = Image.open(full_path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        return img, entry["label"]