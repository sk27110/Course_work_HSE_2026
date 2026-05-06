import os
import json
from tqdm import tqdm


class NewOfflineAugmentedDatasetGenerator:
    def __init__(self, method, save_dir, num_aug=1, max_tries=10):
        self.method = method
        self.save_dir = save_dir
        self.num_aug = num_aug
        self.max_tries = max_tries

    def generate(self, dataset):
        """
        dataset: любой объект, поддерживающий len и индексацию,
                 возвращающий (PIL.Image, int label)
        """
        os.makedirs(self.save_dir, exist_ok=True)

        # Подготовка метода (например, обучение Textual Inversion в DA-Fusion)
        self.method.prepare(dataset)

        index = []  # список записей для общего индекса

        for idx in tqdm(range(len(dataset)), desc="Generating augmented data"):
            img, label = dataset[idx]

            # Папка класса
            class_dir = os.path.join(self.save_dir, str(label))
            os.makedirs(class_dir, exist_ok=True)

            # Сохраняем оригинал
            orig_name = f"{idx}_orig.jpg"
            orig_path = os.path.join(class_dir, orig_name)
            img.save(orig_path)

            # Генерируем аугментации
            aug_names = []
            count = 0
            tries = 0
            while count < self.num_aug and tries < self.max_tries:
                aug = self.method.augment(img)
                if aug is not None:
                    aug_name = f"{idx}_aug_{count}.jpg"
                    aug_path = os.path.join(class_dir, aug_name)
                    aug.save(aug_path)
                    aug_names.append(aug_name)
                    count += 1
                tries += 1

            # Добавляем запись в индекс
            # Используем относительные пути от save_dir (удобно при переносе)
            rel_class_dir = str(label)  # например "0", "1", ...
            entry = {
                "orig": os.path.join(rel_class_dir, orig_name),
                "augs": [os.path.join(rel_class_dir, name) for name in aug_names],
                "label": label
            }
            index.append(entry)

        # Сохраняем общий индекс
        index_path = os.path.join(self.save_dir, "dataset_index.json")
        with open(index_path, "w") as f:
            json.dump(index, f, indent=2)

        print(f"Saved index to {index_path}")