import os
from tqdm import tqdm


class OfflineAugmentedDatasetGenerator:
    """
    Generates augmented dataset on disk.
    """

    def __init__(self, method, save_dir, num_aug=1, max_tries=10):
        self.method = method
        self.save_dir = save_dir
        self.num_aug = num_aug
        self.max_tries = max_tries

    def generate(self, dataset):
        os.makedirs(self.save_dir, exist_ok=True)

        self.method.prepare(dataset)

        for idx in tqdm(range(len(dataset))):
            img, label = dataset[idx]

            class_dir = os.path.join(self.save_dir, str(label))
            os.makedirs(class_dir, exist_ok=True)

            img.save(os.path.join(class_dir, f"{idx}_orig.jpg"))

            count = 0
            tries = 0

            while count < self.num_aug and tries < self.max_tries:
                aug = self.method.augment(img)

                if aug is not None:
                    aug.save(os.path.join(class_dir, f"{idx}_aug_{count}.jpg"))
                    count += 1

                tries += 1