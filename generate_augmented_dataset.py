# generate_augmented_dataset.py

import argparse
from typing import Dict

import yaml

from augmentlib.method_factory import create_augmentation_provider
from data_utils.dataset_factory import create_raw_train_dataset_for_generation
from data_utils.offline_aug_generator import OfflineAugmentedDatasetGenerator
from training.trainer import set_seed


def load_config(config_path: str) -> Dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to generation YAML config.",
    )

    args = parser.parse_args()

    config = load_config(args.config)

    seed = int(config.get("seed", 42))
    set_seed(seed)

    data_cfg = config["data"]
    aug_cfg = config["augmentation"]

    dataset = create_raw_train_dataset_for_generation(
        data_cfg=data_cfg,
        seed=seed,
    )

    provider = create_augmentation_provider(
        aug_cfg=aug_cfg,
        dataset=dataset,
    )

    generator = OfflineAugmentedDatasetGenerator(
        provider=provider,
        save_dir=aug_cfg["save_dir"],
        num_aug=int(aug_cfg.get("num_aug", 1)),
        max_tries=int(aug_cfg.get("max_tries", 10)),
        save_original=bool(aug_cfg.get("save_original", True)),
    )

    print("=" * 90)
    print("Generating offline augmented dataset")
    print(f"Dataset name: {data_cfg.get('dataset_name')}")
    print(f"Dataset size: {len(dataset)}")
    print(f"Save dir: {aug_cfg['save_dir']}")
    print(f"Method: {aug_cfg['method']}")
    print(f"Num augmentations per image: {aug_cfg.get('num_aug', 1)}")
    print("=" * 90)

    generator.generate(dataset)


if __name__ == "__main__":
    main()
