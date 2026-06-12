# Course_work_HSE_2026

Codebase for comparing image augmentation methods in few-shot classification.

The pipeline is split into three independent stages:

1. Source/few-shot dataset construction.
   - Base dataset loading lives in `data_utils/base_dataset.py`.
   - Full/few-shot train subset creation lives in `data_utils/few_shot_dataset.py`.
   - Few-shot indices can be loaded from `few_shot.indices_path` or saved via `few_shot.save_indices_path`.

2. Offline augmentation dataset generation.
   - Entry point: `generate_augmented_dataset.py`.
   - It loads only the source train dataset, creates an augmentation provider, writes augmented images and `dataset_index.json`.
   - It does not train classifiers.

3. Classifier experiment training.
   - Entry point: `run_experiment.py`.
   - Training dataset construction lives in `data_utils/experiment_datasets.py`.
   - `train_dataset_type: original` trains on the source full/few-shot data.
   - `train_dataset_type: mixed_aug` / `augmented` / `offline_aug` trains on a previously saved offline augmented dataset.
   - The augmented dataset length is checked against the referenced source/few-shot dataset to prevent accidental mismatches.

Typical usage:

```bash
python generate_augmented_dataset.py --config configs/generate_miniimagenet_5shot_genmix_aug1.yaml
python run_experiment.py --config configs/miniimagenet_resnet18_5shot_genmix_aug1.yaml
```
