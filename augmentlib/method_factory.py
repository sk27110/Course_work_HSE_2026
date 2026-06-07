# augmentlib/method_factory.py

from typing import Dict, Optional

import torch


class BaseMethodProvider:
    """
    Provider interface.

    Offline generator will call:
      method = provider.get_method(label)
      aug = method.augment(image)
    """

    def prepare(self, dataset):
        pass

    def get_method(self, label: int):
        raise NotImplementedError


class SingleMethodProvider(BaseMethodProvider):
    """
    For methods that do not depend on class label.
    Example: GenMix.
    """

    def __init__(self, method):
        self.method = method

    def prepare(self, dataset):
        if hasattr(self.method, "prepare"):
            self.method.prepare(dataset)

    def get_method(self, label: int):
        return self.method


class PerClassMethodProvider(BaseMethodProvider):
    """
    For methods that require class_name.
    Example: AGA.

    It creates one augmentor per class lazily.
    """

    def __init__(
        self,
        method_cls,
        class_names: list[str],
        common_kwargs: Optional[Dict] = None,
    ):
        self.method_cls = method_cls
        self.class_names = class_names
        self.common_kwargs = common_kwargs or {}
        self.cache = {}

    def prepare(self, dataset):
        # Usually we do not call prepare for every class-specific model,
        # because AGA has no dataset-level prepare.
        pass

    def get_method(self, label: int):
        label = int(label)

        if label not in self.cache:
            if label < 0 or label >= len(self.class_names):
                raise ValueError(
                    f"Label {label} is out of range for class_names "
                    f"of length {len(self.class_names)}"
                )

            class_name = self.class_names[label]

            print(f"[MethodFactory] Creating class-specific augmentor: label={label}, class_name='{class_name}'")

            self.cache[label] = self.method_cls(
                class_name=class_name,
                **self.common_kwargs,
            )

        return self.cache[label]


def _parse_torch_dtype(dtype_name: Optional[str]):
    if dtype_name is None:
        return None

    dtype_name = str(dtype_name).lower()

    if dtype_name in ["float16", "fp16", "half"]:
        return torch.float16

    if dtype_name in ["float32", "fp32"]:
        return torch.float32

    if dtype_name in ["bfloat16", "bf16"]:
        return torch.bfloat16

    raise ValueError(f"Unsupported torch dtype: {dtype_name}")


def create_augmentation_provider(aug_cfg: Dict, dataset):
    """
    Creates augmentation provider from YAML config.

    Supports:
    - genmix
    - aga
    - identity
    """

    method_name = aug_cfg["method"].lower()

    if method_name == "identity":
        return SingleMethodProvider(IdentityAugmentationMethod())

    if method_name == "genmix":
        from augmentlib.methods.genmix import GenMixAugmentor

        method = GenMixAugmentor(
            device=aug_cfg.get("device", "cuda"),
            lambda_fractal=float(aug_cfg.get("lambda_fractal", 0.2)),
            blend_width=int(aug_cfg.get("blend_width", 20)),
            prompts=aug_cfg.get("prompts", None),
            kaggle_dataset=aug_cfg.get(
                "kaggle_dataset",
                "tomandjerry2005/fractal-mixing-set-pixmix",
            ),
            max_generation_attempts=int(
                aug_cfg.get("max_generation_attempts", 5)
            ),
        )

        return SingleMethodProvider(method)

    if method_name == "aga":
        from augmentlib.methods.aga import AGAAugmentor

        if not hasattr(dataset, "class_names") or dataset.class_names is None:
            raise ValueError(
                "AGA requires dataset.class_names, but dataset does not have it."
            )

        common_kwargs = {
            "device": aug_cfg.get("device", None),
            "phi3_model_name": aug_cfg.get(
                "phi3_model_name",
                "microsoft/Phi-3-mini-4k-instruct",
            ),
            "sd_model_name": aug_cfg.get(
                "sd_model_name",
                "runwayml/stable-diffusion-v1-5",
            ),
            "dino_model_name": aug_cfg.get(
                "dino_model_name",
                "IDEA-Research/grounding-dino-base",
            ),
            "sam2_model_name": aug_cfg.get(
                "sam2_model_name",
                "facebook/sam2-hiera-large",
            ),
            "box_threshold": float(aug_cfg.get("box_threshold", 0.25)),
            "text_threshold": float(aug_cfg.get("text_threshold", 0.25)),
            "guidance_scale": float(aug_cfg.get("guidance_scale", 7.5)),
            "num_inference_steps": int(aug_cfg.get("num_inference_steps", 50)),
            "sd_torch_dtype": _parse_torch_dtype(
                aug_cfg.get("sd_torch_dtype", None)
            ),
            "load_models": bool(aug_cfg.get("load_models", True)),
        }

        return PerClassMethodProvider(
            method_cls=AGAAugmentor,
            class_names=dataset.class_names,
            common_kwargs=common_kwargs,
        )

    raise ValueError(f"Unknown augmentation method: {method_name}")


class IdentityAugmentationMethod:
    def prepare(self, dataset):
        pass

    def augment(self, image):
        return image.copy()
