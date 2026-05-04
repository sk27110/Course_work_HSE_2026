import random
import os
from typing import List, Optional

import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset

from ..core.base import BaseAugmentationMethod
from ..core.registry import register_augmentation

from diffusers import StableDiffusionInstructPix2PixPipeline


@register_augmentation("genmix")
class GenMixAugmentor(BaseAugmentationMethod):
    """
    Offline GenMix implementation with automatic Kaggle fractal download
    and lazy loading of fractals (no heavy memory usage).
    """

    def __init__(
        self,
        pipe=None,
        dino_model=None,
        fractal_paths: Optional[List[str]] = None,
        device='cuda',
        lambda_fractal=0.2,
        blend_width=20,
        prompts=None,
        kaggle_dataset="tomandjerry2005/fractal-mixing-set-pixmix",
        max_generation_attempts: int = 5,      # ← новый параметр
    ):
        self.device = device
        self.lambda_fractal = lambda_fractal
        self.blend_width = blend_width
        self.kaggle_dataset = kaggle_dataset
        self.max_generation_attempts = max_generation_attempts   # ← сохраняем

        # === Pipeline ===
        if pipe is None:
            self.pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
                "timbrooks/instruct-pix2pix",
                torch_dtype=torch.float16,
                variant="fp16",
            ).to(self.device)
            
            # ←←← ОТКЛЮЧАЕМ SAFETY CHECKER НАВСЕГДА ←←←
            self.pipe.safety_checker = None
            self.pipe.requires_safety_checker = False
        else:
            self.pipe = pipe
            self.pipe.safety_checker = None
            self.pipe.requires_safety_checker = False

        # === DINOv2 (без изменений) ===
        if dino_model is None:
            self.dino = torch.hub.load(
                'facebookresearch/dinov2',
                'dinov2_vitb14',
                pretrained=True
            ).to(self.device).eval()
        else:
            self.dino = dino_model

        # === Prompts (без изменений) ===
        self.prompts = prompts or [
            "A transformed version of image into autumn",
            "A transformed version of image into snowy",
            "A transformed version of image into sunset",
            "A transformed version of image into watercolor art",
            "A transformed version of image into rainbow",
            "A transformed version of image into aurora",
            "A transformed version of image into mosaic",
            "A transformed version of image into ukiyo-e",
            "A transformed version of image into a sketch with crayon",
        ]

        # === Fractals (без изменений) ===
        self.fractal_paths: List[str] = []
        if fractal_paths:
            self.fractal_paths = fractal_paths
        else:
            self._download_fractals_from_kaggle()

        if not self.fractal_paths:
            raise ValueError("No fractal images found!")

        print(f"[GenMix] Loaded {len(self.fractal_paths)} fractal paths (lazy loading)")
        print(f"[GenMix] Safety checker disabled. Max regeneration attempts: {self.max_generation_attempts}")

        self.mu = None
        self.sigma = None

    def _download_fractals_from_kaggle(self):
        """Автоматически скачивает датасет с Kaggle, если пути не были переданы."""
        try:
            import kagglehub
            print(f"[GenMix] Downloading fractal dataset from Kaggle: {self.kaggle_dataset}")
            path = kagglehub.dataset_download(self.kaggle_dataset)
            print(f"[GenMix] Dataset downloaded to: {path}")

            fractal_dir = os.path.join(path, "fractals_and_fvis")

            if not os.path.exists(fractal_dir):
                # fallback — ищем любой подкаталог с изображениями
                for root, dirs, files in os.walk(path):
                    if any(f.lower().endswith(('.jpg', '.jpeg', '.png')) for f in files):
                        fractal_dir = root
                        break

            self.fractal_paths = [
                os.path.join(fractal_dir, f)
                for f in os.listdir(fractal_dir)
                if f.lower().endswith((".jpg", ".jpeg", ".png"))
            ]

            print(f"[GenMix] Found {len(self.fractal_paths)} fractal images")

        except Exception as e:
            raise RuntimeError(f"Failed to download fractals from Kaggle: {e}") from e

    # -----------------------------
    # Остальные методы без изменений (кроме augment)
    # -----------------------------

    def _get_features(self, img):
        if isinstance(img, Image.Image):
            img = T.ToTensor()(img)
        if img.dim() == 3:
            img = img.unsqueeze(0)

        img = T.Resize(256)(img)
        img = T.CenterCrop(224)(img)
        img = T.Normalize(mean=(0.485, 0.456, 0.406),
                          std=(0.229, 0.224, 0.225))(img)

        with torch.no_grad():
            feats = self.dino.forward_features(img.to(self.device))["x_norm_clstoken"]

        return F.normalize(feats.squeeze(0), dim=0)

    def prepare(self, dataset: Dataset, num_samples=200):
        # ... (твой оригинальный код prepare без изменений)
        feats = []
        for i in range(min(num_samples, len(dataset))):
            img, _ = dataset[i]
            feats.append(self._get_features(img))

        feats = torch.stack(feats)
        sims = []
        for i in range(len(feats)):
            for j in range(i + 1, len(feats)):
                sim = F.cosine_similarity(feats[i], feats[j], dim=0)
                sims.append(sim.item())

        sims = torch.tensor(sims)
        self.mu = sims.mean().item()
        self.sigma = sims.std().item()
        print(f"[GenMix] μ={self.mu:.4f}, σ={self.sigma:.4f}")

    def is_faithful(self, img, gen):
        f1 = self._get_features(img)
        f2 = self._get_features(gen)
        sim = F.cosine_similarity(f1, f2, dim=0)
        threshold = self.mu - 2 * self.sigma
        return sim.item() >= threshold

    def _create_mask(self, h: int, w: int, mask_type: str = 'vertical', flip: bool = False) -> torch.Tensor:
        """
        Создаёт маску для бесшовного смешивания (Seamless Concatenation).

        Параметры:
            h, w    – размеры изображения (высота, ширина).
            mask_type – 'vertical' (левая/правая части) или 'horizontal' (верх/низ).
            flip    – если True, маска инвертируется (меняются местами источник и генерация).

        Возвращает:
            Тензор маски формы (1, h, w) на self.device.
        """
        bw = self.blend_width                     # ширина градиентного перехода
        mask = torch.zeros((h, w), device=self.device)

        if mask_type == 'vertical':
            # Левая часть = 0 (оригинал), правая = 1 (генерация)
            mid = w // 2
            mask[:, :mid - bw] = 0.0
            mask[:, mid + bw:] = 1.0

            # Градиентная полоса перехода
            blend = torch.linspace(0.0, 1.0, 2 * bw, device=self.device)
            mask[:, mid - bw : mid + bw] = blend.unsqueeze(0)   # (2*bw) -> (1, 2*bw)

        elif mask_type == 'horizontal':
            # Верх = 0, низ = 1
            mid = h // 2
            mask[:mid - bw, :] = 0.0
            mask[mid + bw:, :] = 1.0

            blend = torch.linspace(0.0, 1.0, 2 * bw, device=self.device)
            mask[mid - bw : mid + bw, :] = blend.unsqueeze(1)   # (2*bw) -> (2*bw, 1)

        else:
            raise ValueError(f"Unknown mask_type: {mask_type}. Choose 'vertical' or 'horizontal'.")

        # Инверсия маски для перестановки частей (flipped version)
        if flip:
            mask = 1.0 - mask

        # Добавляем размерность канала для удобного поэлементного умножения
        return mask.unsqueeze(0)   # (1, h, w)
    # -----------------------------
    # Главный метод augment — теперь с ленивой загрузкой
    # -----------------------------
    def augment(self, image: Image.Image):
        """Генерирует + проверяет DINO. При отсеивании — автоматически перегенерирует."""
        for attempt in range(1, self.max_generation_attempts + 1):
            prompt = random.choice(self.prompts)

            # Генерация
            generated = self.pipe(
                prompt,
                image=image,
                num_inference_steps=15,
                image_guidance_scale=2.0,
                guidance_scale=4.0
            ).images[0]

            # Приводим к тензорам одинакового размера
            img_t = T.ToTensor()(image).to(self.device)
            gen_t = T.ToTensor()(generated).to(self.device)

            h, w = img_t.shape[1:]
            gen_t = torch.nn.functional.interpolate(
                gen_t.unsqueeze(0), size=(h, w), mode='bilinear'
            ).squeeze(0)

            # Проверка верности через DINOv2
            if self.is_faithful(img_t, gen_t):
                # Успех! Идём дальше — делаем hybrid + fractal
                # Случайный выбор из 4 вариантов согласно статье
                mask_type = random.choice(['vertical', 'horizontal'])
                flip = random.choice([True, False])
                mask = self._create_mask(h, w, mask_type=mask_type, flip=flip)
                
                hybrid = gen_t * mask + img_t * (1 - mask)

                # Ленивая загрузка фрактала
                fractal_path = random.choice(self.fractal_paths)
                fractal_img = Image.open(fractal_path).convert("RGB")
                fractal = T.ToTensor()(fractal_img.resize((w, h))).to(self.device)

                out = (1 - self.lambda_fractal) * hybrid + self.lambda_fractal * fractal

                return T.ToPILImage()(out.clamp(0, 1))

            else:
                print(f"[GenMix] Attempt {attempt}/{self.max_generation_attempts} failed DINO check. Regenerating...")

        # Если все попытки провалились — возвращаем оригинал (чтобы не ломать пайплайн)
        print(f"[GenMix] All {self.max_generation_attempts} attempts failed. Returning original image.")
        return image