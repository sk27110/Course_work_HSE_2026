import random
from typing import List

import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset

from ..core.base import BaseAugmentationMethod
from ..core.registry import register_augmentation

@register_augmentation("genmix")
class GenMixAugmentor(BaseAugmentationMethod):
    """
    Offline GenMix implementation.
    """

    def __init__(
        self,
        pipe,
        dino_model,
        fractal_images: List[Image.Image],
        device='cuda',
        lambda_fractal=0.2,
        blend_width=20,
        prompts=None
    ):
        self.pipe = pipe
        self.dino = dino_model
        self.device = device
        self.lambda_fractal = lambda_fractal
        self.blend_width = blend_width

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

        self.fractals = fractal_images

        self.mu = None
        self.sigma = None
    def _get_features(self, img):
        if isinstance(img, Image.Image):
            img = T.ToTensor()(img)

        if img.dim() == 3:
            img = img.unsqueeze(0)

        img = T.Resize(256)(img)
        img = T.CenterCrop(224)(img)

        img = T.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        )(img)

        with torch.no_grad():
            feats = self.dino.forward_features(img.to(self.device))["x_norm_clstoken"]

        return F.normalize(feats.squeeze(0), dim=0)

    # -----------------------------
    # Prepare (compute μ, σ)
    # -----------------------------
    def prepare(self, dataset: Dataset, num_samples=200):
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

    # -----------------------------
    # Mask
    # -----------------------------
    def _create_mask(self, h, w):
        mask = torch.zeros((h, w))
        bw = self.blend_width

        mask[:, :w//2 - bw] = 0
        mask[:, w//2 + bw:] = 1

        blend = torch.linspace(0, 1, 2 * bw)
        mask[:, w//2 - bw:w//2 + bw] = blend.unsqueeze(0)

        return mask.unsqueeze(0).to(self.device)

    # -----------------------------
    # Main augment
    # -----------------------------
    def augment(self, image: Image.Image):
        prompt = random.choice(self.prompts)

        generated = self.pipe(
            prompt,
            image=image,
            num_inference_steps=15,
            image_guidance_scale=2.0,
            guidance_scale=4.0
        ).images[0]

        img_t = T.ToTensor()(image).to(self.device)
        gen_t = T.ToTensor()(generated).to(self.device)

        h, w = img_t.shape[1:]
        gen_t = torch.nn.functional.interpolate(
            gen_t.unsqueeze(0), size=(h, w), mode='bilinear'
        ).squeeze(0)

        if not self.is_faithful(img_t, gen_t):
            return None

        mask = self._create_mask(h, w)
        hybrid = gen_t * mask + img_t * (1 - mask)

        fractal_path = random.choice(self.fractals)
        fractal_img = Image.open(fractal_path).convert("RGB")
        fractal = T.ToTensor()(fractal_img.resize((w, h))).to(self.device)

        out = (1 - self.lambda_fractal) * hybrid + self.lambda_fractal * fractal

        return T.ToPILImage()(out.clamp(0, 1))
