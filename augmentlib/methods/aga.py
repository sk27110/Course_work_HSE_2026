# augmentlib/methods/aga.py

import random
import numpy as np
import cv2
import torch
from PIL import Image
from typing import Optional

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoProcessor,
    AutoModelForZeroShotObjectDetection,
)
from diffusers import StableDiffusionPipeline
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.build_sam import build_sam2_hf

from ..core.base import BaseAugmentationMethod
from ..core.registry import register_augmentation

# -------------------------------
# Сам класс аугментации
# -------------------------------

@register_augmentation("aga")
class AGAAugmentor(BaseAugmentationMethod):
    """
    Adaptive Generative Augmentation (AGA) with DINO + SAM 2 mask extraction,
    Phi‑3 background prompt generation, Stable Diffusion background generation,
    affine object transformation and blending.

    Parameters
    ----------
    class_name : str
        Target class name for object detection and prompt generation.
    device : str, optional ('cuda' or 'cpu'), auto-detected.
    phi3_model_name : str
    sd_model_name : str
    dino_model_name : str
    sam2_model_name : str
    box_threshold : float
    guidance_scale : float
    num_inference_steps : int
    sd_torch_dtype : torch.dtype
    load_models : bool, load heavy models immediately (default True).
    """

    def __init__(
        self,
        class_name: str,
        device: Optional[str] = None,
        phi3_model_name: str = "microsoft/Phi-3-mini-4k-instruct",
        sd_model_name: str = "runwayml/stable-diffusion-v1-5",
        dino_model_name: str = "IDEA-Research/grounding-dino-base",
        sam2_model_name: str = "facebook/sam2-hiera-large",
        box_threshold: float = 0.25,
        text_threshold: float = 0.25,
        guidance_scale: float = 7.5,
        num_inference_steps: int = 50,
        sd_torch_dtype: Optional[torch.dtype] = None,
        load_models: bool = True,
    ):
        super().__init__()
        self.class_name = class_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.phi3_model_name = phi3_model_name
        self.sd_model_name = sd_model_name
        self.dino_model_name = dino_model_name
        self.sam2_model_name = sam2_model_name
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.guidance_scale = guidance_scale
        self.num_inference_steps = num_inference_steps
        self.sd_torch_dtype = sd_torch_dtype or (
            torch.float16 if self.device == "cuda" else torch.float32
        )

        # Контейнеры для моделей
        self.phi3_tokenizer = None
        self.phi3_model = None
        self.dino_processor = None
        self.dino_model = None
        self.sd_pipe = None

        # SAM2 загружается по требованию (т.к. привязан к конкретному изображению)
        self._sam2_predictor = None  # кэш предиктора

        # Шаблоны промптов (как раньше)
        self.instructions = [
            "Generate a realistic background of",
            "Create a vivid scene of",
        ]
        self.backgrounds = [
            "a futuristic city",
            "a dense forest",
            "a desert landscape",
            "a snowy mountain",
            "a tropical beach",
            "a jungle with waterfalls",
            "an abandoned industrial zone",
        ]
        self.temporal = [
            "at sunrise",
            "at sunset",
            "during golden hour",
            "on a rainy day",
            "at night with neon lights",
        ]

        if load_models:
            self._load_models()

    # ------------------------------------------------------------
    # Загрузка моделей
    # ------------------------------------------------------------
    def _load_models(self):
        """Загружает Phi‑3, Grounding DINO и Stable Diffusion."""
        print("[AGA] Loading models...")

        # Phi‑3
        print("  - Phi‑3...")
        self.phi3_tokenizer = AutoTokenizer.from_pretrained(self.phi3_model_name)
        self.phi3_model = AutoModelForCausalLM.from_pretrained(
            self.phi3_model_name,
            device_map="auto",
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
        )
        if self.phi3_tokenizer.pad_token is None:
            self.phi3_tokenizer.pad_token = self.phi3_tokenizer.eos_token
        self.phi3_model.config.pad_token_id = self.phi3_tokenizer.pad_token_id

        # Grounding DINO
        print("  - Grounding DINO...")
        self.dino_processor = AutoProcessor.from_pretrained(self.dino_model_name)
        self.dino_model = AutoModelForZeroShotObjectDetection.from_pretrained(
            self.dino_model_name
        ).to(self.device)

        # Stable Diffusion
        print("  - Stable Diffusion...")
        self.sd_pipe = StableDiffusionPipeline.from_pretrained(
            self.sd_model_name,
            torch_dtype=self.sd_torch_dtype,
            safety_checker=None,
        ).to(self.device)
        self.sd_pipe.enable_model_cpu_offload()
        print("[AGA] All models loaded.")

    def _get_sam2_predictor(self):
        """Ленивая загрузка SAM2 (один предиктор на экземпляр)."""
        if self._sam2_predictor is None:
            print("[AGA] Loading SAM2...")
            self._sam2_predictor = SAM2ImagePredictor(
                build_sam2_hf(self.sam2_model_name, device=self.device)
            )
        return self._sam2_predictor

    # ------------------------------------------------------------
    # Генератор промпта (Phi‑3)
    # ------------------------------------------------------------
    @staticmethod
    def _normalize_class(class_name: str) -> str:
        mapping = {
            "persian cat": "cat",
            "siamese cat": "cat",
            "golden retriever": "dog",
            "anthurium warocqueanum": "plant",
        }
        return mapping.get(class_name.lower(), class_name.lower())

    def _build_base_prompt(self) -> str:
        bgr = random.choice(self.backgrounds)
        temp = random.choice(self.temporal)
        return f"{bgr} {temp}"

    def _extract_answer(self, text: str) -> str:
        if "<|assistant|>" in text:
            text = text.split("<|assistant|>")[-1]
        if "Scene:" in text:
            text = text.split("Scene:")[-1]
        lines = [l.strip() for l in text.split("\n") if l.strip()]
        if lines:
            text = lines[-1]
        return text.strip(" .,")

    def generate_prompt(self) -> str:
        """Генерирует фоновый промпт с помощью Phi‑3."""
        class_name_norm = self._normalize_class(self.class_name)
        base_prompt = self._build_base_prompt()

        messages = [
            {
                "role": "system",
                "content": "You generate short prompts for image generation.",
            },
            {
                "role": "user",
                "content": f"""
Generate ONE short background description.

Rules:
- Do NOT mention: {class_name_norm}
- Only describe environment
- One sentence only
- Focus on lighting, atmosphere, textures
- Return ONLY the description
- Keep it short (10-15 words)

Scene: {base_prompt}
""",
            },
        ]

        prompt = self.phi3_tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.phi3_tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.phi3_model.generate(
                **inputs,
                max_new_tokens=30,
                temperature=0.4,
                top_p=0.95,
                do_sample=True,
                pad_token_id=self.phi3_tokenizer.pad_token_id,
            )

        raw_text = self.phi3_tokenizer.decode(outputs[0], skip_special_tokens=True)
        text = self._extract_answer(raw_text)
        text = text.replace("\n", " ").strip()
        text += ", cinematic lighting, highly detailed, realistic, 4k"
        return text

    # ------------------------------------------------------------
    # Детектор маски (DINO + SAM2)
    # ------------------------------------------------------------
    def _detect_and_mask(self, image_np: np.ndarray) -> Optional[np.ndarray]:
        """
        Возвращает бинарную маску объекта (H,W) или None.
        """
        h, w = image_np.shape[:2]
        pil_image = Image.fromarray(image_np)

        # 1. Grounding DINO
        text_prompt = f"{self.class_name} . "
        inputs = self.dino_processor(images=pil_image, text=text_prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.dino_model(**inputs)

        target_sizes = torch.tensor([pil_image.size[::-1]]).to(self.device)
        results = self.dino_processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=self.box_threshold,
            text_threshold=self.text_threshold,
            target_sizes=target_sizes,
        )[0]

        boxes = results["boxes"]
        scores = results["scores"]

        if len(boxes) == 0:
            print("[AGA] DINO found no objects. Trying fallback SAM2.")
            return self._fallback_sam2(image_np)

        print(f"[AGA] DINO found {len(boxes)} object(s).")

        # 2. SAM2
        predictor = self._get_sam2_predictor()
        predictor.set_image(image_np)

        combined_mask = np.zeros((h, w), dtype=np.uint8)

        for i, (box, score) in enumerate(zip(boxes, scores)):
            x1, y1, x2, y2 = box.tolist()
            if x2 <= x1 or y2 <= y1:
                continue

            input_box = np.array([[x1, y1, x2, y2]])
            masks, scores_iou, _ = predictor.predict(
                point_coords=None,
                point_labels=None,
                box=input_box,
                multimask_output=False,
            )
            if masks is None or len(masks) == 0:
                continue

            mask_np = masks[0].astype(np.uint8)
            combined_mask = np.logical_or(combined_mask, mask_np).astype(np.uint8)
            print(f"  Object {i+1}: mask added (score {score:.3f}, IoU {scores_iou[0]:.3f})")

        if combined_mask.sum() == 0:
            print("[AGA] SAM2 could not create masks. Fallback.")
            return self._fallback_sam2(image_np)

        return combined_mask

    def _fallback_sam2(self, image_np: np.ndarray) -> np.ndarray:
        """Fallback – одна точка по центру."""
        predictor = self._get_sam2_predictor()
        predictor.set_image(image_np)
        h, w = image_np.shape[:2]
        masks, _, _ = predictor.predict(
            point_coords=np.array([[w // 2, h // 2]]),
            point_labels=np.array([1]),
            multimask_output=False,
        )
        return masks[0].astype(np.uint8)

    # ------------------------------------------------------------
    # Аффинное преобразование и смешивание
    # ------------------------------------------------------------
    @staticmethod
    def _apply_affine_transform(image: np.ndarray, mask: np.ndarray):
        h, w = image.shape[:2]
        angle = random.uniform(-25, 25)
        scale = random.uniform(0.6, 1.2)
        tx = random.uniform(-0.3 * w, 0.3 * w)
        ty = random.uniform(-0.3 * h, 0.3 * h)

        M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, scale)
        M[:, 2] += [tx, ty]

        transformed_img = cv2.warpAffine(
            image, M, (w, h), flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0),
        )
        transformed_mask = cv2.warpAffine(
            mask.astype(np.uint8) * 255, M, (w, h), flags=cv2.INTER_NEAREST,
        )
        transformed_mask = (transformed_mask > 0).astype(np.uint8)
        # Мягкое размытие маски
        transformed_mask = cv2.GaussianBlur(transformed_mask.astype(float), (11, 11), 0)
        transformed_mask = np.clip(transformed_mask, 0, 1)
        return transformed_img, transformed_mask

    @staticmethod
    def _blend_images(background: np.ndarray, foreground: np.ndarray, mask: np.ndarray) -> np.ndarray:
        h, w = background.shape[:2]
        foreground = cv2.resize(foreground, (w, h))
        mask_float = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(np.float32)
        mask_float = cv2.GaussianBlur(mask_float, (15, 15), 0)
        mask_float = np.expand_dims(mask_float, axis=2)

        result = foreground * mask_float + background * (1 - mask_float)
        return np.clip(result, 0, 255).astype(np.uint8)

    # ------------------------------------------------------------
    # Генерация фона
    # ------------------------------------------------------------
    def _generate_background(self, image_np: np.ndarray, prompt: str) -> np.ndarray:
        orig_h, orig_w = image_np.shape[:2]
        gen_h = (orig_h // 8) * 8
        gen_w = (orig_w // 8) * 8

        generated = self.sd_pipe(
            prompt=prompt,
            height=gen_h,
            width=gen_w,
            guidance_scale=self.guidance_scale,
            num_inference_steps=self.num_inference_steps,
        ).images[0]

        generated = generated.resize((orig_w, orig_h), Image.LANCZOS)
        return np.array(generated)

    # ------------------------------------------------------------
    # Главный метод аугментации
    # ------------------------------------------------------------
    def augment(self, image: Image.Image) -> Optional[Image.Image]:
        """
        Применяет AGA к одному изображению.
        """
        # PIL -> numpy RGB
        original_np = np.array(image.convert("RGB"))

        # 1. Детектируем маску объекта
        mask = self._detect_and_mask(original_np)
        if mask is None or mask.sum() == 0:
            print("[AGA] No mask obtained, skipping image.")
            return None

        # 2. Генерируем промпт фона
        prompt = self.generate_prompt()
        print(f"[AGA] Generated prompt: {prompt}")

        # 3. Генерируем фон
        background_np = self._generate_background(original_np, prompt)

        # 4. Аугментируем объект (аффинное преобразование)
        transformed_img, transformed_mask = self._apply_affine_transform(original_np, mask)

        # 5. Смешиваем фон и объект
        final_np = self._blend_images(background_np, transformed_img, transformed_mask)

        # Возвращаем PIL Image
        return Image.fromarray(final_np)