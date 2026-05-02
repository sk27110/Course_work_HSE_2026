import random
import numpy as np
import cv2
import torch
from PIL import Image
from typing import Optional

from ultralytics import YOLO
from segment_anything import sam_model_registry, SamPredictor
from diffusers import StableDiffusionPipeline
from transformers import AutoTokenizer, AutoModelForCausalLM

from ..core.base import BaseAugmentationMethod
from ..core.registry import register_augmentation


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =========================
# PROMPT GENERATOR
# =========================
class AGAPromptGeneratorPhi3:
    def __init__(self, device=DEVICE):
        self.device = device

        self.tokenizer = AutoTokenizer.from_pretrained(
            "microsoft/Phi-3-mini-4k-instruct"
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            "microsoft/Phi-3-mini-4k-instruct",
            device_map="auto",
            torch_dtype=torch.float16 if device == "cuda" else torch.float32
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model.config.pad_token_id = self.tokenizer.pad_token_id

        self.backgrounds = [
            "a dense forest", "a desert", "a snowy mountain",
            "a tropical beach", "a city street", "a misty valley"
        ]

        self.temporal = [
            "at sunrise", "at sunset", "at night",
            "during fog", "in winter", "in summer"
        ]

    def generate_prompt(self, class_name):
        base = f"{random.choice(self.backgrounds)} {random.choice(self.temporal)}"

        messages = [
            {"role": "system", "content": "Generate short image prompts."},
            {"role": "user", "content": f"""
Generate ONE short background description.

Rules:
- Do NOT mention: {class_name}
- Only environment
- One sentence
- 10-15 words
- Realistic

Scene: {base}
"""}
        ]

        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=30,
                temperature=0.4,
                top_p=0.9
            )

        text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        text = text.split("\n")[-1].strip()

        return text + ", realistic, cinematic lighting, 4k"


# =========================
# AUGMENTOR
# =========================
@register_augmentation("aga")
class AGAAugmentor(BaseAugmentationMethod):
    """
    AGA: Object extraction + generative background + compositing.
    """

    def __init__(self, class_name: str):
        self.class_name = class_name

        # === Models ===
        self.yolo = YOLO("yolov8n.pt")

        sam = sam_model_registry["vit_b"](checkpoint="sam_vit_b_01ec64.pth")
        sam.to(device=DEVICE)
        self.predictor = SamPredictor(sam)

        self.pipe = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
            safety_checker=None
        ).to(DEVICE)

        self.pipe.enable_model_cpu_offload()

        self.prompt_gen = AGAPromptGeneratorPhi3()

    def prepare(self, dataset):
        """No precomputation needed."""
        pass

    # -------------------------
    # Core steps
    # -------------------------
    def detect_and_mask(self, image_np):
        results = self.yolo(image_np)

        target_id = None
        for k, v in self.yolo.names.items():
            if v.lower() == self.class_name.lower():
                target_id = k

        if target_id is None:
            return None, None

        for box in results[0].boxes:
            if int(box.cls[0]) == target_id:
                bbox = box.xyxy[0].cpu().numpy()
                break
        else:
            return None, None

        x1, y1, x2, y2 = bbox.astype(int)

        self.predictor.set_image(image_np)
        masks, _, _ = self.predictor.predict(
            box=np.array([x1, y1, x2, y2]),
            multimask_output=False
        )

        return image_np, masks[0].astype(np.uint8)

    def extract_object(self, image, mask):
        obj = image.copy()
        obj[mask == 0] = 0
        return obj

    def transform_object(self, obj, mask):
        h, w = obj.shape[:2]

        angle = random.uniform(-25, 25)
        scale = random.uniform(0.6, 1.2)

        M = cv2.getRotationMatrix2D((w//2, h//2), angle, scale)

        obj_t = cv2.warpAffine(obj, M, (w, h))
        mask_t = cv2.warpAffine(mask * 255, M, (w, h))
        mask_t = (mask_t > 0).astype(np.uint8)

        return obj_t, mask_t

    def generate_background(self, image, prompt):
        h, w = image.shape[:2]

        result = self.pipe(
            prompt=prompt,
            height=(h // 8) * 8,
            width=(w // 8) * 8,
            num_inference_steps=20,
            guidance_scale=7.5
        ).images[0]

        return np.array(result.resize((w, h)))

    def place_object(self, background, obj, mask):
        h, w = background.shape[:2]

        ys, xs = np.where(mask > 0)

        if len(ys) == 0:
            return background

        obj_crop = obj[min(ys):max(ys), min(xs):max(xs)]
        mask_crop = mask[min(ys):max(ys), min(xs):max(xs)]

        oh, ow = obj_crop.shape[:2]

        if oh >= h or ow >= w:
            return background

        x = random.randint(0, w - ow)
        y = random.randint(0, h - oh)

        roi = background[y:y+oh, x:x+ow]

        mask_f = cv2.GaussianBlur(mask_crop.astype(float), (15, 15), 0)
        mask_f = np.expand_dims(mask_f, axis=2)

        blended = obj_crop * mask_f + roi * (1 - mask_f)
        background[y:y+oh, x:x+ow] = blended.astype(np.uint8)

        return background

    # -------------------------
    # MAIN API
    # -------------------------
    def augment(self, image: Image.Image) -> Optional[Image.Image]:
        image_np = np.array(image)

        image_np, mask = self.detect_and_mask(image_np)

        if mask is None:
            return None  # важно для пайплайна

        prompt = self.prompt_gen.generate_prompt(self.class_name)

        background = self.generate_background(image_np, prompt)

        obj = self.extract_object(image_np, mask)
        obj_t, mask_t = self.transform_object(obj, mask)

        final = self.place_object(background, obj_t, mask_t)

        return Image.fromarray(final)