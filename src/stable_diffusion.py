import torch
from torch import autocast
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline
from PIL import Image

pipe = StableDiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4",
    torch_dtype=torch.float16,
    custom_pipeline="stable_diffusion_mega",
    revision="fp16",
    use_auth_token=True
)
pipe.to("cuda")
pipe.enable_attention_slicing()
# Disable safety checker to avoid (many) false positives
pipe.safety_checker = lambda images, clip_input: (images, False)


def generate_image(prompt: str):
    with autocast("cuda"):
        image = pipe.text2img(prompt)["sample"][0]
    return image


def expand_image(image, prompt: str):
    with autocast("cuda"):
        resized_image = resize_image(512, image)
        generated_img = pipe.img2img(prompt, init_image=resized_image, strength=0.1)["sample"][0]
    return generated_img


def resize_image(dim, img):
    img = Image.open(img)
    resized = img.resize((dim, dim), Image.LANCZOS)
    alpha_removed = resized.convert("RGB")
    return alpha_removed
