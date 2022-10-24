import torch
from torch import autocast
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline
from PIL import Image

pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16, revision="fp16").to("cuda")
pipe.enable_attention_slicing()
# TODO: Replace this hack once change #889 in the repo is in a pip packaged release and components property is released.
sub_models = {k: v for k, v in vars(pipe).items() if not k.startswith("_")}
img_pipe = StableDiffusionImg2ImgPipeline(**sub_models)
# Disable safety checker to avoid (many) false positives
pipe.safety_checker = lambda images, clip_input: (images, False)


def generate_image(prompt: str):
    with autocast("cuda"):
        image = pipe(prompt).images[0]
    return image


def expand_image(image, prompt: str):
    with autocast("cuda"):
        resized_image = resize_image(512, image)
        generated_img = img_pipe(prompt, init_image=resized_image, strength=0.7).images[0]
    return generated_img


def resize_image(dim, img):
    img = Image.open(img)
    resized = img.resize((dim, dim), Image.LANCZOS)
    alpha_removed = resized.convert("RGB")
    return alpha_removed
