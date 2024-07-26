import torch
import os
from utils import seed_everything
from diffusers import (
    AutoencoderKL,
    DiffusionPipeline,
    UNet2DConditionModel
)
from transformers import (
    CLIPTokenizer,
    CLIPTextModel
)
import random

import argparse
from PIL import Image
import gc


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed",
                        type=int,
                        default=2023,
                        help="The random seed.")
    parser.add_argument("--generated_results",
                        type=str,
                        default="/home/shenhaoyu/dataset/generated",
                        help="The result for generation.")

    parser.add_argument("--dataset",
                        type=str,
                        default="CIFAR100",
                        help="Dataset name for synthetic data generation.")

    parser.add_argument("--gpu_id",
                        type=int,
                        default=2,
                        help="The gpu id for training.")

    parser.add_argument("--scale",
                        type=int,
                        default=600,
                        help="The scale factor(# photos of each class).")

    parser.add_argument("--batch_size",
                        type=int,
                        default=3,
                        help="How many images to generate in one shot.")

    parser.add_argument("--guidance_scale",
                        type=int,
                        default=7.5,
                        help="Guidance scale for Stable Diffusion.")
    opt = parser.parse_args()
    return opt


def generate(opt):
    seed = opt.seed
    gpu_id = opt.gpu_id
    scale = opt.scale
    batch_size = opt.batch_size
    guidance_scale = opt.guidance_scale

    assert scale % batch_size == 0
    seed_everything(seed=seed)
    saving_folder = opt.generated_results
    os.makedirs(saving_folder, exist_ok=True)
    model_id = "stabilityai/stable-diffusion-xl-base-1.0"
    torch_device = "cuda:" + str(gpu_id)

    base = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
    ).to(torch_device)

    refiner = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-refiner-1.0",
        text_encoder_2=base.text_encoder_2,
        vae=base.vae,
        torch_dtype=torch.float16,
        use_safetensors=True,
        variant="fp16",).to(torch_device)

    #
    torch.compile(base.unet, mode="reduce-overhead", fullgraph=True)
    torch.compile(refiner.unet, mode="reduce-overhead", fullgraph=True)

    generator = torch.Generator(device="cuda:" + str(gpu_id)).manual_seed(seed)
    # use StableDiffusionXLPipeline for generation

    result_dir = opt.generated_results + opt.dataset
    os.makedirs(result_dir, exist_ok=True)
    # https://huggingface.co/docs/diffusers/en/using-diffusers/reusing_seeds

    fixed_format = [
        "a photo of a {}",
        "a rendering of a {}",
        "a cropped photo of the {}",
        "the photo of a {}",
        "a photo of a clean {}",
        "a photo of a dirty {}",
        "a dark photo of the {}",
        "a photo of my {}",
        "a photo of the cool {}",
        "a close-up photo of a {}",
        "a bright photo of the {}",
        "a cropped photo of a {}",
        "a photo of the {}",
        "a good photo of the {}",
        "a photo of one {}",
        "a close-up photo of the {}",
        "a rendition of the {}",
        "a photo of the clean {}",
        "a rendition of a {}",
        "a photo of a nice {}",
        "a good photo of a {}",
        "a photo of the nice {}",
        "a photo of the small {}",
        "a photo of the weird {}",
        "a photo of the large {}",
        "a photo of a cool {}",
        "a photo of a small {}",
    ]

    import random
    random.seed(opt.seed)
    # https://www.cs.toronto.edu/~kriz/cifar.html
    if opt.dataset == "CIFAR10":
        names = ["airplane", "automobile", "bird", "cat",
                 "deer", "dog", "frog", "horse", "ship", "truck"]
    elif opt.dataset == "CIFAR100":
        import json
        with open("./dataset/cifar100.json", "r+") as f:
            idx_to_label = json.load(f)
            names = [item for key, item in idx_to_label.items()]
            print(names)
    n_cls = len(names)
    for i in range(n_cls):
        name = names[i]
        subdir = os.path.join(result_dir, name)
        os.makedirs(subdir, exist_ok=True)
        generated_image = 0
        if len(os.listdir(subdir)) > 0:
            for subimage in os.listdir(subdir):
                num = int(subimage.replace(".jpg", ""))
                generated_image = max(generated_image, num + 1)
        for idx, batch in enumerate(range((scale - generated_image) // batch_size)):
            image = base(prompt=[random.choice(fixed_format).format(name) for _ in range(batch_size)],
                         num_inference_steps=15,
                         denoising_end=0.8,
                         guidance_scale=guidance_scale,
                         generator=generator,
                         output_type="latent").images
            image = refiner(prompt=[random.choice(fixed_format).format(name) for _ in range(batch_size)],
                            num_inference_steps=15,
                            denoising_start=0.8,
                            guidance_scale=guidance_scale,
                            image=image
                            ).images
            for j, img in enumerate(image):
                img.save(os.path.join(
                    subdir, f"{idx * batch_size + j + generated_image}.jpg"))
            gc.collect()
            torch.cuda.empty_cache()


if __name__ == "__main__":
    opt = get_args()
    generate(opt)
