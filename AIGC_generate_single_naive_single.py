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
import json


def get_args():
    """
    Parse command-line arguments for the script.

    Returns:
    - opt (Namespace): Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed",
                        type=int,
                        default=2023,
                        help="The random seed.")
    parser.add_argument("--generated_results",
                        type=str,
                        default="/home/shenhaoyu/dataset/generated",
                        help="The directory for saving generated results.")
    parser.add_argument("--dataset",
                        type=str,
                        default="CIFAR100",
                        help="Dataset name for synthetic data generation.")
    parser.add_argument("--gpu_id",
                        type=int,
                        default=2,
                        help="The GPU id for training.")
    parser.add_argument("--scale",
                        type=int,
                        default=600,
                        help="The scale factor (number of photos of each class).")
    parser.add_argument("--batch_size",
                        type=int,
                        default=3,
                        help="How many images to generate in one batch.")
    parser.add_argument("--guidance_scale",
                        type=int,
                        default=7.5,
                        help="Guidance scale for Stable Diffusion.")

    opt = parser.parse_args()
    return opt


def generate(opt):
    """
    Generate synthetic images using a diffusion model.

    Parameters:
    - opt (Namespace): Command-line arguments.
    """
    seed = opt.seed
    gpu_id = opt.gpu_id
    scale = opt.scale
    batch_size = opt.batch_size
    guidance_scale = opt.guidance_scale

    # Ensure the scale is divisible by the batch size
    assert scale % batch_size == 0

    # Set random seed for reproducibility
    seed_everything(seed=seed)

    # Create the directory for saving results
    saving_folder = opt.generated_results
    os.makedirs(saving_folder, exist_ok=True)

    # Model ID for the diffusion pipeline
    model_id = "/home/shenhaoyu/dataset/stabilityai/stable-diffusion-xl-base-1.0"
    torch_device = "cuda:" + str(gpu_id)

    # Load the base and refiner diffusion pipelines
    base = DiffusionPipeline.from_pretrained(
        "/home/shenhaoyu/dataset/stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True
    ).to(torch_device)

    refiner = DiffusionPipeline.from_pretrained(
        "/home/shenhaoyu/dataset/stabilityai/stable-diffusion-xl-refiner-1.0",
        text_encoder_2=base.text_encoder_2,
        vae=base.vae,
        torch_dtype=torch.float16,
        use_safetensors=True,
        variant="fp16"
    ).to(torch_device)

    # Compile the U-Nets for performance optimization

    # Ahead-of-time compilation is a process where the model's operations are analyzed and optimized
    # before the model is run. This is in contrast to just-in-time (JIT) compilation, where the
    # optimizations happen during execution. AOT compilation can lead to better performance because
    # the optimizations are done once, and the optimized code can then be reused for multiple runs.
    torch.compile(base.unet, mode="reduce-overhead", fullgraph=True)
    torch.compile(refiner.unet, mode="reduce-overhead", fullgraph=True)

    # Set up the random generator for the given GPU
    # Generators are used to control the randomness in PyTorch, such as when generating random numbers or shuffling data.
    generator = torch.Generator(device="cuda:" + str(gpu_id)).manual_seed(seed)

    # Directory for saving generated results
    result_dir = os.path.join(opt.generated_results, opt.dataset)
    os.makedirs(result_dir, exist_ok=True)

    # Fixed format strings for image prompts
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
        "a photo of a small {}"
    ]

    # Set the random seed for reproducibility
    random.seed(opt.seed)

    # Get the class names based on the dataset
    if opt.dataset == "CIFAR10":
        names = ["airplane", "automobile", "bird", "cat",
                 "deer", "dog", "frog", "horse", "ship", "truck"]
    elif opt.dataset == "CIFAR100":
        with open("./dataset/cifar100.json", "r") as f:
            idx_to_label = json.load(f)
            names = [item for key, item in idx_to_label.items()]
            print(names)

    n_cls = len(names)

    # Loop through each class to generate images
    for i in range(n_cls):
        name = names[i]  # Get the name of the current class
        # Create a subdirectory path for the current class
        subdir = os.path.join(result_dir, name)
        # Create the subdirectory if it doesn't exist
        os.makedirs(subdir, exist_ok=True)
        generated_image = 0  # Initialize the counter for generated images

        # Check if there are existing images in the directory
        if len(os.listdir(subdir)) > 0:
            for subimage in os.listdir(subdir):
                # Extract the number from the image file name and update the counter
                num = int(subimage.replace(".jpg", ""))
                generated_image = max(generated_image, num + 1)

        # Generate images in batches
        for idx, batch in enumerate(range((scale - generated_image) // batch_size)):
            # Generate initial images with the base pipeline
            image = base(
                prompt=[random.choice(fixed_format).format(name) for _ in range(
                    batch_size)],  # Create prompts for image generation
                num_inference_steps=15,  # Number of inference steps for the base pipeline
                denoising_end=0.8,  # End denoising after 80% of the steps
                guidance_scale=guidance_scale,  # Scale for guidance
                generator=generator,  # Random number generator for reproducibility
                output_type="latent"  # Output type is latent
            ).images

            # Refine the generated images
            image = refiner(
                prompt=[random.choice(fixed_format).format(name) for _ in range(
                    batch_size)],  # Create prompts for refining the images
                num_inference_steps=15,  # Number of inference steps for the refiner pipeline
                denoising_start=0.8,  # Start denoising from 80% of the steps
                guidance_scale=guidance_scale,  # Scale for guidance
                image=image  # Input the previously generated images for refinement
            ).images

            # Save the generated images
            for j, img in enumerate(image):
                img.save(os.path.join(
                    subdir, f"{idx * batch_size + j + generated_image}.jpg"))  # Save each image with a unique filename

            # Collect garbage and empty GPU cache to manage memory usage
            gc.collect()
            torch.cuda.empty_cache()


if __name__ == "__main__":
    opt = get_args()
    generate(opt)
