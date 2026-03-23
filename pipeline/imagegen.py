import argparse
import os
from PIL import Image
from diffusers import StableDiffusion3Pipeline
import torch
from tqdm import tqdm

# ------------------------------------------------------------------------------------------------ #
#                            Pretrained Image Generator + Stable Fast 3D                           #
# ------------------------------------------------------------------------------------------------ #

def main():
    # Parse arguments base_model, lora weights, output directory
    parser = argparse.ArgumentParser(description="Run SD3 on a given text prompt")
    parser.add_argument(
        "--base_model",
        type=str,
        help="Base model to use",
        default="stabilityai/stable-diffusion-3-medium-diffusers",
    )
    parser.add_argument(
        "--lora_weights",
        type=str,
        help="Lora weights to use",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        help="Prompt to use",
    )
    parser.add_argument(
        "--negative_prompt",
        type=str,
        help="Negative prompt to use",
        default="blurry, low quality, bad lighting",
    )
    parser.add_argument(
        "--num_inference_steps", type=int, help="Number of inference steps", default=30
    )
    parser.add_argument(
        "--guidance_scale", type=float, help="Guidance scale", default=7.5
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Output directory",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        help="Cache directory",
    )
    parser.add_argument("--height", type=int, help="Height of the image", default=512)
    parser.add_argument("--width", type=int, help="Width of the image", default=512)
    args = parser.parse_args()

    pipeline = load_model(args.base_model, args.lora_weights, args.cache_dir)
    # If args.prompt is a txt, load it and save it in a list
    if args.prompt.endswith(".txt"):
        with open(args.prompt, "r") as f:
            prompts = f.readlines()
    else:
        prompts = [args.prompt]

    print(f"Generating {len(prompts)} images in {args.output_dir}.")
    for i, prompt in enumerate(tqdm(prompts, desc="Image Generation")):
        image = generate_image(
            pipeline,
            prompt,
            args.height,
            args.width,
            args.negative_prompt,
            args.num_inference_steps,
            args.guidance_scale,
        )
        save_image(image, args.output_dir, i)


def save_image(image, output_dir, index=0):
    os.makedirs(output_dir, exist_ok=True)
    image.save(os.path.join(output_dir, f"image_{index}.png"))


def load_model(base_model, lora_weights, cache_dir, fp32=False, progress_bar=True):
    pipeline = StableDiffusion3Pipeline.from_pretrained(
        base_model,
        cache_dir=cache_dir,
        torch_dtype=torch.bfloat16 if not fp32 else torch.float32,
        device_map="cuda",
    )
    try:
        print(f"Loading LORA weights from {lora_weights}")
        # load attention processors
        pipeline.load_lora_weights(lora_weights)
    except Exception as e:
        print(f"Loading Base Model Only. Error loading LORA weights: {e}")

    if not progress_bar:
        pipeline.set_progress_bar_config(disable=True)

    return pipeline


def generate_image(
    pipeline,
    prompt,
    negative_prompt,
    width,
    height,
    num_inference_steps,
    guidance_scale,
):
    image = pipeline(
        prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        height=height,
        width=width,
    ).images[0]


    return image


if __name__ == "__main__":
    main()
