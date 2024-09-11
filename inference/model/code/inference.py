import base64
import io
import json

import torch
from diffusers import (StableDiffusionXLImg2ImgPipeline,
                       StableDiffusionXLInpaintPipeline,
                       StableDiffusionXLPipeline)
from PIL import Image
from pytorch_lightning import seed_everything


def image_to_base64(image: Image.Image) -> str:
    """Convert a PIL Image to a base64 string"""
    print("image_to_base64")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def base64_to_image(base64_image: str) -> Image.Image:
    """Convert a base64 string to a PIL Image"""
    print("base64_to_image")
    return Image.open(io.BytesIO(base64.decodebytes(bytes(base64_image, "utf-8"))))


def model_fn(model_dir, context=None):
    print("model_fn")

    base_pipeline = StableDiffusionXLPipeline.from_single_file(
        f"{model_dir}/sd_xl_base_1.0.safetensors",
        torch_dtype=torch.float16,
        use_safetensors=True,
    )
    base_pipeline.load_lora_weights(
        f"{model_dir}/pytorch_lora_weights.safetensors",
        use_safetensors=True,
    )

    img2img_pipeline = StableDiffusionXLImg2ImgPipeline.from_pipe(
        base_pipeline)
    img2img_pipeline.load_lora_weights(
        f"{model_dir}/pytorch_lora_weights.safetensors",
        use_safetensors=True,
    )

    inpainting_pipeline = StableDiffusionXLInpaintPipeline.from_pipe(
        base_pipeline)
    inpainting_pipeline.load_lora_weights(
        f"{model_dir}/pytorch_lora_weights.safetensors",
        use_safetensors=True,
    )

    # refiner_pipeline = StableDiffusionXLImg2ImgPipeline.from_single_file(
    #     f"{model_dir}/sd_xl_refiner_1.0.safetensors",
    #     torch_dtype=torch.float16,
    #     use_safetensors=True,
    # )
    # refiner_pipeline.enable_model_cpu_offload()

    return {"base": base_pipeline, "img2img": img2img_pipeline, "inpaint": inpainting_pipeline}


def input_fn(request_body, request_content_type):
    print("input_fn")
    model_input = json.loads(request_body)
    return model_input


def predict_fn(data, model, context=None):
    print(predict_fn)

    init_image = None
    mask_image = None
    seed = 0
    steps = 50
    height = 1024
    width = 1024
    cfg_scale = 7.0
    image_strength = 0.35

    prompt = data["text_prompts"][0]["text"]

    if "height" in data:
        height = data["height"]
    if "width" in data:
        width = data["width"]
    if "cfg_scale" in data:
        cfg_scale = data["cfg_scale"]
    if "steps" in data:
        steps = data["steps"]
    if "seed" in data:
        seed = data["seed"]
        seed_everything(seed)
    if "init_image" in data:
        init_image = base64_to_image(data["init_image"])
    if "mask_image" in data:
        mask_image = base64_to_image(data["mask_image"])
    if "image_strength" in data:
        image_strength = data["image_strength"]

    if init_image is None:
        print("text2img")

        pipe = model["base"].to("cuda")

        output = pipe(
            prompt=prompt,
            height=height,
            width=width,
            num_inference_steps=steps,
            guidance_scale=cfg_scale,
        ).images[0]

        del pipe

    elif mask_image is None:
        print("img2img")

        pipe = model["img2img"].to("cuda")

        output = pipe(
            prompt,
            image=init_image,
            strength=image_strength,
            num_inference_steps=steps,
            guidance_scale=cfg_scale,
        ).images[0]

        del pipe

    else:
        print("inpaint")

        pipe = model["inpaint"].to("cuda")

        output = pipe(
            prompt=prompt,
            image=init_image,
            mask_image=mask_image,
            height=height,
            width=width,
            strength=image_strength,
            num_inference_steps=steps,
            guidance_scale=cfg_scale,
        ).images[0]

        del pipe

    torch.cuda.empty_cache()

    return output


def output_fn(prediction, accept):
    print("output_fn")

    return json.dumps({"artifacts": image_to_base64(prediction)})
