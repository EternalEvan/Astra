"""
UniVideo Inference Script

Supports text-to-video (t2v) and image-to-video (i2v) generation using
HunyuanVideo with an optional Qwen2.5-VL MLLM for metaquery-based prompt
enhancement.

When `num_metaqueries: 0` in the YAML config, the MLLM is skipped entirely
and the raw text prompt is used directly with the HunyuanVideo pipeline.

Usage:
    python univideo_inference.py --demo_task t2v --config configs/univideo_qwen2p5vl7b_hidden_hunyuanvideo.yaml
"""

import argparse
import os
import sys

import torch
import yaml


def load_config(config_path: str) -> dict:
    """Load and parse a YAML configuration file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def build_mllm(mllm_config: dict, device: str = "cuda"):
    """
    Build the Qwen2.5-VL MLLM only when num_metaqueries > 0.

    Returns None when num_metaqueries == 0 so that the caller can skip
    MLLM-dependent code paths without any model downloads.
    """
    num_metaqueries = mllm_config.get("num_metaqueries", 0)
    if num_metaqueries == 0:
        return None

    mllm_id = mllm_config["mllm_id"]
    print(f"Using Qwen MLLM {mllm_id}")

    try:
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    except ImportError:
        raise ImportError(
            "Qwen2.5-VL requires transformers >= 4.49.0. "
            "Please upgrade: pip install -U transformers"
        )

    torch_dtype = torch.bfloat16
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        mllm_id,
        torch_dtype=torch_dtype,
        device_map=device,
    )
    if mllm_config.get("_gradient_checkpointing", False):
        model.gradient_checkpointing_enable()
    model.eval()

    processor = AutoProcessor.from_pretrained(mllm_id)
    return {"model": model, "processor": processor}


def encode_prompt_with_mllm(
    mllm,
    prompt: str,
    mllm_config: dict,
    images=None,
    video_frames=None,
):
    """
    Use the Qwen2.5-VL MLLM to generate an enhanced prompt via metaqueries.

    Only called when num_metaqueries > 0 and mllm is not None.
    """
    num_metaqueries = mllm_config.get("num_metaqueries", 0)
    assert num_metaqueries > 0 and mllm is not None

    model = mllm["model"]
    processor = mllm["processor"]
    system_prompt = mllm_config.get("system_prompt", "You are a helpful assistant.")
    use_chat_template = mllm_config.get("use_chat_template", True)
    max_input_text_tokens = mllm_config.get("max_input_text_tokens", 2048)

    content = []
    if images is not None:
        for img in images:
            content.append({"type": "image", "image": img})
    if video_frames is not None:
        content.append({"type": "video", "video": video_frames})
    content.append({"type": "text", "text": prompt})

    messages = []
    if use_chat_template:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": content})

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(
        text=[text],
        padding=True,
        return_tensors="pt",
        max_length=max_input_text_tokens,
        truncation=True,
    ).to(model.device)

    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=512)

    input_len = inputs["input_ids"].shape[1]
    generated = output_ids[:, input_len:]
    enhanced_prompt = processor.batch_decode(
        generated, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )[0].strip()
    return enhanced_prompt


def build_pipeline(pipeline_config: dict, transformer_ckpt_path: str = None, device: str = "cuda"):
    """Build the HunyuanVideo inference pipeline."""
    from diffusers import HunyuanVideoPipeline, HunyuanVideoTransformer3DModel
    from diffusers.schedulers import FlowMatchEulerDiscreteScheduler

    hunyuan_model_id = pipeline_config["hunyuan_model_id"]
    timestep_shift = pipeline_config.get("timestep_shift", 7.0)

    print(f"Loading HunyuanVideo pipeline from {hunyuan_model_id}")

    if transformer_ckpt_path is not None and os.path.isfile(transformer_ckpt_path):
        print(f"Loading transformer weights from {transformer_ckpt_path}")
        transformer = HunyuanVideoTransformer3DModel.from_pretrained(
            hunyuan_model_id,
            subfolder="transformer",
            torch_dtype=torch.bfloat16,
        )
        state_dict = torch.load(transformer_ckpt_path, map_location="cpu", weights_only=True)
        transformer.load_state_dict(state_dict, strict=False)
    else:
        transformer = HunyuanVideoTransformer3DModel.from_pretrained(
            hunyuan_model_id,
            subfolder="transformer",
            torch_dtype=torch.bfloat16,
        )

    pipe = HunyuanVideoPipeline.from_pretrained(
        hunyuan_model_id,
        transformer=transformer,
        torch_dtype=torch.float16,
    ).to(device)

    pipe.scheduler = FlowMatchEulerDiscreteScheduler.from_config(
        pipe.scheduler.config, shift=timestep_shift
    )
    pipe.vae.enable_tiling()

    return pipe


def run_t2v(
    pipe,
    prompt: str,
    output_path: str = "output.mp4",
    height: int = 720,
    width: int = 1280,
    num_frames: int = 61,
    num_inference_steps: int = 50,
    guidance_scale: float = 6.0,
    seed: int = 0,
):
    """Run text-to-video generation."""
    prompt_preview = prompt if len(prompt) <= 80 else f"{prompt[:80]}..."
    print(f"Generating video: prompt='{prompt_preview}' ({height}x{width}, {num_frames} frames)")

    generator = torch.Generator("cpu").manual_seed(seed)
    output = pipe(
        prompt=prompt,
        height=height,
        width=width,
        num_frames=num_frames,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator=generator,
    )
    video_frames = output.frames[0]

    try:
        import imageio
        imageio.mimsave(output_path, video_frames, fps=24, quality=8)
        print(f"Video saved to: {output_path}")
    except ImportError:
        print("Warning: imageio not found. Install with `pip install imageio[ffmpeg]` to save videos.")
        print(f"Generated {len(video_frames)} frames.")

    return video_frames


def run_i2v(
    pipe,
    prompt: str,
    image_path: str,
    output_path: str = "output.mp4",
    height: int = 720,
    width: int = 1280,
    num_frames: int = 61,
    num_inference_steps: int = 50,
    guidance_scale: float = 6.0,
    seed: int = 0,
):
    """Run image-to-video generation."""
    from PIL import Image

    print(f"Generating video from image: {image_path}")
    image = Image.open(image_path).convert("RGB")
    image = image.resize((width, height))

    generator = torch.Generator("cpu").manual_seed(seed)
    output = pipe(
        prompt=prompt,
        image=image,
        height=height,
        width=width,
        num_frames=num_frames,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator=generator,
    )
    video_frames = output.frames[0]

    try:
        import imageio
        imageio.mimsave(output_path, video_frames, fps=24, quality=8)
        print(f"Video saved to: {output_path}")
    except ImportError:
        print("Warning: imageio not found. Install with `pip install imageio[ffmpeg]` to save videos.")

    return video_frames


def main():
    parser = argparse.ArgumentParser(description="UniVideo inference script")
    parser.add_argument(
        "--demo_task",
        type=str,
        required=True,
        choices=["t2v", "i2v"],
        help="Inference task: t2v (text-to-video) or i2v (image-to-video)",
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the YAML configuration file",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="A serene mountain landscape with flowing water and lush green trees.",
        help="Text prompt for video generation",
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Path to input image (required for i2v task)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output.mp4",
        help="Output video file path",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=720,
        help="Output video height in pixels",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1280,
        help="Output video width in pixels",
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=61,
        help="Number of frames to generate",
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=50,
        help="Number of denoising steps",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=6.0,
        help="Classifier-free guidance scale",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run inference on (e.g. cuda, cpu)",
    )
    args = parser.parse_args()

    if args.demo_task == "i2v" and args.image is None:
        parser.error("--image is required for the i2v task")

    config = load_config(args.config)
    mllm_config = config.get("mllm_config", {})
    pipeline_config = config.get("pipeline_config", {})
    transformer_ckpt_path = config.get("transformer_ckpt_path", None)

    # Build MLLM only when num_metaqueries > 0.
    # When num_metaqueries == 0 the MLLM is skipped entirely; no model files
    # are downloaded and the raw text prompt is forwarded to the video pipeline.
    mllm = build_mllm(mllm_config, device=args.device)

    prompt = args.prompt
    if mllm is not None:
        # Optionally provide conditioning images/pixels to the MLLM
        cond_images = None
        cond_video_frames = None
        if pipeline_config.get("mllm_use_ref_img") and args.image is not None:
            from PIL import Image
            cond_images = [Image.open(args.image).convert("RGB")]
        prompt = encode_prompt_with_mllm(
            mllm,
            prompt,
            mllm_config,
            images=cond_images,
            video_frames=cond_video_frames,
        )
        prompt_preview = prompt if len(prompt) <= 120 else f"{prompt[:120]}..."
        print(f"MLLM enhanced prompt: {prompt_preview}")

    pipe = build_pipeline(pipeline_config, transformer_ckpt_path, device=args.device)

    if args.demo_task == "t2v":
        run_t2v(
            pipe,
            prompt=prompt,
            output_path=args.output,
            height=args.height,
            width=args.width,
            num_frames=args.num_frames,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            seed=args.seed,
        )
    elif args.demo_task == "i2v":
        run_i2v(
            pipe,
            prompt=prompt,
            image_path=args.image,
            output_path=args.output,
            height=args.height,
            width=args.width,
            num_frames=args.num_frames,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()
