"""
Qwen-image-edit image-to-image generation example.
This example demonstrates how to use LightX2V with Qwen-Image-Edit model for I2I generation.
"""

import os
import sys
import glob

def _setup_nvidia_libs():
    """Ensure nvidia libraries are in LD_LIBRARY_PATH"""
    if os.environ.get("NVIDIA_LIBS_SETUP") == "1":
        return

    nvidia_libs = []
    for path in sys.path:
        nvidia_path = os.path.join(path, "nvidia")
        if os.path.isdir(nvidia_path):
            libs = glob.glob(os.path.join(nvidia_path, "*", "lib"))
            nvidia_libs.extend(libs)
    
    if not nvidia_libs:
        return

    current_ld_path = os.environ.get("LD_LIBRARY_PATH", "")
    new_paths = ":".join(nvidia_libs)
    
    if current_ld_path:
        # Check if paths are already present to avoid redundant restarts
        if all(lib in current_ld_path for lib in nvidia_libs):
            return
        new_ld_path = f"{current_ld_path}:{new_paths}"
    else:
        new_ld_path = new_paths
        
    os.environ["LD_LIBRARY_PATH"] = new_ld_path
    os.environ["NVIDIA_LIBS_SETUP"] = "1"
    
    print(f"Restarting script with updated LD_LIBRARY_PATH for NVIDIA libraries: {len(nvidia_libs)} paths added.")
    try:
        os.execv(sys.executable, [sys.executable] + sys.argv)
    except Exception as e:
        print(f"Failed to restart script: {e}")

_setup_nvidia_libs()

from lightx2v import LightX2VPipeline

# Initialize pipeline for Qwen-image-edit I2I task
# For Qwen-Image-Edit-2509, use model_cls="qwen-image-edit-2509"
pipe = LightX2VPipeline(
    model_path="/works/checkpoints/Qwen-Image-Edit-2511",
    model_cls="qwen-image-edit-2511",
    task="i2i",
)

# Alternative: create generator from config JSON file
# pipe.create_generator(
#     config_json="../configs/qwen_image/qwen_image_i2i_2511_lora.json"
# )

# Enable offloading to significantly reduce VRAM usage with minimal speed impact
# Suitable for RTX 30/40/50 consumer GPUs
pipe.enable_offload(
    cpu_offload=True,
    offload_granularity="block", #["block", "phase"]
    text_encoder_offload=True,
    vae_offload=False,
)

# Load distilled LoRA weights
pipe.enable_lora(
    [
        {"path": "/works/checkpoints/Qwen-Image-Lightning/Qwen-Image-Edit-2511-Lightning-8steps-V1.0-fp32.safetensors", "strength": 1.0},
    ],
    lora_dynamic_apply=False,  # Support inference with LoRA weights, save memory but slower, default is False
)
# Create generator manually with specified parameters
pipe.create_generator(
    attn_mode="flash_attn2",
    resize_mode="adaptive",
    infer_steps=8,
    guidance_scale=1,
    aspect_ratio=None,  # 원본 비율 유지
)

# Generation parameters
seed = 42
prompt = "Make an ID Portrait of a man. Keep Original Face, eye, nose, lips, hair. Put on a navy blue shirt and a red tie."
negative_prompt = "black heads in skin,black dots in skin."
image_path = "/works/LightX2V/assets/inputs/imgs/kmjeon_260127.png"  # or "/path/to/img_0.jpg,/path/to/img_1.jpg"
save_result_path = "output_kmjeon.png"

# Generate video
pipe.generate(
    seed=seed,
    image_path=image_path,
    prompt=prompt,
    negative_prompt=negative_prompt,
    save_result_path=save_result_path,
)
