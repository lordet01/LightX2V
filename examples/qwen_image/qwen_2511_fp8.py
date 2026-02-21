"""
Qwen-image-edit image-to-image generation example.
This example demonstrates how to use LightX2V with Qwen-Image-Edit model for I2I generation.
"""

import os
import sys
import glob
from pathlib import Path

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
from landmark_overlay_idphoto import (
    StressTestOptions,
    build_idphoto_landmark_overlay,
)

# Initialize pipeline for Qwen-image-edit I2I task
# For Qwen-Image-Edit-2509, use model_cls="qwen-image-edit-2509"
pipe = LightX2VPipeline(
    model_path="/works/checkpoints/Qwen-Image-Edit-2511",
    model_cls="qwen-image-edit-2511",
    task="i2i",
)

# Alternative: create generator from config JSON file
# pipe.create_generator(
#     config_json="../configs/qwen_image/qwen_image_i2i_2511_distill_fp8.json"
# )

# Enable offloading to significantly reduce VRAM usage with minimal speed impact
# Suitable for RTX 30/40/50 consumer GPUs
pipe.enable_offload(
    cpu_offload=True,
    offload_granularity="block", #["block", "phase"]
    text_encoder_offload=True,
    vae_offload=False,
)

# Load fp8 distilled weights (and int4 Qwen2_5 vl model (optional))
pipe.enable_quantize(
    dit_quantized=True,
    dit_quantized_ckpt="/works/checkpoints/Qwen-Image-Lightning/qwen_image_edit_2511_fp8_e4m3fn_scaled_lightning_8steps_v1.0.safetensors",
    quant_scheme="fp8-sgl",
    #text_encoder_quantized=True,
    #text_encoder_quantized_ckpt="/works/checkpoints/lightx2v/Encoders/Qwen25-VL-4bit-GPTQ",
    #text_encoder_quant_scheme="int4"
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
# image1=원본, image2=랜드마크 오버레이, image3=얼굴 외접(face_org) → image3 얼굴·악세서리 무조건 유지
prompt = (
    "Generate the image following the angle and composition of image 2 in its landmarks. "
    "You must preserve the face and accessories from image 3 exactly. "
    "Change background to gray gradient."
    "Change looks to black suits with white shirt and Gucci pattern tie."
    "Eyes open and looking straight forward."
    "Change hair to Brown Dandy Cut."
)
negative_prompt = ""
# image1 = 입력 원본, image2 = 수정된 랜드마크 오버레이, image3 = face_org
#image1_path = "/works/z-profile/assets/male/kmjeon_tilt.jpg"
image1_path = "/works/z-profile/assets/female/jennie.png"
cases = [
    {
        "tag": "raw_detected",
        "use_raw_detected_points": True,
        "stress_options": StressTestOptions(
            enabled=False,
            force_pose_face_keypoint_match=True,
        )
    },
    {
        "tag": "frontal3d",
        "use_raw_detected_points": False,
        "stress_options": StressTestOptions(
            enabled=False,
            force_pose_face_keypoint_match=True,
            auto_to_frontal=True,
            use_torso_plane_facing=True,
        )
    }
]

for case in cases:
    overlay_tag = case["tag"]
    use_raw = case["use_raw_detected_points"]
    stress_options = case["stress_options"]

    overlay_output_path = str(
        Path(__file__).resolve().parent / "outputs" / f"{Path(image1_path).stem}_landmark_overlay_{overlay_tag}.png"
    )
    overlay_info = build_idphoto_landmark_overlay(
        input_image_path=image1_path,
        overlay_output_path=overlay_output_path,
        stress_options=stress_options,
        use_raw_detected_points=use_raw,
    )

    print("\n=== CASE:", overlay_tag, "===")
    print("[LandmarkOverlay] stress_options:", overlay_info["stress_options"])
    print("[LandmarkOverlay] overlay_output_path:", overlay_info["overlay_output_path"])
    print("[LandmarkOverlay] face_count:", overlay_info["face_count"])
    print("[LandmarkOverlay] pose_count:", overlay_info["pose_count"])
    print("[LandmarkOverlay] pose_allowed_count:", overlay_info["pose_allowed_count"])
    print("[LandmarkOverlay] face_org_path:", overlay_info.get("face_org_path"))

    # image1=원본, image2=오버레이, image3=face_org(얼굴 외접)
    face_org_path = overlay_info.get("face_org_path", "")
    if face_org_path:
        image_path = f"{image1_path},{overlay_output_path},{face_org_path}"
    else:
        image_path = f"{image1_path},{overlay_output_path}"
    save_result_path = f"output_fp8_s8_{Path(image1_path).stem}_idphoto_overlay_{overlay_tag}.png"

    pipe.generate(
        seed=seed,
        image_path=image_path,
        prompt=prompt,
        negative_prompt=negative_prompt,
        save_result_path=save_result_path,
    )
    print("[Generate] saved:", save_result_path)
