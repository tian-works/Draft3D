"""
Workflow construction helpers for ComfyUI.

These functions build the JSON graph structures used by ComfyUI. They are
extracted from the original `GUI.py` so they can be reused independently
from the GUI layer.
"""

from __future__ import annotations

import time as _time
from typing import Any, Dict, Optional


def build_workflow_with_controlnet(
    prompt: str,
    seed: int = 0,
    steps: int = 9,
    cfg: float = 1.0,
    batch_size: int = 1,
    sketch_filename: Optional[str] = None,
    sketch_subfolder: Optional[str] = None,
    control_strength: float = 0.85,
    canny_low: float = 0.1,
    canny_high: float = 0.32,
) -> Dict[str, Any]:
    """
    Build a workflow that uses ControlNet and Canny edge detection
    (for sketch + prompt generation), based on a user‑provided
    z-image-turbo workflow (optimized version).
    """
    # If seed is -1, use a random seed
    if seed == -1:
        import random

        seed = random.randint(0, 2**32 - 1)

    workflow: Dict[str, Any] = {
        "9": {
            "inputs": {
                "filename_prefix": "ZImageTurbo",
                "images": ["70:43", 0],
            },
            "class_type": "SaveImage",
        },
        "56": {
            "inputs": {
                "images": ["57", 0],
            },
            "class_type": "PreviewImage",
        },
        "57": {
            "inputs": {
                "low_threshold": canny_low,
                "high_threshold": canny_high,
                "image": ["58", 0],
            },
            "class_type": "Canny",
        },
        "58": {
            "inputs": {
                "image": sketch_filename if sketch_filename else "placeholder.png",
            },
            "class_type": "LoadImage",
        },
        "70:39": {
            "inputs": {
                "clip_name": "qwen_3_4b.safetensors",
                "type": "lumina2",
                "device": "default",
            },
            "class_type": "CLIPLoader",
        },
        "70:46": {
            "inputs": {
                "unet_name": "z_image_turbo_bf16.safetensors",
                "weight_dtype": "default",
            },
            "class_type": "UNETLoader",
        },
        "70:40": {
            "inputs": {
                "vae_name": "ae.safetensors",
            },
            "class_type": "VAELoader",
        },
        "70:64": {
            "inputs": {
                "name": "Z-Image-Turbo-Fun-Controlnet-Union.safetensors",
            },
            "class_type": "ModelPatchLoader",
        },
        "70:43": {
            "inputs": {
                "samples": ["70:44", 0],
                "vae": ["70:40", 0],
            },
            "class_type": "VAEDecode",
        },
        "70:47": {
            "inputs": {
                "shift": 3,
                "model": ["70:60", 0],
            },
            "class_type": "ModelSamplingAuraFlow",
        },
        "70:44": {
            "inputs": {
                "seed": seed,
                "steps": steps,
                "cfg": cfg,
                "sampler_name": "res_multistep",
                "scheduler": "simple",
                "denoise": 1.0,
                "model": ["70:47", 0],
                "positive": ["70:45", 0],
                "negative": ["70:42", 0],
                "latent_image": ["70:41", 0],
            },
            "class_type": "KSampler",
        },
        "70:42": {
            "inputs": {
                "conditioning": ["70:45", 0],
            },
            "class_type": "ConditioningZeroOut",
        },
        "70:60": {
            "inputs": {
                "strength": control_strength,
                "model": ["70:46", 0],
                "model_patch": ["70:64", 0],
                "vae": ["70:40", 0],
                "image": ["57", 0],
            },
            "class_type": "QwenImageDiffsynthControlnet",
        },
        "70:41": {
            "inputs": {
                "width": ["70:69", 0],
                "height": ["70:69", 1],
                "batch_size": batch_size,
            },
            "class_type": "EmptySD3LatentImage",
        },
        "70:45": {
            "inputs": {
                "text": prompt,
                "clip": ["70:39", 0],
            },
            "class_type": "CLIPTextEncode",
        },
        "70:69": {
            "inputs": {
                "image": ["57", 0],
            },
            "class_type": "GetImageSize",
        },
    }

    # If a subfolder is provided, attach it to the LoadImage node
    if sketch_subfolder:
        workflow["58"]["inputs"]["subfolder"] = sketch_subfolder

    return workflow


def build_workflow(
    prompt: str,
    seed: int = 0,
    steps: int = 4,
    cfg: float = 1.0,
    width: int = 512,
    height: int = 512,
    batch_size: int = 1,
    use_sketch: bool = False,
    sketch_filename: Optional[str] = None,
    sketch_subfolder: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Build the base ComfyUI workflow.

    Note: this is based on a user‑provided workflow, with default size
    changed to 512×512.
    """
    workflow: Dict[str, Any] = {
        "9": {
            "inputs": {"filename_prefix": "z-image", "images": ["57:8", 0]},
            "class_type": "SaveImage",
        },
        "58": {
            "inputs": {"value": prompt},
            "class_type": "PrimitiveStringMultiline",
        },
        "61": {
            "inputs": {
                "string_a": "Pixel art style,",
                "string_b": ["58", 0],
                "delimiter": "",
            },
            "class_type": "StringConcatenate",
        },
        "57:30": {
            "inputs": {
                "clip_name": "qwen_3_4b.safetensors",
                "type": "lumina2",
                "device": "default",
            },
            "class_type": "CLIPLoader",
        },
        "57:29": {
            "inputs": {"vae_name": "ae.safetensors"},
            "class_type": "VAELoader",
        },
        "57:33": {
            "inputs": {"conditioning": ["57:27", 0]},
            "class_type": "ConditioningZeroOut",
        },
        "57:8": {
            "inputs": {"samples": ["57:3", 0], "vae": ["57:29", 0]},
            "class_type": "VAEDecode",
        },
        "57:28": {
            "inputs": {
                "unet_name": "z_image_turbo_bf16.safetensors",
                "weight_dtype": "default",
            },
            "class_type": "UNETLoader",
        },
        "57:27": {
            "inputs": {"text": ["58", 0], "clip": ["57:30", 0]},
            "class_type": "CLIPTextEncode",
        },
        "57:13": {
            "inputs": {"width": width, "height": height, "batch_size": batch_size},
            "class_type": "EmptySD3LatentImage",
        },
        "57:3": {
            "inputs": {
                "seed": seed,
                "steps": steps,
                "cfg": cfg,
                "sampler_name": "res_multistep",
                "scheduler": "simple",
                "denoise": 1,
                "model": ["57:11", 0],
                "positive": ["57:27", 0],
                "negative": ["57:33", 0],
                "latent_image": ["57:13", 0]
                if not use_sketch
                else ["57:100", 0],  # Use encoded sketch latent if available
            },
            "class_type": "KSampler",
        },
        "57:11": {
            "inputs": {"shift": 3, "model": ["57:28", 0]},
            "class_type": "ModelSamplingAuraFlow",
        },
    }

    # If using a sketch, add LoadImage and VAEEncode nodes
    if use_sketch and sketch_filename:
        load_image_inputs: Dict[str, Any] = {
            "image": sketch_filename,
        }

        if sketch_subfolder:
            load_image_inputs["subfolder"] = sketch_subfolder

        workflow["57:99"] = {
            "inputs": load_image_inputs,
            "class_type": "LoadImage",
        }

        workflow["57:100"] = {
            "inputs": {
                "pixels": ["57:99", 0],
                "vae": ["57:29", 0],
            },
            "class_type": "VAEEncode",
        }

    return workflow


def build_workflow_z_image_turbo_edit(
    prompt: str,
    image_filename: str,
    image_subfolder: Optional[str] = None,
    seed: int = -1,
    steps: int = 9,
    cfg: float = 1.0,
    control_strength: float = 0.85,
    canny_low: float = 0.1,
    canny_high: float = 0.32,
    batch_size: int = 1,
) -> Dict[str, Any]:
    """
    Build a Z-Image-Turbo ControlNet workflow for image editing.
    """
    # Validate prompt
    if not prompt or not prompt.strip():
        raise ValueError(
            "Prompt must not be empty; a text prompt is required to control image editing."
        )

    prompt = prompt.strip()
    print(f"[Workflow] Using prompt: {prompt}")

    # If seed is -1, use a time-based seed
    if seed == -1:
        seed = int(_time.time() * 1000000) % (2**32)

    workflow: Dict[str, Any] = {
        "9": {
            "inputs": {
                "filename_prefix": "ZImageTurbo",
                "images": ["70:43", 0],
            },
            "class_type": "SaveImage",
        },
        "56": {
            "inputs": {
                "images": ["57", 0],
            },
            "class_type": "PreviewImage",
        },
        "57": {
            "inputs": {
                "low_threshold": canny_low,
                "high_threshold": canny_high,
                "image": ["58", 0],
            },
            "class_type": "Canny",
        },
        "58": {
            "inputs": {
                "image": image_filename,
            },
            "class_type": "LoadImage",
        },
        "70:39": {
            "inputs": {
                "clip_name": "qwen_3_4b.safetensors",
                "type": "lumina2",
                "device": "default",
            },
            "class_type": "CLIPLoader",
        },
        "70:46": {
            "inputs": {
                "unet_name": "z_image_turbo_bf16.safetensors",
                "weight_dtype": "default",
            },
            "class_type": "UNETLoader",
        },
        "70:40": {
            "inputs": {
                "vae_name": "ae.safetensors",
            },
            "class_type": "VAELoader",
        },
        "70:64": {
            "inputs": {
                "name": "Z-Image-Turbo-Fun-Controlnet-Union.safetensors",
            },
            "class_type": "ModelPatchLoader",
        },
        "70:43": {
            "inputs": {
                "samples": ["70:44", 0],
                "vae": ["70:40", 0],
            },
            "class_type": "VAEDecode",
        },
        "70:47": {
            "inputs": {
                "shift": 3,
                "model": ["70:60", 0],
            },
            "class_type": "ModelSamplingAuraFlow",
        },
        "70:44": {
            "inputs": {
                "seed": seed,
                "steps": steps,
                "cfg": cfg,
                "sampler_name": "res_multistep",
                "scheduler": "simple",
                "denoise": 1.0,
                "model": ["70:47", 0],
                "positive": ["70:45", 0],
                "negative": ["70:42", 0],
                "latent_image": ["70:41", 0],
            },
            "class_type": "KSampler",
        },
        "70:42": {
            "inputs": {
                "conditioning": ["70:45", 0],
            },
            "class_type": "ConditioningZeroOut",
        },
        "70:60": {
            "inputs": {
                "strength": control_strength,
                "model": ["70:46", 0],
                "model_patch": ["70:64", 0],
                "vae": ["70:40", 0],
                "image": ["57", 0],
            },
            "class_type": "QwenImageDiffsynthControlnet",
        },
        "70:41": {
            "inputs": {
                "width": ["70:69", 0],
                "height": ["70:69", 1],
                "batch_size": batch_size,
            },
            "class_type": "EmptySD3LatentImage",
        },
        "70:45": {
            "inputs": {
                "text": prompt,
                "clip": ["70:39", 0],
            },
            "class_type": "CLIPTextEncode",
        },
        "70:69": {
            "inputs": {
                "image": ["58", 0],
            },
            "class_type": "GetImageSize",
        },
    }

    if image_subfolder:
        workflow["58"]["inputs"]["subfolder"] = image_subfolder

    return workflow


def build_workflow_img2img(
    prompt: str,
    image_filename: str,
    image_subfolder: Optional[str] = None,
    seed: int = 0,
    steps: int = 4,
    cfg: float = 1.0,
    denoise: float = 0.75,
    width: int = 512,
    height: int = 512,
    batch_size: int = 1,
) -> Dict[str, Any]:
    """
    Build an image‑to‑image (img2img) workflow for editing an existing image.
    """
    workflow: Dict[str, Any] = {
        "9": {
            "inputs": {"filename_prefix": "z-image-edited", "images": ["57:8", 0]},
            "class_type": "SaveImage",
        },
        "58": {
            "inputs": {"value": prompt},
            "class_type": "PrimitiveStringMultiline",
        },
        "61": {
            "inputs": {
                "string_a": "Pixel art style,",
                "string_b": ["58", 0],
                "delimiter": "",
            },
            "class_type": "StringConcatenate",
        },
        "57:30": {
            "inputs": {
                "clip_name": "qwen_3_4b.safetensors",
                "type": "lumina2",
                "device": "default",
            },
            "class_type": "CLIPLoader",
        },
        "57:29": {
            "inputs": {"vae_name": "ae.safetensors"},
            "class_type": "VAELoader",
        },
        "57:33": {
            "inputs": {"conditioning": ["57:27", 0]},
            "class_type": "ConditioningZeroOut",
        },
        "57:8": {
            "inputs": {"samples": ["57:3", 0], "vae": ["57:29", 0]},
            "class_type": "VAEDecode",
        },
        "57:28": {
            "inputs": {
                "text": ["61", 0],
                "clip": ["57:30", 0],
            },
            "class_type": "CLIPTextEncode",
        },
        "57:27": {
            "inputs": {
                "text": "low quality, blurry, distorted, ugly, bad anatomy, bad proportions, extra limbs, "
                "cloned face, disfigured, out of frame, ugly, extra limbs, bad anatomy, gross "
                "proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, "
                "mutated hands, fused fingers, too many fingers, long neck",
                "clip": ["57:30", 0],
            },
            "class_type": "CLIPTextEncode",
        },
        "57:3": {
            "inputs": {
                "seed": seed,
                "steps": steps,
                "cfg": cfg,
                "sampler_name": "euler",
                "scheduler": "simple",
                "denoise": denoise,
                "model": ["57:46", 0],
                "positive": ["57:28", 0],
                "negative": ["57:33", 0],
                "latent_image": ["57:41", 0],
            },
            "class_type": "KSampler",
        },
        "57:46": {
            "inputs": {"model_name": "lumina2.safetensors", "device": "default"},
            "class_type": "CheckpointLoaderSimple",
        },
        "57:41": {
            "inputs": {
                "pixels": ["57:40", 0],
                "vae": ["57:29", 0],
            },
            "class_type": "VAEEncode",
        },
        "57:40": {
            "inputs": {
                "image": image_filename,
            },
            "class_type": "LoadImage",
        },
        "57:39": {
            "inputs": {
                "width": width,
                "height": height,
            },
            "class_type": "EmptySD3LatentImage",
        },
    }

    if image_subfolder:
        workflow["57:40"]["inputs"]["subfolder"] = image_subfolder

    return workflow


def build_workflow_hunyuan3d(
    image_filename: str,
    image_subfolder: Optional[str] = None,
    seed: int = 952805179515179,
    steps: int = 30,
    cfg: float = 5.0,
    resolution: int = 1024,
) -> Dict[str, Any]:
    """
    Build a Hunyuan3D workflow for generating 3D models (.glb files).
    """
    workflow: Dict[str, Any] = {
        "1": {
            "inputs": {"ckpt_name": "hunyuan_3d_v2.1.safetensors"},
            "class_type": "ImageOnlyCheckpointLoader",
        },
        "2": {
            "inputs": {"image": image_filename},
            "class_type": "LoadImage",
        },
        "3": {
            "inputs": {"shift": 1.0, "model": ["1", 0]},
            "class_type": "ModelSamplingAuraFlow",
        },
        "4": {
            "inputs": {"resolution": resolution, "batch_size": 1},
            "class_type": "EmptyLatentHunyuan3Dv2",
        },
        "6": {
            "inputs": {"clip_vision_output": ["13", 0]},
            "class_type": "Hunyuan3Dv2Conditioning",
        },
        "7": {
            "inputs": {
                "seed": seed,
                "steps": steps,
                "cfg": cfg,
                "sampler_name": "euler",
                "scheduler": "normal",
                "denoise": 1.0,
                "model": ["3", 0],
                "positive": ["6", 0],
                "negative": ["6", 1],
                "latent_image": ["4", 0],
            },
            "class_type": "KSampler",
        },
        "8": {
            "inputs": {
                "num_chunks": 8000,
                "octree_resolution": 256,
                "samples": ["7", 0],
                "vae": ["1", 2],
            },
            "class_type": "VAEDecodeHunyuan3D",
        },
        "9": {
            "inputs": {
                "algorithm": "surface net",
                "threshold": 0.6,
                "voxel": ["8", 0],
            },
            "class_type": "VoxelToMesh",
        },
        "10": {
            "inputs": {
                "filename_prefix": "ComfyUI_Hunyuan3D",
                "mesh": ["9", 0],
            },
            "class_type": "SaveGLB",
        },
        "13": {
            "inputs": {
                "crop": "center",
                "clip_vision": ["1", 1],
                "image": ["2", 0],
            },
            "class_type": "CLIPVisionEncode",
        },
    }

    if image_subfolder:
        workflow["2"]["inputs"]["subfolder"] = image_subfolder

    return workflow

