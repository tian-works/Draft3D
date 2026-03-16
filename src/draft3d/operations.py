"""
High-level generation and editing operations built on top of ComfyUI workflows.

These functions provide convenient entry points for:
- 2D image generation from prompt (+ optional sketch / ControlNet)
- 2D image editing
- Background removal
- 3D model generation from a 2D reference image
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import Any, Callable, Dict, Iterable, List, Optional

import requests

from .comfy_client import (
    queue_prompt,
    upload_image_to_comfyui,
    wait_for_completion,
)
from .config import COMFY_API_URL
from .io_utils import get_output_folder
from .workflows import (
    build_workflow,
    build_workflow_hunyuan3d,
    build_workflow_with_controlnet,
    build_workflow_z_image_turbo_edit,
)


def generate_image(
    prompt: str,
    seed: int = 0,
    steps: int = 4,
    cfg: float = 1.0,
    width: int = 512,
    height: int = 512,
    batch_size: int = 1,
    use_sketch: bool = False,
    sketch_path: Optional[str] = None,
    control_strength: float = 0.85,
    canny_low: float = 0.1,
    canny_high: float = 0.32,
    on_image_saved: Optional[Callable[[str, int, int], None]] = None,
) -> Optional[List[str]]:
    """
    Main function for generating images, supporting batch generation.

    Args:
        on_image_saved: optional callback called for each saved image with
                        (image_path, image_index, total_images).
    """
    sketch_filename: Optional[str] = None
    sketch_subfolder: Optional[str] = None

    # If using a sketch, upload the image first
    if use_sketch and sketch_path and os.path.exists(sketch_path):
        print("Uploading sketch image to ComfyUI...")
        sketch_filename, sketch_subfolder = upload_image_to_comfyui(sketch_path)
        if not sketch_filename:
            print("Warning: sketch upload failed, falling back to prompt‑only generation.")
            use_sketch = False

    # If using a sketch, use the ControlNet workflow; otherwise use the standard workflow
    if use_sketch and sketch_filename:
        print("Using ControlNet workflow (sketch + prompt)")
        print(f"Sketch filename: {sketch_filename}, subfolder: {sketch_subfolder}")
        print(f"ControlNet strength: {control_strength}, Canny thresholds: [{canny_low}, {canny_high}]")
        workflow = build_workflow_with_controlnet(
            prompt=prompt,
            seed=seed,
            steps=max(steps, 9) if steps < 9 else steps,  # ControlNet workflow recommends at least 9 steps
            cfg=cfg,
            batch_size=batch_size,
            sketch_filename=sketch_filename,
            sketch_subfolder=sketch_subfolder,
            control_strength=control_strength,
            canny_low=canny_low,
            canny_high=canny_high,
        )
        # Debug: print LoadImage node configuration from the workflow
        if "58" in workflow:
            try:
                import json

                print(f"LoadImage node inputs: {workflow['58']['inputs']}")
                print(f"Full LoadImage node: {json.dumps(workflow['58'], indent=2, ensure_ascii=False)}")
            except Exception:
                pass
    else:
        print("Using standard workflow (prompt‑only)")
        workflow = build_workflow(
            prompt=prompt,
            seed=seed,
            steps=steps,
            cfg=cfg,
            width=width,
            height=height,
            batch_size=batch_size,
            use_sketch=False,
            sketch_filename=None,
            sketch_subfolder=None,
        )

    prompt_id = queue_prompt(workflow)
    if not prompt_id:
        return None

    result = wait_for_completion(prompt_id)
    if not result:
        return None

    outputs: Dict[str, Any] = result.get("outputs", {})
    save_paths: List[str] = []

    # Generate a base timestamp (shared by all images in the same batch)
    base_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # First count the total number of images
    total_images = 0
    for _, node_data in outputs.items():
        if "images" in node_data:
            total_images += len(node_data["images"])

    image_index = 0  # Index of image within the same batch

    for _, node_data in outputs.items():
        if "images" not in node_data:
            continue

        for image_info in node_data["images"]:
            filename = image_info.get("filename")
            subfolder = image_info.get("subfolder")

            image_url = f"{COMFY_API_URL}/view"
            if subfolder:
                image_url += f"?subfolder={subfolder}&filename={filename}"
            else:
                image_url += f"?filename={filename}"

            print(f"Image generated: {filename}")
            print(f"Download URL: {image_url}")

            try:
                response = requests.get(image_url, timeout=60)
                if response.status_code == 200:
                    # Get output folder path
                    output_folder = get_output_folder()

                    # Build filename with timestamp and index
                    file_base, file_ext = os.path.splitext(filename)

                    if total_images > 1:
                        new_filename = f"{file_base}_{base_timestamp}_{image_index:02d}{file_ext}"
                    else:
                        new_filename = f"{file_base}_{base_timestamp}{file_ext}"

                    # Save to output folder
                    save_path = os.path.join(output_folder, new_filename)
                    with open(save_path, "wb") as f:
                        f.write(response.content)
                    print(f"Image saved to: {save_path}")
                    save_paths.append(save_path)

                    # If a callback is provided, call it immediately (for live preview)
                    if on_image_saved:
                        try:
                            on_image_saved(save_path, image_index, total_images)
                        except Exception as e:
                            print(f"on_image_saved callback raised an exception: {e}")

                    image_index += 1
            except requests.RequestException as e:
                print(f"Failed to download image: {e}")
            except Exception as e:
                print(f"Failed to save image: {e}")

    return save_paths or None


def edit_image(
    prompt: str,
    image_path: str,
    seed: int = -1,
    steps: int = 9,
    cfg: float = 1.0,
    control_strength: float = 0.85,
    canny_low: float = 0.1,
    canny_high: float = 0.32,
    batch_size: int = 1,
) -> Optional[List[str]]:
    """
    Main function for editing an existing image using the Z-Image-Turbo
    ControlNet workflow (single image, batch_size must be 1).
    """
    if not os.path.exists(image_path):
        print(f"Error: image file does not exist: {image_path}")
        return None

    # Upload the image to be edited
    print("Uploading image to be edited to ComfyUI...")
    image_filename, image_subfolder = upload_image_to_comfyui(image_path)
    if not image_filename:
        print("Error: image upload failed")
        return None

    print(f"Image uploaded: {image_filename}")
    print("Using Z-Image-Turbo ControlNet workflow for editing")
    print("Edit mode: single image (batch_size=1)")
    print(f"Edit prompt: {prompt}")
    print(f"ControlNet strength: {control_strength}, Canny thresholds: [{canny_low}, {canny_high}]")
    print(f"Steps: {steps}, CFG: {cfg}, Seed: {seed}")

    # Build workflow
    workflow = build_workflow_z_image_turbo_edit(
        prompt=prompt,
        image_filename=image_filename,
        image_subfolder=image_subfolder,
        seed=seed,
        steps=steps,
        cfg=cfg,
        control_strength=control_strength,
        canny_low=canny_low,
        canny_high=canny_high,
        batch_size=batch_size,
    )

    # Debug: print Prompt node configuration
    if "70:45" in workflow:
        print(f"Prompt node inputs: {workflow['70:45']['inputs']}")

    prompt_id = queue_prompt(workflow)
    if not prompt_id:
        return None

    print(f"Image edit task submitted, prompt_id: {prompt_id}")
    print("Waiting for edit to complete...")

    result = wait_for_completion(prompt_id)
    if not result:
        return None

    outputs: Dict[str, Any] = result.get("outputs", {})
    save_paths: List[str] = []

    base_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    total_images = 0
    for _, node_data in outputs.items():
        if "images" in node_data:
            total_images += len(node_data["images"])

    image_index = 0

    for _, node_data in outputs.items():
        if "images" not in node_data:
            continue

        for image_info in node_data["images"]:
            filename = image_info.get("filename")
            subfolder = image_info.get("subfolder")

            image_url = f"{COMFY_API_URL}/view"
            if subfolder:
                image_url += f"?subfolder={subfolder}&filename={filename}"
            else:
                image_url += f"?filename={filename}"

            print(f"Edited image generated: {filename}")
            print(f"Download URL: {image_url}")

            try:
                response = requests.get(image_url, timeout=60)
                if response.status_code == 200:
                    output_folder = get_output_folder()
                    file_base, file_ext = os.path.splitext(filename)

                    if total_images > 1:
                        new_filename = (
                            f"{file_base}_edited_{base_timestamp}_{image_index:02d}{file_ext}"
                        )
                    else:
                        new_filename = f"{file_base}_edited_{base_timestamp}{file_ext}"

                    save_path = os.path.join(output_folder, new_filename)

                    with open(save_path, "wb") as f:
                        f.write(response.content)

                    print(f"Edited image saved to: {save_path}")
                    save_paths.append(save_path)
                    image_index += 1
                else:
                    print(f"Failed to download edited image: HTTP {response.status_code}")
            except requests.RequestException as e:
                print(f"Failed to download edited image: {e}")
            except Exception as e:
                print(f"Failed to save edited image: {e}")

    if save_paths:
        output_folder = os.path.dirname(save_paths[0])
        print(f"✅ Edited images saved to folder: {output_folder}")
        if len(save_paths) == 1:
            print(f"Successfully edited single image: {os.path.basename(save_paths[0])}")
        else:
            print(f"Total edited images: {len(save_paths)}")

    return save_paths or None


def remove_background(image_path: str, output_path: Optional[str] = None) -> Optional[str]:
    """
    Remove the background from an image, keeping only the foreground subject.

    Returns:
        Path to the processed image, or None on failure.
    """
    if not os.path.exists(image_path):
        print(f"Error: image file does not exist: {image_path}")
        return None

    try:
        # Prefer using rembg if available
        try:
            from rembg import remove as rembg_remove  # type: ignore[import]

            print("Removing background using rembg...")
            with open(image_path, "rb") as input_file:
                input_data = input_file.read()

            output_data = rembg_remove(input_data)

            if output_path is None:
                base, ext = os.path.splitext(image_path)
                output_path = f"{base}_no_bg{ext}"

            with open(output_path, "wb") as output_file:
                output_file.write(output_data)

            print(f"Background removed (rembg), saved to: {output_path}")
            return output_path

        except ImportError:
            # Fallback: simple heuristic with Pillow + numpy
            print("rembg is not installed, using a simple heuristic method to remove background...")
            try:
                from PIL import Image  # type: ignore[import]
                import numpy as np  # type: ignore[import]

                img = Image.open(image_path).convert("RGBA")
                img_array = np.array(img)

                height, width = img_array.shape[:2]

                center_x, center_y = width // 2, height // 2
                max_dist = (center_x**2 + center_y**2) ** 0.5

                y_coords, x_coords = np.ogrid[:height, :width]
                dist_from_center = ((x_coords - center_x) ** 2 + (y_coords - center_y) ** 2) ** 0.5

                corner_colors = [
                    img_array[0, 0, :3],
                    img_array[0, width - 1, :3],
                    img_array[height - 1, 0, :3],
                    img_array[height - 1, width - 1, :3],
                ]

                bg_color = np.mean(corner_colors, axis=0)
                color_diff = ((img_array[:, :, :3] - bg_color) ** 2).sum(axis=2) ** 0.5

                threshold = 40
                bg_mask = color_diff < threshold
                img_array[bg_mask, 3] = 0

                # Edge-based refinement
                edge_mask = dist_from_center / max_dist > 0.8
                combined_mask = edge_mask & (color_diff < threshold * 1.5)
                img_array[combined_mask, 3] = 0

                result_img = Image.fromarray(img_array, "RGBA")

                if output_path is None:
                    base, ext = os.path.splitext(image_path)
                    output_path = f"{base}_no_bg{ext}"

                if not output_path.lower().endswith(".png"):
                    output_path = os.path.splitext(output_path)[0] + ".png"

                result_img.save(output_path, "PNG")
                print(f"Background removed (simple method), saved to: {output_path}")
                return output_path

            except ImportError:
                print("Error: Pillow or numpy is required for background removal fallback.")
                print("Install with: pip install Pillow numpy or pip install rembg")
                return None
            except Exception as e:
                print(f"Background removal failed: {e}")
                return None

    except Exception as e:
        print(f"Unexpected error during background removal: {e}")
        return None


def generate_3d_model(
    image_path: str,
    seed: int = 952805179515179,
    steps: int = 30,
    cfg: float = 5.0,
    resolution: int = 1024,
    remove_bg: bool = True,
) -> Optional[str]:
    """
    Main function for generating a 3D model (.glb file) from an input image.
    """
    if not os.path.exists(image_path):
        print(f"Error: image file does not exist: {image_path}")
        return None

    # Optionally remove background first
    processed_image_path = image_path
    temp_bg_removed = False
    if remove_bg:
        print("Removing background before 3D generation...")
        bg_removed_path = remove_background(image_path)
        if bg_removed_path:
            processed_image_path = bg_removed_path
            temp_bg_removed = True
            print(f"Background removed, using processed image: {processed_image_path}")
        else:
            print("Warning: background removal failed, using original image instead")
            processed_image_path = image_path

    # Upload reference image
    print("Uploading reference image to ComfyUI...")
    image_filename, image_subfolder = upload_image_to_comfyui(processed_image_path)
    if not image_filename:
        print("Error: image upload failed")
        if temp_bg_removed and os.path.exists(processed_image_path):
            try:
                os.remove(processed_image_path)
            except Exception:
                pass
        return None

    print(f"Reference image uploaded: {image_filename}")

    # Build workflow
    workflow = build_workflow_hunyuan3d(
        image_filename=image_filename,
        image_subfolder=image_subfolder,
        seed=seed,
        steps=steps,
        cfg=cfg,
        resolution=resolution,
    )

    # Submit task
    prompt_id = queue_prompt(workflow)
    if not prompt_id:
        if temp_bg_removed and os.path.exists(processed_image_path):
            try:
                os.remove(processed_image_path)
            except Exception:
                pass
        return None

    print(f"3D generation task submitted, prompt_id: {prompt_id}")
    print("Waiting for 3D generation to complete (Hunyuan3D is slow, 2–10 minutes is typical)...")

    # Wait for completion (longer timeout for 3D generation)
    result = wait_for_completion(prompt_id, timeout=600, interval=3)
    if not result:
        if temp_bg_removed and os.path.exists(processed_image_path):
            try:
                os.remove(processed_image_path)
            except Exception:
                pass
        return None

    output_folder = get_output_folder()

    try:
        outputs: Dict[str, Any] = result.get("outputs", {})
        if "10" in outputs and "3d" in outputs["10"]:
            glb_info = outputs["10"]["3d"][0]
            filename = glb_info.get("filename")
            subfolder = glb_info.get("subfolder", "")
            file_type = glb_info.get("type", "output")

            image_url = f"{COMFY_API_URL}/view"
            params = {
                "filename": filename,
                "subfolder": subfolder,
                "type": file_type,
            }
            image_url += "?" + "&".join(f"{k}={v}" for k, v in params.items() if v)

            print(f"3D model generated: {filename}")
            print(f"Download URL: {image_url}")

            try:
                response = requests.get(image_url, timeout=120, stream=True)
                if response.status_code == 200:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    file_base, file_ext = os.path.splitext(filename)
                    new_filename = f"{file_base}_{timestamp}{file_ext}"

                    save_path = os.path.join(output_folder, new_filename)
                    with open(save_path, "wb") as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)

                    file_size_mb = os.path.getsize(save_path) / (1024 * 1024)
                    print(f"3D model saved to: {save_path}")
                    print(f"File size: {file_size_mb:.2f} MB")

                    if temp_bg_removed and os.path.exists(processed_image_path):
                        try:
                            os.remove(processed_image_path)
                            print(f"Temporary file removed: {processed_image_path}")
                        except Exception as e:
                            print(f"Failed to remove temporary file: {e}")

                    return save_path

                print(f"Download failed: HTTP {response.status_code}")
                if temp_bg_removed and os.path.exists(processed_image_path):
                    try:
                        os.remove(processed_image_path)
                    except Exception:
                        pass
                return None
            except requests.RequestException as e:
                print(f"Failed to download 3D model: {e}")
                if temp_bg_removed and os.path.exists(processed_image_path):
                    try:
                        os.remove(processed_image_path)
                    except Exception:
                        pass
                return None

        print("Error: no 3D model output found in workflow result")
        if temp_bg_removed and os.path.exists(processed_image_path):
            try:
                os.remove(processed_image_path)
            except Exception:
                pass
        return None
    except Exception as e:
        print(f"Error while processing 3D model output: {e}")
        try:
            import traceback

            traceback.print_exc()
        except Exception:
            pass
        if temp_bg_removed and os.path.exists(processed_image_path):
            try:
                os.remove(processed_image_path)
            except Exception:
                pass
        return None

