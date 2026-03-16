"""
HTTP client helpers for talking to the ComfyUI API.

These functions are extracted from the original `GUI.py` so they can be
reused from both the GUI layer and command-line / scripting use cases.
"""

from __future__ import annotations

import json
import time
import uuid
from typing import Any, Dict, Optional

import requests

from .config import COMFY_API_URL


def queue_prompt(prompt_workflow: Dict[str, Any]) -> Optional[str]:
    """
    Send a workflow to the ComfyUI API and return the prompt_id on success.
    """
    client_id = str(uuid.uuid4())

    # Debug: print LoadImage node configuration (if present)
    if "58" in prompt_workflow:
        try:
            print(
                f"[DEBUG] LoadImage node config: "
                f"{json.dumps(prompt_workflow['58'], indent=2, ensure_ascii=False)}"
            )
        except Exception:
            pass

    try:
        response = requests.post(
            f"{COMFY_API_URL}/prompt",
            json={"prompt": prompt_workflow, "client_id": client_id},
            timeout=30,
        )
    except requests.RequestException as e:
        print(f"Failed to submit workflow: {e}")
        return None

    if response.status_code == 200:
        result = response.json()
        print(f"Task submitted, ID: {result['prompt_id']}")
        return result["prompt_id"]

    print(f"Submit failed: {response.status_code} - {response.text}")
    # Print error details if available
    try:
        error_detail = response.json()
        print(f"Error detail: {json.dumps(error_detail, indent=2, ensure_ascii=False)}")
    except Exception:
        pass
    return None


def get_history(prompt_id: str) -> Optional[Dict[str, Any]]:
    """
    Get task execution history for a prompt id.
    """
    try:
        response = requests.get(f"{COMFY_API_URL}/history/{prompt_id}", timeout=30)
    except requests.RequestException as e:
        print(f"Failed to get history: {e}")
        return None

    if response.status_code == 200:
        return response.json()

    print(f"Failed to get history: {response.status_code}")
    return None


def wait_for_completion(
    prompt_id: str,
    timeout: int = 300,
    interval: int = 2,
) -> Optional[Dict[str, Any]]:
    """
    Wait for a task to complete and return the history entry when done.
    """
    start_time = time.time()

    while time.time() - start_time < timeout:
        history = get_history(prompt_id)

        if history and prompt_id in history:
            status = history[prompt_id]

            if status["status"].get("completed", False):
                print("Task completed!")
                return status
            if status["status"].get("error", False):
                print(f"Task failed: {status['status'].get('error', 'Unknown error')}")
                return None

        time.sleep(interval)

    print("Task timeout")
    return None


def upload_image_to_comfyui(image_path: str) -> tuple[str, str]:
    """
    Upload an image to ComfyUI's input directory.

    Returns:
        (filename, subfolder) on success; falls back to using the basename
        and an empty subfolder on failure.
    """
    import os

    try:
        with open(image_path, "rb") as f:
            files = {
                "image": (os.path.basename(image_path), f, "image/png"),
            }
            data = {"overwrite": "true"}

            response = requests.post(
                f"{COMFY_API_URL}/upload/image",
                files=files,
                data=data,
                timeout=30,
            )

        if response.status_code == 200:
            result = response.json()
            # ComfyUI returns: {"name": "filename.png", "subfolder": "", "type": "input"}
            filename = result.get("name", os.path.basename(image_path))
            subfolder = result.get("subfolder", "")
            file_type = result.get("type", "input")
            print(f"Image uploaded: {filename}, subfolder: {subfolder}, type: {file_type}")
            # Small delay to ensure ComfyUI finishes processing the upload
            time.sleep(0.5)
            return filename, subfolder

        print(f"Image upload failed: {response.status_code} - {response.text}")
        # If upload fails, fall back to using the filename directly (assuming it's already in input/)
        return os.path.basename(image_path), ""
    except Exception as e:
        print(f"Exception while uploading image: {e}")
        # If upload fails, fall back to using the filename directly (assuming it's already in input/)
        import os

        return os.path.basename(image_path), ""

