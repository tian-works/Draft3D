"""
Configuration values for the Draft3D project.

These are kept minimal at this stage and mirror the constants used in the
existing `GUI.py`. As the code is refactored, more configuration options
can be centralized here.
"""

from __future__ import annotations

import os

# ComfyUI API endpoint (default local configuration)
COMFY_API_URL: str = "http://127.0.0.1:8188"

# Root output folder name used for generated images and 3D models
OUTPUT_FOLDER_NAME: str = "generated_images"


def get_output_root() -> str:
    """
    Return the absolute path to the root output folder.

    This mirrors the behavior in the original `GUI.py` where the
    output path is resolved relative to the current working directory.
    """
    return os.path.join(os.getcwd(), OUTPUT_FOLDER_NAME)

