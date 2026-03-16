"""
IO utilities for Draft3D.

This module provides small, reusable helpers that deal with the filesystem
and basic OS interactions. The functions here are adapted from the original
`GUI.py` implementation, but kept free of any GUI dependencies so they can
be reused in scripts or other tooling.
"""

from __future__ import annotations

import os
import platform
import subprocess
from datetime import datetime

from .config import get_output_root


def get_output_folder() -> str:
    """
    Get (and create if needed) the dated output folder under the root path.

    The folder structure is:
        <output_root>/<YYYY-MM-DD>

    where <output_root> is controlled by `draft3d.config`.
    """
    date_str = datetime.now().strftime("%Y-%m-%d")
    main_folder = get_output_root()

    if not os.path.exists(main_folder):
        os.makedirs(main_folder)

    date_folder = os.path.join(main_folder, date_str)
    if not os.path.exists(date_folder):
        os.makedirs(date_folder)

    return date_folder


def open_folder(folder_path: str) -> bool:
    """
    Open a folder in the OS file explorer (cross-platform best effort).

    Returns True on success, False otherwise.
    """
    if not os.path.exists(folder_path):
        return False

    try:
        system = platform.system()
        if system == "Windows":
            os.startfile(folder_path)  # type: ignore[attr-defined]
        elif system == "Darwin":  # macOS
            subprocess.Popen(["open", folder_path])
        else:  # Linux and others
            subprocess.Popen(["xdg-open", folder_path])
        return True
    except Exception:
        return False

