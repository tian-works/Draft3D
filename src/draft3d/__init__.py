"""
Core, GUI-independent functionality for the Draft3D project.

This package is intended to host:

- Configuration and constants (API endpoints, default paths and parameters).
- IO utilities for managing output folders and opening results.
- Communication helpers for interacting with the ComfyUI HTTP API.
- Workflow building utilities that construct ComfyUI graph definitions.
- Logging and VTK/PyVista related helpers.

Initially, the existing functionality resides in the monolithic `GUI.py` file.
Over time, code can be gradually moved into the following modules:

- `draft3d.config`
- `draft3d.io_utils`
- `draft3d.comfy_client`
- `draft3d.workflows`
- `draft3d.logging_utils`
"""

