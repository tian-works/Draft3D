"""
Convenience script to launch the Draft3D GUI.

Usage (from the project root):

    python scripts/run_gui.py

This will delegate to the `draft3d_gui.app.main()` entry point, which for
now simply runs the existing `GUI.py` script. As the code is refactored
into the `draft3d_gui` package, no changes to this launcher will be needed.
"""

from __future__ import annotations

from draft3d_gui.app import main


if __name__ == "__main__":
    main()

