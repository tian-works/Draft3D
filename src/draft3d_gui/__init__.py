"""
Qt-based GUI package for the Draft3D project.

In the current transition phase, the main implementation still lives in the
top-level `GUI.py` file. The goal of this package is to progressively
refactor the GUI into:

- `draft3d_gui.main_window`: main application window and high-level layout.
- `draft3d_gui.widgets.*`: specialized widgets (sketch canvas, 3D viewer,
  parameter panels, etc.).
- `draft3d_gui.dialogs`: custom dialogs and message boxes.
- `draft3d_gui.app`: GUI entry point that creates and runs `QApplication`.

For now, `draft3d_gui.app` simply reuses the existing `GUI.py` entry point
so that the new structure remains backwards compatible.
"""

