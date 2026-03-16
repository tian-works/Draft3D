"""
Application entry point for the Draft3D GUI package.

This module now delegates to `draft3d_gui.main_window.main`, which in
turn uses the existing `GUI.MainWindow` implementation. Over time, the
Qt code can be migrated fully into `draft3d_gui`, but callers do not
need to change – they can always import `draft3d_gui.app.main`.
"""

from __future__ import annotations

from .main_window import main as _gui_main


def main() -> None:
    """Launch the Draft3D GUI."""
    _gui_main()


if __name__ == "__main__":
    main()


