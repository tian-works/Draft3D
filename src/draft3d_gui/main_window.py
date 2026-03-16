"""
Qt main window wrapper for Draft3D.

This module provides a stable, package-based entry to the GUI so that
other code only needs to import from `draft3d_gui.main_window`,
while the legacy `GUI.py` file can remain unchanged internally.

In a later refactor, the actual Qt classes can be migrated here and
`GUI.py` can be reduced to a thin compatibility shim.
"""

from __future__ import annotations

import sys

try:
    # Prefer PySide6 if available (matches current GUI.py logic)
    from PySide6.QtWidgets import QApplication
except ImportError:  # pragma: no cover - fallback
    from PyQt5.QtWidgets import QApplication  # type: ignore[import]

# Import the existing Qt window implementation from the legacy module.
# This keeps behavior 100% the same while allowing code to depend on
# the package path `draft3d_gui.main_window.MainWindow`.
from GUI import MainWindow as _LegacyMainWindow  # type: ignore[import]


class MainWindow(_LegacyMainWindow):
    """
    Thin subclass of the legacy `GUI.MainWindow`.

    This class is here so that new code can import
    `draft3d_gui.main_window.MainWindow` without depending directly
    on the top-level `GUI.py` file.
    """

    pass


def main() -> None:
    """
    Application entry point for the Draft3D GUI.

    This mirrors the `main()` function in `GUI.py`, but is exposed
    through the `draft3d_gui` package.
    """
    app = QApplication(sys.argv)
    window = MainWindow()
    window.showMaximized()
    sys.exit(app.exec_())


__all__ = ["MainWindow", "main"]

