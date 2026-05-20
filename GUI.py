import os
import sys

_project_root = os.path.dirname(os.path.abspath(__file__))
_src_path = os.path.join(_project_root, "src")
if _src_path not in sys.path:
    sys.path.insert(0, _src_path)

from draft3d_gui.main_window import MainWindow
from draft3d_gui.qt_compat import QApplication, run_app


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(run_app(app))


if __name__ == "__main__":
    main()
