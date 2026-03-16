import sys
import os

# Add src directory to Python path so that draft3d and draft3d_gui packages can be imported
# This ensures GUI.py can be run directly from the project root
_project_root = os.path.dirname(os.path.abspath(__file__))
_src_path = os.path.join(_project_root, "src")
if _src_path not in sys.path:
    sys.path.insert(0, _src_path)

import uuid
import time
import json
import shutil
import requests
import subprocess
import platform
import logging
from datetime import datetime

from draft3d.io_utils import get_output_folder, open_folder
from draft3d.operations import generate_image, edit_image, generate_3d_model, remove_background


# Configure logging filter to suppress VTK OpenGL related errors
class VTKErrorFilter(logging.Filter):
    def filter(self, record):
        # Filter out VTK OpenGL related errors
        if record.getMessage():
            msg = str(record.getMessage())
            # Extended error pattern matching
            error_patterns = [
                'wglMakeCurrent',
                'invalid handle',
                'vtkWin32OpenGLRenderWindow',
                'vtkWin32OpenGLRenderWin',
                'vtkWin32OpenGLRen',
                'ERR|',
                'MakeCurrent()',
                'code 6',
                'invalid handle',
                'WARNING:root:None'
            ]
            if any(pattern in msg for pattern in error_patterns):
                return False  # Skip logging these messages

        # Also check logger name
        if record.name:
            if 'vtk' in record.name.lower() or 'opengl' in record.name.lower():
                return False

        return True  # Log all other messages


# Add filter to root logger
logging.getLogger().addFilter(VTKErrorFilter())

# Also add filter for known specific loggers
for logger_name in ['root', 'vtk', 'pyvista']:
    try:
        logger = logging.getLogger(logger_name)
        logger.addFilter(VTKErrorFilter())
        logger.setLevel(logging.ERROR)  # Only display ERROR level and above
    except:
        pass

# Configure VTK OutputWindow to suppress OpenGL errors
try:
    import vtk


    class SilentVTKOutput:
        """Silent VTK output class to suppress OpenGL related errors"""

        def DisplayText(self, text):
            if not text:
                return

            text_str = str(text)
            # Extended error pattern matching
            error_patterns = [
                'wglMakeCurrent',
                'invalid handle',
                'vtkWin32OpenGLRenderWindow',
                'vtkWin32OpenGLRenderWin',
                'vtkWin32OpenGLRen',
                'ERR|',
                'MakeCurrent()',
                'code 6',
                'invalid handle',
                'failed in MakeCurrent',
                'WARNING:root:None'
            ]

            # Do not output if it contains any error pattern
            if any(pattern in text_str for pattern in error_patterns):
                return  # Suppress these errors

            # Other errors can be printed here when debugging if needed
            # print(f"VTK: {text}")

        def DisplayErrorText(self, text):
            self.DisplayText(text)

        def DisplayWarningText(self, text):
            self.DisplayText(text)

        def DisplayGenericWarningText(self, text):
            self.DisplayText(text)

        def DisplayDebugText(self, text):
            self.DisplayText(text)


    # Configure VTK to use silent output
    try:
        silent_output = SilentVTKOutput()
        vtk.vtkOutputWindow.SetInstance(silent_output)
        vtk.vtkOutputWindow.SetGlobalWarningDisplay(0)
        # Disable all warning and error display
        try:
            vtk.vtkObject.GlobalWarningDisplayOff()
        except:
            pass
    except:
        pass

    # Try to set environment variable to suppress VTK output
    try:
        os.environ['VTK_LOGGING_LEVEL'] = 'ERROR'  # Only show ERROR level
    except:
        pass
except ImportError:
    # If vtk cannot be imported, simply skip
    pass

try:
    import numpy as np

    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None

# Try to import PySide6 (recommended, supports Python 3.12); fall back to PyQt5 if unavailable
try:
    from PySide6.QtCore import Qt, QPoint, QEvent
    from PySide6.QtGui import QPixmap, QPainter, QPen, QColor, QKeySequence, QShortcut
    from PySide6.QtWidgets import (
        QApplication,
        QWidget,
        QDialog,
        QLabel,
        QPushButton,
        QVBoxLayout,
        QHBoxLayout,
        QTextEdit,
        QSpinBox,
        QDoubleSpinBox,
        QFileDialog,
        QGroupBox,
        QComboBox,
        QInputDialog,
        QMessageBox,
        QListWidget,
        QListWidgetItem,
        QCheckBox,
        QSizePolicy,
        QSlider,
        QScrollArea,
        QSplitter,
        QRadioButton,
        QButtonGroup,
        QTabWidget,
        QColorDialog,
    )

    QT_VERSION = "PySide6"
    print("Using PySide6")
except ImportError:
    from PyQt5.QtCore import Qt, QPoint, QEvent
    from PyQt5.QtGui import QPixmap, QPainter, QPen, QColor, QKeySequence
    from PyQt5.QtWidgets import (
        QApplication,
        QWidget,
        QDialog,
        QLabel,
        QPushButton,
        QVBoxLayout,
        QHBoxLayout,
        QTextEdit,
        QSpinBox,
        QDoubleSpinBox,
        QFileDialog,
        QGroupBox,
        QComboBox,
        QInputDialog,
        QMessageBox,
        QListWidget,
        QListWidgetItem,
        QCheckBox,
        QSizePolicy,
        QSlider,
        QScrollArea,
        QSplitter,
        QRadioButton,
        QButtonGroup,
        QTabWidget,
        QColorDialog,
        QShortcut,
    )

    QT_VERSION = "PyQt5"
    print("Using PyQt5")

# Try to import PyVista for 3D model visualization
try:
    import pyvista as pv
    from pyvistaqt import QtInteractor

    PYVISTA_AVAILABLE = True
    print("PyVista loaded, 3D model preview is available")
except ImportError:
    PYVISTA_AVAILABLE = False
    pv = None
    QtInteractor = None
    print("Warning: PyVista is not installed, 3D model preview will be unavailable")
    print("Please install: pip install pyvista pyvistaqt")

# ComfyUI API configuration
API_URL = "http://127.0.0.1:8188"  # ComfyUI API endpoint

OUTPUT_FOLDER = "generated_images"  # Root output folder name


def build_workflow(prompt, seed=0, steps=4, cfg=1.0, width=512, height=512, batch_size=1, use_sketch=False,
                   sketch_filename=None, sketch_subfolder=None):
    # Deprecated local implementation; kept only for backward compatibility.
    # The real implementation now lives in `draft3d.workflows.build_workflow`.
    from draft3d.workflows import build_workflow as _build_workflow

    return _build_workflow(
        prompt=prompt,
        seed=seed,
        steps=steps,
        cfg=cfg,
        width=width,
        height=height,
        batch_size=batch_size,
        use_sketch=use_sketch,
        sketch_filename=sketch_filename,
        sketch_subfolder=sketch_subfolder,
    )


def build_workflow_z_image_turbo_edit(prompt, image_filename, image_subfolder=None, seed=-1, steps=9, cfg=1.0,
                                      control_strength=0.85, canny_low=0.1, canny_high=0.32, batch_size=1):
    # Deprecated local implementation; kept only for backward compatibility.
    # The real implementation now lives in `draft3d.workflows.build_workflow_z_image_turbo_edit`.
    from draft3d.workflows import build_workflow_z_image_turbo_edit as _build_edit

    return _build_edit(
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


def build_workflow_img2img(prompt, image_filename, image_subfolder=None, seed=0, steps=4, cfg=1.0, denoise=0.75,
                           width=512, height=512, batch_size=1):
    # Deprecated local implementation; kept only for backward compatibility.
    # The real implementation now lives in `draft3d.workflows.build_workflow_img2img`.
    from draft3d.workflows import build_workflow_img2img as _build_img2img

    return _build_img2img(
        prompt=prompt,
        image_filename=image_filename,
        image_subfolder=image_subfolder,
        seed=seed,
        steps=steps,
        cfg=cfg,
        denoise=denoise,
        width=width,
        height=height,
        batch_size=batch_size,
    )


def build_workflow_hunyuan3d(image_filename, image_subfolder=None, seed=952805179515179, steps=30, cfg=5.0,
                             resolution=1024):
    # Deprecated local implementation; kept only for backward compatibility.
    # The real implementation now lives in `draft3d.workflows.build_workflow_hunyuan3d`.
    from draft3d.workflows import build_workflow_hunyuan3d as _build_hy3d

    return _build_hy3d(
        image_filename=image_filename,
        image_subfolder=image_subfolder,
        seed=seed,
        steps=steps,
        cfg=cfg,
        resolution=resolution,
    )


def generate_image(prompt, seed=0, steps=4, cfg=1.0, width=512, height=512, batch_size=1, use_sketch=False,
                   sketch_path=None, control_strength=0.85, canny_low=0.1, canny_high=0.32, on_image_saved=None):
    # Thin wrapper kept for backward compatibility; delegates to `draft3d.operations.generate_image`.
    from draft3d.operations import generate_image as _generate_image

    return _generate_image(
        prompt=prompt,
        seed=seed,
        steps=steps,
        cfg=cfg,
        width=width,
        height=height,
        batch_size=batch_size,
        use_sketch=use_sketch,
        sketch_path=sketch_path,
        control_strength=control_strength,
        canny_low=canny_low,
        canny_high=canny_high,
        on_image_saved=on_image_saved,
    )


def edit_image(prompt, image_path, seed=-1, steps=9, cfg=1.0, control_strength=0.85, canny_low=0.1, canny_high=0.32,
               batch_size=1):
    # Thin wrapper kept for backward compatibility; delegates to `draft3d.operations.edit_image`.
    from draft3d.operations import edit_image as _edit_image

    return _edit_image(
        prompt=prompt,
        image_path=image_path,
        seed=seed,
        steps=steps,
        cfg=cfg,
        control_strength=control_strength,
        canny_low=canny_low,
        canny_high=canny_high,
        batch_size=batch_size,
    )


def remove_background(image_path, output_path=None):
    # Thin wrapper kept for backward compatibility; delegates to `draft3d.operations.remove_background`.
    from draft3d.operations import remove_background as _remove_background

    return _remove_background(image_path=image_path, output_path=output_path)


def generate_3d_model(image_path, seed=952805179515179, steps=30, cfg=5.0, resolution=1024, remove_bg=True):
    # Thin wrapper kept for backward compatibility; delegates to `draft3d.operations.generate_3d_model`.
    from draft3d.operations import generate_3d_model as _generate_3d_model

    return _generate_3d_model(
        image_path=image_path,
        seed=seed,
        steps=steps,
        cfg=cfg,
        resolution=resolution,
        remove_bg=remove_bg,
    )


class NoWheelSlider(QSlider):
    """Slider that ignores mouse wheel events; can only be adjusted by dragging."""

    def wheelEvent(self, event):
        # Ignore wheel events to disable wheel‑based adjustment
        event.ignore()


class CustomSpinBox(QSpinBox):
    """Custom QSpinBox with visible triangle arrows on buttons."""
    
    def paintEvent(self, event):
        super().paintEvent(event)
        try:
            if QT_VERSION == "PySide6":
                from PySide6.QtWidgets import QStyleOptionSpinBox, QStyle
                from PySide6.QtGui import QPolygon
            else:
                from PyQt5.QtWidgets import QStyleOptionSpinBox, QStyle
                from PyQt5.QtGui import QPolygon
        except:
            return
            
        option = QStyleOptionSpinBox()
        self.initStyleOption(option)
        
        style = self.style()
        up_rect = style.subControlRect(QStyle.CC_SpinBox, option, QStyle.SC_SpinBoxUp, self)
        down_rect = style.subControlRect(QStyle.CC_SpinBox, option, QStyle.SC_SpinBoxDown, self)
        
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Draw up arrow (triangle pointing up)
        if up_rect.isValid():
            arrow_color = QColor("#9ca3af")
            painter.setPen(Qt.NoPen)
            painter.setBrush(arrow_color)
            center_x = up_rect.center().x()
            center_y = up_rect.center().y()
            top_y = center_y - 3
            bottom_y = center_y + 3
            size = 2.5
            points = [QPoint(center_x, top_y), QPoint(center_x - size, bottom_y), QPoint(center_x + size, bottom_y)]
            painter.drawPolygon(QPolygon(points))
        
        # Draw down arrow (triangle pointing down)
        if down_rect.isValid():
            arrow_color = QColor("#9ca3af")
            painter.setPen(Qt.NoPen)
            painter.setBrush(arrow_color)
            center_x = down_rect.center().x()
            center_y = down_rect.center().y()
            top_y = center_y - 3
            bottom_y = center_y + 3
            size = 2.5
            points = [QPoint(center_x, bottom_y), QPoint(center_x - size, top_y), QPoint(center_x + size, top_y)]
            painter.drawPolygon(QPolygon(points))
        
        painter.end()


class CustomDoubleSpinBox(QDoubleSpinBox):
    """Custom QDoubleSpinBox with visible triangle arrows on buttons."""
    
    def paintEvent(self, event):
        super().paintEvent(event)
        try:
            if QT_VERSION == "PySide6":
                from PySide6.QtWidgets import QStyleOptionSpinBox, QStyle
                from PySide6.QtGui import QPolygon
            else:
                from PyQt5.QtWidgets import QStyleOptionSpinBox, QStyle
                from PyQt5.QtGui import QPolygon
        except:
            return
            
        option = QStyleOptionSpinBox()
        self.initStyleOption(option)
        
        style = self.style()
        up_rect = style.subControlRect(QStyle.CC_SpinBox, option, QStyle.SC_SpinBoxUp, self)
        down_rect = style.subControlRect(QStyle.CC_SpinBox, option, QStyle.SC_SpinBoxDown, self)
        
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Draw up arrow
        if up_rect.isValid():
            arrow_color = QColor("#9ca3af")
            painter.setPen(Qt.NoPen)
            painter.setBrush(arrow_color)
            center_x = up_rect.center().x()
            center_y = up_rect.center().y()
            top_y = center_y - 3
            bottom_y = center_y + 3
            size = 2.5
            points = [QPoint(center_x, top_y), QPoint(center_x - size, bottom_y), QPoint(center_x + size, bottom_y)]
            painter.drawPolygon(QPolygon(points))
        
        # Draw down arrow
        if down_rect.isValid():
            arrow_color = QColor("#9ca3af")
            painter.setPen(Qt.NoPen)
            painter.setBrush(arrow_color)
            center_x = down_rect.center().x()
            center_y = down_rect.center().y()
            top_y = center_y - 3
            bottom_y = center_y + 3
            size = 2.5
            points = [QPoint(center_x, bottom_y), QPoint(center_x - size, top_y), QPoint(center_x + size, top_y)]
            painter.drawPolygon(QPolygon(points))
        
        painter.end()


class ImageZoomDialog(QDialog):
    """Image zoom dialog that supports mouse wheel zoom."""

    def __init__(self, image_path, parent=None):
        super().__init__(parent)
        self.setWindowTitle("🔍 Image Preview")
        self.setModal(False)  # Non‑modal so that outside clicks can be detected
        self.setWindowFlags(Qt.Dialog | Qt.WindowCloseButtonHint | Qt.WindowTitleHint)

        # Set dialog size
        self.resize(1000, 800)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)

        # Scroll area to support panning when the image is zoomed in
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setAlignment(Qt.AlignCenter)
        scroll_area.setStyleSheet("""
            QScrollArea {
                border: 2px solid #e5e7eb;
                border-radius: 10px;
                background-color: #f9fafb;
            }
        """)

        # Create image label
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setScaledContents(False)
        self.image_label.setStyleSheet("background-color: transparent;")

        # Load original image
        self.original_pixmap = QPixmap(image_path)
        if not self.original_pixmap.isNull():
            # Initialize zoom scale
            pix_width = self.original_pixmap.width()
            pix_height = self.original_pixmap.height()

            # Compute initial scale so that the image fits the dialog
            dialog_width = 980
            dialog_height = 750
            width_ratio = dialog_width / pix_width
            height_ratio = dialog_height / pix_height
            self.scale_ratio = min(width_ratio, height_ratio, 1.0)  # Do not upscale initially, only shrink

            # Render initial image
            self.update_image_display()
        else:
            self.image_label.setText("Unable to load image")
            self.image_label.setStyleSheet("color: #ef4444; font-size: 12pt; background-color: transparent;")
            self.original_pixmap = None
            self.scale_ratio = 1.0

        # Add image label into scroll area
        scroll_area.setWidget(self.image_label)
        layout.addWidget(scroll_area)

        # Save scroll area reference for wheel handling
        self.scroll_area = scroll_area

        # Install event filters on scroll area and label to capture wheel events
        scroll_area.installEventFilter(self)
        self.image_label.installEventFilter(self)
        # Enable mouse tracking so wheel events are captured properly
        scroll_area.setMouseTracking(True)
        self.image_label.setMouseTracking(True)

        # Button row
        button_layout = QHBoxLayout()
        button_layout.addStretch()

        # Reset zoom button
        reset_zoom_btn = QPushButton("🔄 Reset Zoom")
        reset_zoom_btn.setStyleSheet("""
            QPushButton {
                background-color: #6b7280;
                color: white;
                border: none;
                border-radius: 8px;
                padding: 8px 20px;
                font-size: 10pt;
                font-weight: 600;
            }
            QPushButton:hover {
                background-color: #4b5563;
            }
        """)
        reset_zoom_btn.clicked.connect(self.reset_zoom)
        button_layout.addWidget(reset_zoom_btn)

        # Close button
        close_btn = QPushButton("Close")
        close_btn.setStyleSheet("""
            QPushButton {
                background-color: #6366f1;
                color: white;
                border: none;
                border-radius: 8px;
                padding: 8px 20px;
                font-size: 10pt;
                font-weight: 600;
            }
            QPushButton:hover {
                background-color: #4f46e5;
            }
        """)
        close_btn.clicked.connect(self.accept)
        button_layout.addWidget(close_btn)

        layout.addLayout(button_layout)

        # Install a global event filter to detect clicks outside the dialog
        self.app = QApplication.instance()
        if self.app:
            self.app.installEventFilter(self)

    def update_image_display(self):
        """Update displayed image according to current zoom scale."""
        if self.original_pixmap is None or self.original_pixmap.isNull():
            return

        pix_width = self.original_pixmap.width()
        pix_height = self.original_pixmap.height()

        # Compute new size
        new_width = int(pix_width * self.scale_ratio)
        new_height = int(pix_height * self.scale_ratio)

        # Scale image
        scaled_pix = self.original_pixmap.scaled(
            new_width,
            new_height,
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )

        self.image_label.setPixmap(scaled_pix)
        # Resize label to match new pixmap size
        self.image_label.resize(scaled_pix.size())

    def reset_zoom(self):
        """Reset zoom scale back to the initial fit‑to‑window size."""
        if self.original_pixmap is None or self.original_pixmap.isNull():
            return

        # Recompute initial scale (fit to dialog)
        pix_width = self.original_pixmap.width()
        pix_height = self.original_pixmap.height()
        dialog_width = 980
        dialog_height = 750
        width_ratio = dialog_width / pix_width
        height_ratio = dialog_height / pix_height
        self.scale_ratio = min(width_ratio, height_ratio, 1.0)  # Do not upscale initially, only shrink

        # Re‑render image
        self.update_image_display()

    def wheelEvent(self, event):
        """Handle mouse wheel events to zoom the image."""
        if self.original_pixmap is None or self.original_pixmap.isNull():
            return super().wheelEvent(event)

        # Wheel delta (positive when scrolling up, negative when down)
        delta = event.angleDelta().y()

        # Zoom factor (change 10% per wheel notch)
        zoom_factor = 1.1 if delta > 0 else 0.9

        # Update scale ratio
        self.scale_ratio *= zoom_factor

        # Clamp zoom range (10%–1000%)
        self.scale_ratio = max(0.1, min(10.0, self.scale_ratio))

        # Re‑render image
        self.update_image_display()

        # Accept event so it is not propagated further
        event.accept()

    def eventFilter(self, obj, event):
        """Event filter that supports wheel zooming and closing on outside click."""
        # Handle wheel events occurring on scroll area or image label
        if event.type() == QEvent.Wheel:
            if hasattr(self, 'scroll_area') and (obj == self.scroll_area or obj == self.image_label):
                # Forward wheel events to dialog's wheelEvent handler
                self.wheelEvent(event)
                return True

        # Close dialog when clicking outside its bounds
        if event.type() == QEvent.MouseButtonPress and self.isVisible():
            try:
                # Get global mouse position
                global_pos = event.globalPos()

                # Get dialog rectangle in global coordinates
                dialog_rect = self.frameGeometry()

                # Check whether click is outside dialog
                if not dialog_rect.contains(global_pos):
                    # If click landed on the dialog or its children, do not close
                    clicked_widget = QApplication.widgetAt(global_pos)
                    if clicked_widget:
                        if clicked_widget == self or self.isAncestorOf(clicked_widget):
                            return super().eventFilter(obj, event)
                    # Click was outside, close the dialog
                    self.reject()
                    return True
            except Exception:
                # On any error, fall back to default behavior
                pass

        return super().eventFilter(obj, event)

    def closeEvent(self, event):
        """Remove global event filter when the dialog is closed."""
        if hasattr(self, 'app') and self.app:
            self.app.removeEventFilter(self)
        super().closeEvent(event)


def is_plotter_valid(plotter):
    """Check whether a PyVista plotter is still valid (window handle not broken)."""
    try:
        if not plotter:
            return False
        if not hasattr(plotter, 'renderer') or not plotter.renderer:
            return False
        # Check that the render window exists (without calling potentially blocking methods)
        try:
            render_window = plotter.renderer.GetRenderWindow()
            if not render_window:
                return False
            # Only check basic attributes; avoid expensive calls like GetGenericWindowId()
            return True
        except:
            return False
    except:
        return False


def safe_render(plotter):
    """Render safely, guarding against invalid window handles."""
    if not is_plotter_valid(plotter):
        return False
    try:
        plotter.render()
        return True
    except Exception as e:
        # Ignore invalid handle errors
        if "wglMakeCurrent" not in str(e) and "invalid handle" not in str(e):
            print(f"Non‑critical render error: {e}")
        return False


def remove_axes_from_plotter(plotter):
    """Remove all axis/orientation actors from a PyVista plotter."""
    try:
        if not is_plotter_valid(plotter):
            return
        renderer = plotter.renderer
        if not hasattr(renderer, 'RemoveActor'):
            return

        # 移除所有可能的axes actors
        actors = renderer.GetActors()
        actors.InitTraversal()
        actor = actors.GetNextItem()
        while actor:
            if actor and hasattr(actor, 'GetClassName'):
                class_name = actor.GetClassName()
                if 'Axes' in class_name or 'OrientationMarker' in class_name or 'vtkAxesActor' in class_name:
                    try:
                        renderer.RemoveActor(actor)
                    except:
                        pass
            actor = actors.GetNextItem()

        # 尝试通过plotter的属性移除
        try:
            if hasattr(plotter, 'axes_actor') and plotter.axes_actor:
                plotter.remove_actor(plotter.axes_actor)
        except:
            pass
    except:
        pass


def filter_external_boxes(mesh):
    """
    Filter out outer boxes/walls from a GLB scene, keeping only the main model.
    Outer boxes are typically large, simple bounding geometry.
    """
    try:
        # If this is a MultiBlock (multiple meshes), try to remove outer container geometry
        if hasattr(mesh, 'n_blocks') and mesh.n_blocks > 1:
            # Compute volume and bounding box for each block
            blocks_info = []
            for i in range(mesh.n_blocks):
                block = mesh[i]
                if block is not None and hasattr(block, 'n_cells') and block.n_cells > 0:
                    bounds = block.bounds
                    # Bounding‑box volume
                    volume = (bounds[1] - bounds[0]) * (bounds[3] - bounds[2]) * (bounds[5] - bounds[4])
                    # Maximum dimension length
                    max_dim = max(bounds[1] - bounds[0], bounds[3] - bounds[2], bounds[5] - bounds[4])
                    blocks_info.append({
                        'index': i,
                        'block': block,
                        'volume': volume,
                        'max_dim': max_dim,
                        'n_cells': block.n_cells
                    })

            if blocks_info:
                # Sort by volume; the largest block may be an outer box
                blocks_info.sort(key=lambda x: x['volume'], reverse=True)

                # If the largest block is much bigger (e.g. >2×), treat it as an outer box
                if len(blocks_info) > 1:
                    largest_volume = blocks_info[0]['volume']
                    second_volume = blocks_info[1]['volume']

                    # If the largest block is significantly larger, filter it out
                    if largest_volume > second_volume * 2:
                        print(f"Detected external box, filtering it out (volume: {largest_volume:.2f} vs {second_volume:.2f})")
                        # Merge all remaining blocks
                        filtered_blocks = [info['block'] for info in blocks_info[1:]]
                        if filtered_blocks:
                            # Merge all remaining blocks
                            from pyvista import MultiBlock
                            combined = MultiBlock(filtered_blocks)
                            return combined.extract_geometry()
                        else:
                            return blocks_info[0]['block']

                # If all blocks are similar in size, keep them all (likely parts of the model)
                filtered_blocks = [info['block'] for info in blocks_info]
                from pyvista import MultiBlock
                combined = MultiBlock(filtered_blocks)
                return combined.extract_geometry()

        # Single mesh – nothing to filter
        return mesh
    except Exception as e:
        print(f"Error while filtering external boxes: {e}, using original mesh instead")
        import traceback
        traceback.print_exc()
        return mesh


class Model3DViewerDialog(QDialog):
    """Standalone 3D model viewer window."""

    def __init__(self, glb_path, parent=None):
        super().__init__(parent)
        self.setWindowTitle("🎲 3D Model Viewer")
        self.setModal(False)
        self.setWindowFlags(Qt.Dialog | Qt.WindowCloseButtonHint | Qt.WindowTitleHint | Qt.WindowMinMaxButtonsHint)

        # Flag indicating whether the window is in the process of closing
        self._is_closing = False

        # Save reference to parent window
        self.parent_window = parent

        # Initial window size
        self.resize(1200, 900)

        # Store mesh reference and current settings
        self.mesh = None
        self.mesh_actor = None
        self.original_glb_path = glb_path  # Original GLB path

        # Pull current display settings from main window if available, otherwise use defaults
        if parent and hasattr(parent, 'model_3d_color'):
            self.current_color = parent.model_3d_color
            self.show_edges = parent.model_3d_show_edges
            self.smooth_shading = parent.model_3d_smooth_shading
            self.ambient_light = parent.model_3d_ambient_light
            self.diffuse_light = parent.model_3d_diffuse_light
        else:
            # Default settings
            self.current_color = (1.0, 1.0, 1.0)  # default white
            self.show_edges = False
            self.smooth_shading = True
            self.ambient_light = 0.3
            self.diffuse_light = 0.7
 
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)

        # Left: 3D view
        view_layout = QVBoxLayout()
        view_layout.setContentsMargins(0, 0, 0, 0)

        if PYVISTA_AVAILABLE:
            # Use PyVista to display 3D model
            try:
                # Suppress OpenGL warnings
                import warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    self.plotter = QtInteractor(self)

                self.plotter.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

                # Configure PyVista plotter (guard against OpenGL issues)
                try:
                    self.plotter.set_background((0.06, 0.06, 0.12))
                    self.plotter.enable_trackball_style()
                except Exception as e:
                    print(f"Plotter configuration warning (can be ignored): {e}")

                # Load GLB model
                try:
                    self.mesh = pv.read(glb_path)
                    # Filter out external boxes/walls
                    self.mesh = filter_external_boxes(self.mesh)
                    self.update_mesh_display()
                    try:
                        self.plotter.reset_camera()
                        self.plotter.view_isometric()
                        # Ensure axes are not shown
                        remove_axes_from_plotter(self.plotter)
                    except Exception as e:
                        if "wglMakeCurrent" not in str(e) and "句柄无效" not in str(e):
                            print(f"相机设置警告（可忽略）: {e}")

                    print(f"Loading 3D model in separate window: {glb_path}")
                except Exception as e:
                    print(f"Failed to load 3D model: {e}")
                    import traceback
                    traceback.print_exc()
                    self.plotter.add_text(
                        f"Load failed: {str(e)}\nPath: {glb_path}",
                        position='upper_left',
                        font_size=14,
                        color='red'
                    )

                view_layout.addWidget(self.plotter.interactor, stretch=10)

                # Hint label for controls
                label = QLabel("Left drag to rotate | Scroll to zoom | Right drag to pan | Middle click to reset view")
                label.setStyleSheet("color: white; background: rgba(0,0,0,140); padding: 8px; border-radius: 4px;")
                view_layout.addWidget(label)

                # Control panel
                control_panel = QGroupBox("🎨 Display Settings")
                control_panel.setStyleSheet("""
                    QGroupBox {
                        font-size: 10pt;
                        font-weight: 600;
                        color: #1f2937;
                        border: 2px solid #e5e7eb;
                        border-radius: 8px;
                        margin-top: 10px;
                        padding-top: 15px;
                        background-color: #ffffff;
                    }
                    QGroupBox::title {
                        font-family: 'Times New Roman', serif;
                        subcontrol-origin: margin;
                        left: 10px;
                        padding: 0 5px;
                    }
                """)
                control_layout = QVBoxLayout()
                control_layout.setSpacing(10)

                # Surface color
                color_layout = QHBoxLayout()
                color_label = QLabel("Surface Color:")
                color_label.setStyleSheet("font-size: 9pt; min-width: 80px;")
                color_layout.addWidget(color_label)

                self.color_button = QPushButton()
                self.color_button.setFixedSize(40, 30)
                self.color_button.setStyleSheet(
                    f"background-color: rgb({int(self.current_color[0] * 255)}, {int(self.current_color[1] * 255)}, {int(self.current_color[2] * 255)}); border: 2px solid #d1d5db; border-radius: 4px;")
                self.color_button.clicked.connect(self.on_color_changed)
                color_layout.addWidget(self.color_button)
                color_layout.addStretch()
                control_layout.addLayout(color_layout)

                # Wireframe toggle
                self.edges_checkbox = QCheckBox("Show Wireframe")
                self.edges_checkbox.setChecked(self.show_edges)
                self.edges_checkbox.stateChanged.connect(self.on_edges_changed)
                self.edges_checkbox.setStyleSheet("font-size: 9pt;")
                control_layout.addWidget(self.edges_checkbox)

                # Smooth shading toggle
                self.smooth_checkbox = QCheckBox("Smooth Shading")
                self.smooth_checkbox.setChecked(self.smooth_shading)
                self.smooth_checkbox.stateChanged.connect(self.on_smooth_shading_changed)
                self.smooth_checkbox.setStyleSheet("font-size: 9pt;")
                control_layout.addWidget(self.smooth_checkbox)

                # Ambient light intensity
                ambient_layout = QHBoxLayout()
                ambient_label = QLabel("Ambient Light:")
                ambient_label.setStyleSheet("font-size: 9pt; min-width: 80px;")
                ambient_layout.addWidget(ambient_label)

                self.ambient_slider = NoWheelSlider(Qt.Horizontal)
                self.ambient_slider.setRange(0, 100)
                self.ambient_slider.setValue(int(self.ambient_light * 100))
                # While dragging, update label only; apply on release
                self.ambient_slider.valueChanged.connect(self.on_ambient_light_label_changed)
                self.ambient_slider.sliderReleased.connect(self.on_ambient_light_changed)
                self.ambient_slider.setStyleSheet("""
                    QSlider::groove:horizontal {
                        border: 1px solid #d1d5db;
                        height: 6px;
                        background: #e5e7eb;
                        border-radius: 3px;
                    }
                    QSlider::handle:horizontal {
                        background: #6366f1;
                        border: 1px solid #4f46e5;
                        width: 16px;
                        margin: -5px 0;
                        border-radius: 8px;
                    }
                """)
                ambient_layout.addWidget(self.ambient_slider)
                self.ambient_label = QLabel(f"{self.ambient_light:.2f}")
                self.ambient_label.setStyleSheet("font-size: 9pt; min-width: 40px;")
                ambient_layout.addWidget(self.ambient_label)
                control_layout.addLayout(ambient_layout)

                # Diffuse light intensity
                diffuse_layout = QHBoxLayout()
                diffuse_label = QLabel("Diffuse Light:")
                diffuse_label.setStyleSheet("font-size: 9pt; min-width: 80px;")
                diffuse_layout.addWidget(diffuse_label)

                self.diffuse_slider = NoWheelSlider(Qt.Horizontal)
                self.diffuse_slider.setRange(0, 100)
                self.diffuse_slider.setValue(int(self.diffuse_light * 100))
                # While dragging, update label only; apply on release
                self.diffuse_slider.valueChanged.connect(self.on_diffuse_light_label_changed)
                self.diffuse_slider.sliderReleased.connect(self.on_diffuse_light_changed)
                self.diffuse_slider.setStyleSheet("""
                    QSlider::groove:horizontal {
                        border: 1px solid #d1d5db;
                        height: 6px;
                        background: #e5e7eb;
                        border-radius: 3px;
                    }
                    QSlider::handle:horizontal {
                        background: #6366f1;
                        border: 1px solid #4f46e5;
                        width: 16px;
                        margin: -5px 0;
                        border-radius: 8px;
                    }
                """)
                diffuse_layout.addWidget(self.diffuse_slider)
                self.diffuse_label = QLabel(f"{self.diffuse_light:.2f}")
                self.diffuse_label.setStyleSheet("font-size: 9pt; min-width: 40px;")
                diffuse_layout.addWidget(self.diffuse_label)
                control_layout.addLayout(diffuse_layout)

                # Save button
                save_btn = QPushButton("💾 Save")
                save_btn.setStyleSheet("""
                    QPushButton {
                        background-color: #10b981;
                        color: white;
                        border: none;
                        border-radius: 6px;
                        padding: 8px;
                        font-size: 9pt;
                        font-weight: 600;
                    }
                    QPushButton:hover {
                        background-color: #059669;
                    }
                """)
                save_btn.clicked.connect(self.on_save_3d_model)
                control_layout.addWidget(save_btn)

                control_panel.setLayout(control_layout)

                # Right: control panel
                right_panel = QWidget()
                right_panel.setFixedWidth(250)
                right_panel.setStyleSheet("background-color: #f9fafb; border-radius: 8px;")
                right_layout = QVBoxLayout(right_panel)
                right_layout.setContentsMargins(10, 10, 10, 10)
                right_layout.addWidget(control_panel)
                right_layout.addStretch()

                main_layout.addLayout(view_layout, stretch=10)
                main_layout.addWidget(right_panel)

            except Exception as e:
                print(f"Failed to initialize PyVista: {e}")
                import traceback
                traceback.print_exc()
                placeholder = QLabel(
                    f"PyVista initialization failed: {str(e)}\n\nPlease check if PyVista is installed correctly")
                placeholder.setAlignment(Qt.AlignCenter)
                placeholder.setStyleSheet("""
                    QLabel {
                        border: 2px dashed #d1d5db;
                        border-radius: 10px;
                        background-color: #f9fafb;
                        color: #6b7280;
                        font-size: 12pt;
                        padding: 40px;
                    }
                """)
                view_layout.addWidget(placeholder)
        else:
            # If PyVista is unavailable, show instructions instead
            placeholder = QLabel(
                "3D model preview requires PyVista\n\nPlease install: pip install pyvista pyvistaqt\n\nTo view 3D models in a separate window")
            placeholder.setAlignment(Qt.AlignCenter)
            placeholder.setStyleSheet("""
                QLabel {
                    border: 2px dashed #d1d5db;
                    border-radius: 10px;
                    background-color: #f9fafb;
                    color: #6b7280;
                    font-size: 12pt;
                    padding: 40px;
                }
            """)
            view_layout.addWidget(placeholder)

    def update_mesh_display(self):
        """Update mesh display according to current settings."""
        # Do nothing if the window is closing
        if hasattr(self, '_is_closing') and self._is_closing:
            return

        if not PYVISTA_AVAILABLE or self.mesh is None or not hasattr(self, 'plotter') or not self.plotter:
            return

        # Check that plotter is still valid
        if not is_plotter_valid(self.plotter):
            return

        try:
            # Suppress OpenGL warnings
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                # Remove previous mesh actor
                if self.mesh_actor is not None:
                    try:
                        self.plotter.remove_actor(self.mesh_actor)
                    except:
                        pass

                # Add new mesh with current shading options
                self.mesh_actor = self.plotter.add_mesh(
                    self.mesh,
                    color=self.current_color,
                    smooth_shading=self.smooth_shading,
                    show_edges=self.show_edges,
                    interpolate_before_map=True,
                    pbr=False,  # disable PBR for simpler control over color and lighting
                    ambient=self.ambient_light,
                    diffuse=self.diffuse_light,
                    specular=0.3,
                    specular_power=30.0
                )
                # Ensure no axes are shown
                remove_axes_from_plotter(self.plotter)

                # Render safely
                safe_render(self.plotter)
        except Exception as e:
            # Only log non‑OpenGL‑handle related errors
            if "wglMakeCurrent" not in str(e) and "invalid handle" not in str(e):
                print(f"Failed to update mesh display: {e}")
                import traceback
                traceback.print_exc()

    def on_color_changed(self):
        """Callback for changing surface color."""
        color = QColorDialog.getColor(
            QColor(int(self.current_color[0] * 255), int(self.current_color[1] * 255),
                   int(self.current_color[2] * 255)),
            self, "Select surface color"
        )
        if color.isValid():
            self.current_color = (color.red() / 255.0, color.green() / 255.0, color.blue() / 255.0)
            self.color_button.setStyleSheet(
                f"background-color: rgb({color.red()}, {color.green()}, {color.blue()}); "
                f"border: 2px solid #d1d5db; border-radius: 4px;"
            )
            self.update_mesh_display()
            # Sync to main window preview
            self.sync_to_main_window()

    def on_edges_changed(self, state):
        """Toggle wireframe display."""
        self.show_edges = (state == Qt.Checked)
        self.update_mesh_display()
        # Sync to main window preview
        self.sync_to_main_window()

    def on_smooth_shading_changed(self, state):
        """Toggle smooth shading."""
        self.smooth_shading = (state == Qt.Checked)
        self.update_mesh_display()
        # Sync to main window preview
        self.sync_to_main_window()

    def on_ambient_light_label_changed(self, value):
        """Update ambient light label (preview only)."""
        ambient_value = value / 100.0
        self.ambient_label.setText(f"{ambient_value:.2f}")

    def on_ambient_light_changed(self):
        """Apply ambient light intensity when slider is released."""
        value = self.ambient_slider.value()
        self.ambient_light = value / 100.0
        self.ambient_label.setText(f"{self.ambient_light:.2f}")
        self.update_mesh_display()
        # Sync to main window preview
        self.sync_to_main_window()

    def on_diffuse_light_label_changed(self, value):
        """Update diffuse light label (preview only)."""
        diffuse_value = value / 100.0
        self.diffuse_label.setText(f"{diffuse_value:.2f}")

    def on_diffuse_light_changed(self):
        """Apply diffuse light intensity when slider is released."""
        value = self.diffuse_slider.value()
        self.diffuse_light = value / 100.0
        self.diffuse_label.setText(f"{self.diffuse_light:.2f}")
        self.update_mesh_display()
        # Sync to main window preview
        self.sync_to_main_window()

    def sync_to_main_window(self):
        """Sync current 3D display settings back to the main window preview."""
        # Do nothing if the dialog is closing
        if hasattr(self, '_is_closing') and self._is_closing:
            return

        if self.parent_window and hasattr(self.parent_window, 'model_3d_color'):
            # 更新主窗口的设置
            self.parent_window.model_3d_color = self.current_color
            self.parent_window.model_3d_show_edges = self.show_edges
            self.parent_window.model_3d_smooth_shading = self.smooth_shading
            self.parent_window.model_3d_ambient_light = self.ambient_light
            self.parent_window.model_3d_diffuse_light = self.diffuse_light

            # 更新主窗口的UI控件（临时断开信号以避免循环更新）
            if hasattr(self.parent_window, 'model_3d_color_button'):
                color = QColor(int(self.current_color[0] * 255), int(self.current_color[1] * 255),
                               int(self.current_color[2] * 255))
                self.parent_window.model_3d_color_button.setStyleSheet(
                    f"background-color: rgb({color.red()}, {color.green()}, {color.blue()}); "
                    f"border: 2px solid #d1d5db; border-radius: 4px;"
                )
            if hasattr(self.parent_window, 'model_3d_edges_checkbox'):
                # 临时断开信号
                self.parent_window.model_3d_edges_checkbox.blockSignals(True)
                self.parent_window.model_3d_edges_checkbox.setChecked(self.show_edges)
                self.parent_window.model_3d_edges_checkbox.blockSignals(False)
            if hasattr(self.parent_window, 'model_3d_smooth_checkbox'):
                # 临时断开信号
                self.parent_window.model_3d_smooth_checkbox.blockSignals(True)
                self.parent_window.model_3d_smooth_checkbox.setChecked(self.smooth_shading)
                self.parent_window.model_3d_smooth_checkbox.blockSignals(False)
            if hasattr(self.parent_window, 'model_3d_ambient_slider'):
                # 临时断开信号
                self.parent_window.model_3d_ambient_slider.blockSignals(True)
                self.parent_window.model_3d_ambient_slider.setValue(int(self.ambient_light * 100))
                self.parent_window.model_3d_ambient_slider.blockSignals(False)
            if hasattr(self.parent_window, 'model_3d_ambient_label'):
                self.parent_window.model_3d_ambient_label.setText(f"{self.ambient_light:.2f}")
            if hasattr(self.parent_window, 'model_3d_diffuse_slider'):
                # 临时断开信号
                self.parent_window.model_3d_diffuse_slider.blockSignals(True)
                self.parent_window.model_3d_diffuse_slider.setValue(int(self.diffuse_light * 100))
                self.parent_window.model_3d_diffuse_slider.blockSignals(False)
            if hasattr(self.parent_window, 'model_3d_diffuse_label'):
                self.parent_window.model_3d_diffuse_label.setText(f"{self.diffuse_light:.2f}")

            # 更新主窗口的3D预览显示（使用延迟执行，避免在关闭过程中触发）
            if hasattr(self.parent_window, 'update_model_3d_display'):
                try:
                    # 检查主窗口的plotter是否仍然有效
                    if (hasattr(self.parent_window, 'model_3d_plotter') and
                            self.parent_window.model_3d_plotter is not None):
                        self.parent_window.update_model_3d_display()
                        # 确保主窗口也不显示坐标系
                        remove_axes_from_plotter(self.parent_window.model_3d_plotter)
                        # 延迟渲染，避免冲突
                        try:
                            from PySide6.QtCore import QTimer
                            if QT_VERSION == "PyQt5":
                                from PyQt5.QtCore import QTimer

                            def delayed_render():
                                try:
                                    if (hasattr(self.parent_window, 'model_3d_plotter') and
                                            self.parent_window.model_3d_plotter is not None):
                                        # 使用安全渲染，检查句柄有效性
                                        safe_render(self.parent_window.model_3d_plotter)
                                except:
                                    pass

                            QTimer.singleShot(50, delayed_render)  # 50ms后渲染
                        except:
                            # 如果QTimer不可用，直接渲染（但捕获异常）
                            try:
                                self.parent_window.model_3d_plotter.render()
                            except:
                                pass
                except Exception as e:
                    # 如果更新失败，不影响关闭流程
                    if "wglMakeCurrent" not in str(e) and "句柄无效" not in str(e):
                        print(f"同步到主窗口时出错（可忽略）: {e}")

            # 保存配置，记住颜色和其他设置
            if hasattr(self.parent_window, 'save_config'):
                try:
                    self.parent_window.save_config()
                except:
                    pass

    def on_save_3d_model(self):
        """保存编辑后的3D模型"""
        if not PYVISTA_AVAILABLE or self.mesh is None:
            QMessageBox.warning(self, "保存失败", "3D模型未加载或PyVista不可用")
            return

        if not NUMPY_AVAILABLE:
            QMessageBox.warning(self, "保存失败", "需要安装numpy库: pip install numpy")
            return

        try:
            # 创建带颜色的mesh副本
            mesh_to_save = self.mesh.copy()

            # 处理MultiBlock情况：如果是MultiBlock，需要合并为单个mesh
            def extract_single_mesh(mesh):
                """递归提取单个mesh，处理MultiBlock"""
                if not hasattr(mesh, 'n_blocks'):
                    # 已经是单个mesh
                    return mesh

                if mesh.n_blocks == 0:
                    raise ValueError("MultiBlock中没有块")

                # 收集所有有效的块
                blocks = []
                for i in range(mesh.n_blocks):
                    block = mesh[i]
                    if block is not None:
                        # 递归处理嵌套的MultiBlock
                        if hasattr(block, 'n_blocks') and block.n_blocks > 0:
                            block = extract_single_mesh(block)
                        # 确保是单个mesh且有顶点
                        if hasattr(block, 'n_points') and block.n_points > 0:
                            blocks.append(block)

                if not blocks:
                    raise ValueError("MultiBlock中没有有效的块")

                # 合并所有块
                if len(blocks) == 1:
                    return blocks[0]
                else:
                    # 使用merge合并，然后确保结果是单个mesh
                    merged = pv.merge(blocks)
                    # 如果merge后还是MultiBlock，使用extract_geometry
                    if hasattr(merged, 'n_blocks') and merged.n_blocks > 0:
                        return merged.extract_geometry()
                    return merged

            # 提取单个mesh
            mesh_to_save = extract_single_mesh(mesh_to_save)

            # 确保mesh_to_save是单个mesh对象
            if not hasattr(mesh_to_save, 'n_points'):
                raise ValueError(f"无法处理mesh类型: {type(mesh_to_save)}")

            # 将颜色应用到mesh的顶点
            # 创建一个颜色数组，每个顶点使用相同的颜色
            n_points = mesh_to_save.n_points
            colors = np.array([self.current_color] * n_points) * 255  # 转换为0-255范围
            colors = colors.astype(np.uint8)

            # 设置顶点颜色
            mesh_to_save['colors'] = colors

            # 或者使用RGB数组
            mesh_to_save['RGB'] = colors

            # 生成保存路径
            output_folder = get_output_folder()
            base_name = os.path.splitext(os.path.basename(self.original_glb_path))[0]
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_filename = f"{base_name}_colored_{timestamp}.glb"
            save_path = os.path.join(output_folder, save_filename)

            # 保存为GLB文件
            # 注意：PyVista的save方法可能不支持直接保存GLB，可能需要使用其他方法
            # 尝试保存为PLY或OBJ格式，或者使用trimesh等库
            try:
                # 尝试使用PyVista保存
                mesh_to_save.save(save_path)
                QMessageBox.information(
                    self,
                    "保存成功",
                    f"3D模型已保存到:\n{save_path}\n\n"
                    f"文件大小: {os.path.getsize(save_path) / (1024 * 1024):.2f} MB"
                )
                print(f"3D模型已保存到: {save_path}")
            except Exception as e:
                # 如果GLB保存失败，尝试保存为PLY格式
                print(f"保存GLB失败，尝试保存为PLY格式: {e}")
                save_path_ply = save_path.replace('.glb', '.ply')
                mesh_to_save.save(save_path_ply)
                QMessageBox.information(
                    self,
                    "保存成功",
                    f"3D模型已保存为PLY格式:\n{save_path_ply}\n\n"
                    f"文件大小: {os.path.getsize(save_path_ply) / (1024 * 1024):.2f} MB\n\n"
                    f"注意: 由于GLB格式限制，已保存为PLY格式"
                )
                print(f"3D模型已保存为PLY格式: {save_path_ply}")

        except Exception as e:
            error_msg = f"保存3D模型失败: {str(e)}"
            print(error_msg)
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self, "保存失败", error_msg)

    def closeEvent(self, event):
        """关闭窗口时清理PyVista资源并同步设置到主窗口"""
        # 标记窗口正在关闭，防止后续操作
        self._is_closing = True

        # 先停止所有渲染操作，避免冲突（使用try-except避免阻塞）
        if PYVISTA_AVAILABLE and hasattr(self, 'plotter') and self.plotter:
            try:
                # 停止渲染（快速检查，避免阻塞）
                if hasattr(self.plotter, 'renderer') and self.plotter.renderer:
                    try:
                        render_window = self.plotter.renderer.GetRenderWindow()
                        if render_window:
                            # 停止所有渲染操作
                            render_window.SetAbortRender(1)
                            # 释放OpenGL上下文，避免句柄无效错误
                            try:
                                render_window.Finalize()
                            except:
                                pass
                            # 禁用渲染窗口
                            try:
                                render_window.SetOffScreenRendering(1)  # 切换到离屏渲染，避免访问已失效的窗口句柄
                            except:
                                pass
                    except:
                        pass  # 如果窗口已关闭，忽略错误

                # 立即清理plotter引用，防止后续访问
                plotter_ref = self.plotter
                self.plotter = None

                # 尝试关闭plotter（在后台线程中，避免阻塞）
                def cleanup_plotter():
                    try:
                        if plotter_ref:
                            if hasattr(plotter_ref, 'clear'):
                                plotter_ref.clear()
                            if hasattr(plotter_ref, 'close'):
                                plotter_ref.close()
                    except:
                        pass

                # 使用QTimer在后台清理
                try:
                    from PySide6.QtCore import QTimer
                    if QT_VERSION == "PyQt5":
                        from PyQt5.QtCore import QTimer
                    QTimer.singleShot(0, cleanup_plotter)  # 立即在下一个事件循环中清理
                except:
                    pass
            except:
                pass

        # 延迟同步设置，避免在关闭过程中触发渲染
        if self.parent_window:
            try:
                # 使用QTimer延迟执行同步，确保窗口完全关闭后再更新主窗口
                from PySide6.QtCore import QTimer
                if QT_VERSION == "PyQt5":
                    from PyQt5.QtCore import QTimer

                def delayed_sync():
                    try:
                        self.sync_to_main_window()
                    except:
                        pass

                QTimer.singleShot(100, delayed_sync)  # 100ms后执行同步

                # 延迟清理资源，避免阻塞关闭过程
                def delayed_cleanup():
                    try:
                        if PYVISTA_AVAILABLE and hasattr(self, 'plotter') and self.plotter:
                            try:
                                # 清除所有actors
                                if hasattr(self.plotter, 'clear'):
                                    self.plotter.clear()
                                # 关闭plotter
                                if hasattr(self.plotter, 'close'):
                                    self.plotter.close()
                            except:
                                pass
                            finally:
                                self.plotter = None

                        # 清理mesh引用
                        self.mesh = None
                        self.mesh_actor = None
                    except:
                        pass

                QTimer.singleShot(200, delayed_cleanup)  # 200ms后清理资源
            except:
                # 如果延迟失败，直接同步（但不在关闭过程中渲染）
                try:
                    if self.parent_window and hasattr(self.parent_window, 'model_3d_color'):
                        # 只更新设置，不触发渲染
                        self.parent_window.model_3d_color = self.current_color
                        self.parent_window.model_3d_show_edges = self.show_edges
                        self.parent_window.model_3d_smooth_shading = self.smooth_shading
                        self.parent_window.model_3d_ambient_light = self.ambient_light
                        self.parent_window.model_3d_diffuse_light = self.diffuse_light
                except:
                    pass

        # 立即清理引用（先清理引用，避免后续操作访问已关闭的窗口）
        self.plotter = None
        self.mesh = None
        self.mesh_actor = None

        super().closeEvent(event)

    def eventFilter(self, obj, event):
        """事件过滤器，检测点击外部区域时关闭对话框"""
        if event.type() == QEvent.MouseButtonPress and self.isVisible():
            try:
                # 获取点击位置的全局坐标
                global_pos = event.globalPos()

                # 获取对话框的矩形区域（全局坐标）
                dialog_rect = self.frameGeometry()

                # 检查点击是否在对话框外部
                if not dialog_rect.contains(global_pos):
                    # 检查点击的对象是否是对话框或其子控件
                    clicked_widget = QApplication.widgetAt(global_pos)
                    if clicked_widget:
                        # 如果点击的是对话框或其子控件，不关闭
                        if clicked_widget == self or self.isAncestorOf(clicked_widget):
                            return super().eventFilter(obj, event)
                    # 点击在外部，关闭对话框
                    self.reject()
                    return True
            except Exception:
                # 如果出现异常，继续正常处理
                pass

        return super().eventFilter(obj, event)

    def closeEvent(self, event):
        """对话框关闭时移除事件过滤器"""
        if hasattr(self, 'app') and self.app:
            self.app.removeEventFilter(self)
        super().closeEvent(event)


class CanvasEditWindow(QDialog):
    """独立画布编辑窗口 - 增强版画图工具"""

    def __init__(self, main_canvas, parent=None):
        super().__init__(parent)
        self.main_canvas = main_canvas  # 主窗口的画布引用
        self.setWindowTitle("🎨 画布编辑 - 增强版")
        self.setModal(False)  # 非模态窗口，可以同时操作主窗口
        self.resize(900, 800)

        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(15)
        main_layout.setContentsMargins(20, 20, 20, 20)

        # 工具栏区域
        toolbar_layout = QHBoxLayout()
        toolbar_layout.setSpacing(10)

        # 颜色选择按钮
        self.color_btn = QPushButton("🎨 Color")
        self.color_btn.setStyleSheet("""
            QPushButton {
                background-color: #8b5cf6;
                padding: 8px 15px;
                font-weight: 600;
            }
            QPushButton:hover {
                background-color: #7c3aed;
            }
        """)
        self.color_btn.clicked.connect(self.on_choose_color)

        # 颜色预览标签
        self.color_preview = QLabel()
        self.color_preview.setFixedSize(40, 30)
        self.color_preview.setStyleSheet("""
            QLabel {
                border: 2px solid #d1d5db;
                border-radius: 5px;
                background-color: #212121;
            }
        """)
        self.update_color_preview(QColor(33, 33, 33))

        # 画笔粗细标签和滑块
        pen_width_label = QLabel("Pen Width:")
        pen_width_label.setStyleSheet("font-size: 10pt; font-weight: 500; color: #374151;")

        self.pen_width_slider = NoWheelSlider(Qt.Horizontal)
        self.pen_width_slider.setRange(1, 50)
        self.pen_width_slider.setValue(4)
        self.pen_width_slider.setSingleStep(1)
        self.pen_width_slider.setStyleSheet("""
            QSlider::groove:horizontal {
                border: 1px solid #d1d5db;
                height: 6px;
                background: #e5e7eb;
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: #6366f1;
                border: 2px solid #ffffff;
                width: 18px;
                height: 18px;
                border-radius: 9px;
                margin: -6px 0;
            }
            QSlider::handle:horizontal:hover {
                background: #4f46e5;
            }
        """)
        self.pen_width_slider.valueChanged.connect(self.on_pen_width_changed)

        self.pen_width_value_label = QLabel("4")
        self.pen_width_value_label.setStyleSheet("font-size: 10pt; font-weight: 600; color: #6366f1; min-width: 30px;")

        toolbar_layout.addWidget(self.color_btn)
        toolbar_layout.addWidget(self.color_preview)
        toolbar_layout.addWidget(pen_width_label)
        toolbar_layout.addWidget(self.pen_width_slider)
        toolbar_layout.addWidget(self.pen_width_value_label)
        toolbar_layout.addStretch()

        main_layout.addLayout(toolbar_layout)

        # 创建独立画布（更大的尺寸用于更强的画图）
        self.canvas = PaintCanvas(width=768, height=768, parent=self)
        # 复制主画布的内容
        if main_canvas and main_canvas.pixmap():
            # 如果主画布有内容，先缩放以适应新尺寸
            main_pixmap = main_canvas.pixmap()
            if not main_pixmap.isNull():
                scaled_pixmap = main_pixmap.scaled(
                    768, 768,
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation
                )
                # 创建新画布并居中绘制
                new_pixmap = QPixmap(768, 768)
                new_pixmap.fill(QColor(250, 250, 250))
                painter = QPainter(new_pixmap)
                x_offset = (768 - scaled_pixmap.width()) // 2
                y_offset = (768 - scaled_pixmap.height()) // 2
                painter.drawPixmap(x_offset, y_offset, scaled_pixmap)
                painter.end()
                self.canvas.setPixmap(new_pixmap)
            # 复制历史记录
            if hasattr(main_canvas, 'history') and main_canvas.history:
                self.canvas.history = [QPixmap(pix) for pix in main_canvas.history]
                self.canvas.history_index = main_canvas.history_index if hasattr(main_canvas, 'history_index') else len(
                    self.canvas.history) - 1
            else:
                self.canvas.save_state()

        # 同步画笔颜色和粗细
        if hasattr(main_canvas, 'pen_color'):
            self.canvas.pen_color = QColor(main_canvas.pen_color)
            self.update_color_preview(self.canvas.pen_color)
        if hasattr(main_canvas, 'pen_width'):
            self.canvas.pen_width = main_canvas.pen_width
            self.pen_width_slider.setValue(self.canvas.pen_width)
            self.pen_width_value_label.setText(str(self.canvas.pen_width))

        main_layout.addWidget(self.canvas, stretch=1)

        # 按钮区域
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(10)

        # 撤销/重做按钮
        self.undo_btn = QPushButton("↶ Undo")
        self.undo_btn.setStyleSheet("""
            QPushButton {
                background-color: #6b7280;
                font-size: 12pt;
                padding: 12px 24px;
                font-weight: 600;
            }
            QPushButton:hover {
                background-color: #4b5563;
            }
        """)
        self.undo_btn.clicked.connect(self.canvas.undo)

        self.redo_btn = QPushButton("↷ Redo")
        self.redo_btn.setStyleSheet("""
            QPushButton {
                background-color: #6b7280;
                font-size: 12pt;
                padding: 12px 24px;
                font-weight: 600;
            }
            QPushButton:hover {
                background-color: #4b5563;
            }
        """)
        self.redo_btn.clicked.connect(self.canvas.redo)

        self.clear_btn = QPushButton("🗑️ Clear")
        self.clear_btn.setStyleSheet("""
            QPushButton {
                background-color: #ef4444;
                font-size: 12pt;
                padding: 12px 24px;
                font-weight: 600;
            }
            QPushButton:hover {
                background-color: #dc2626;
            }
        """)
        self.clear_btn.clicked.connect(self.canvas.clear_canvas)

        self.import_btn = QPushButton("📂 Import")
        self.import_btn.setStyleSheet("""
            QPushButton {
                background-color: #6366f1;
                font-size: 12pt;
                padding: 12px 24px;
                font-weight: 600;
            }
            QPushButton:hover {
                background-color: #4f46e5;
            }
        """)
        self.import_btn.clicked.connect(self.on_import_image)

        self.save_btn = QPushButton("💾 Save Sketch")
        self.save_btn.setStyleSheet("""
            QPushButton {
                background-color: #10b981;
                font-size: 12pt;
                padding: 12px 24px;
                font-weight: 600;
            }
            QPushButton:hover {
                background-color: #059669;
            }
        """)
        self.save_btn.clicked.connect(self.on_save_sketch)

        btn_layout.addWidget(self.undo_btn)
        btn_layout.addWidget(self.redo_btn)
        btn_layout.addWidget(self.clear_btn)
        btn_layout.addWidget(self.import_btn)
        btn_layout.addWidget(self.save_btn)
        btn_layout.addStretch()

        self.apply_btn = QPushButton("✅ Apply")
        self.apply_btn.setStyleSheet("""
            QPushButton {
                background-color: #6366f1;
                font-size: 12pt;
                padding: 12px 24px;
                font-weight: 600;
            }
            QPushButton:hover {
                background-color: #4f46e5;
            }
        """)
        self.apply_btn.clicked.connect(self.on_apply_and_close)
        btn_layout.addWidget(self.apply_btn)

        main_layout.addLayout(btn_layout)

        # 设置窗口样式
        self.setStyleSheet("""
            QDialog {
                background-color: #f3f4f6;
            }
        """)

    def update_color_preview(self, color):
        """更新颜色预览"""
        self.color_preview.setStyleSheet(f"""
            QLabel {{
                border: 2px solid #d1d5db;
                border-radius: 5px;
                background-color: rgb({color.red()}, {color.green()}, {color.blue()});
            }}
        """)

    def on_choose_color(self):
        """选择画笔颜色"""
        color = QColorDialog.getColor(self.canvas.pen_color, self, "选择画笔颜色")
        if color.isValid():
            self.canvas.pen_color = color
            self.update_color_preview(color)

    def on_pen_width_changed(self, value):
        """画笔粗细改变"""
        self.canvas.pen_width = value
        self.pen_width_value_label.setText(str(value))

    def on_import_image(self):
        """导入图片到画布"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "选择图片",
            "",
            "Image Files (*.png *.jpg *.jpeg *.bmp);;All Files (*.*)"
        )
        if file_path:
            self.canvas.load_image(file_path)

    def on_save_sketch(self):
        """保存手绘图"""
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "保存手绘图",
            "",
            "PNG Image (*.png);;JPEG Image (*.jpg);;All Files (*.*)"
        )
        if file_path:
            self.canvas.save_canvas(file_path)
            QMessageBox.information(self, "Save Successful", f"Sketch saved to:\n{file_path}")

    def on_apply_and_close(self):
        """应用更改到主画布并关闭窗口"""
        if self.main_canvas:
            # 将768x768的画布缩放回主画布的尺寸
            canvas_pixmap = self.canvas.pixmap()
            if not canvas_pixmap.isNull():
                # 获取主画布尺寸
                main_width = self.main_canvas.canvas_width
                main_height = self.main_canvas.canvas_height

                # 缩放画布内容以适应主画布
                scaled_pixmap = canvas_pixmap.scaled(
                    main_width, main_height,
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation
                )

                # 创建新画布并居中绘制
                new_pixmap = QPixmap(main_width, main_height)
                new_pixmap.fill(QColor(250, 250, 250))
                painter = QPainter(new_pixmap)
                x_offset = (main_width - scaled_pixmap.width()) // 2
                y_offset = (main_height - scaled_pixmap.height()) // 2
                painter.drawPixmap(x_offset, y_offset, scaled_pixmap)
                painter.end()

                self.main_canvas.setPixmap(new_pixmap)
                # 复制历史记录（只保留当前状态）
                if hasattr(self.main_canvas, 'history'):
                    self.main_canvas.history = [QPixmap(new_pixmap)]
                    self.main_canvas.history_index = 0
                # 同步画笔设置
                if hasattr(self.main_canvas, 'pen_color'):
                    self.main_canvas.pen_color = QColor(self.canvas.pen_color)
                if hasattr(self.main_canvas, 'pen_width'):
                    self.main_canvas.pen_width = self.canvas.pen_width
                self.main_canvas.update()
        self.accept()  # 使用accept()关闭对话框


class PaintCanvas(QLabel):
    """现代化手绘画布，支持撤销/重做"""

    def __init__(self, width=512, height=512, parent=None):
        super().__init__(parent)
        self.canvas_width = width
        self.canvas_height = height

        pixmap = QPixmap(self.canvas_width, self.canvas_height)
        pixmap.fill(QColor(250, 250, 250))  # 更柔和的背景色
        self.setPixmap(pixmap)

        self.last_point = QPoint()
        self.drawing = False

        self.pen_color = QColor(33, 33, 33)  # 深灰色，更现代
        self.pen_width = 4

        # 撤销/重做历史记录
        self.history = []  # 历史状态列表
        self.history_index = -1  # 当前历史索引
        self.max_history = 50  # 最大历史记录数

        # 保存初始状态
        self.save_state()

        # 现代化样式
        self.setStyleSheet("""
            QLabel {
                border: 2px solid #e0e0e0;
                border-radius: 12px;
                background-color: #fafafa;
            }
        """)

        # 设置焦点策略，以便接收键盘事件
        self.setFocusPolicy(Qt.StrongFocus)

        # 启用拖放功能
        self.setAcceptDrops(True)

    def save_state(self):
        """保存当前画布状态到历史记录"""
        # 获取当前pixmap，确保它存在且有效
        current_pixmap = self.pixmap()
        if current_pixmap.isNull():
            # 如果pixmap无效，创建一个新的
            current_pixmap = QPixmap(self.canvas_width, self.canvas_height)
            current_pixmap.fill(QColor(250, 250, 250))
            self.setPixmap(current_pixmap)

        # 创建当前画布的副本（深拷贝）
        pixmap_copy = QPixmap(current_pixmap)

        # 如果当前不在历史末尾，删除后面的历史
        if self.history_index < len(self.history) - 1:
            self.history = self.history[:self.history_index + 1]

        # 添加新状态
        self.history.append(pixmap_copy)
        self.history_index = len(self.history) - 1

        # 限制历史记录数量
        if len(self.history) > self.max_history:
            self.history.pop(0)
            self.history_index -= 1

    def undo(self):
        """撤销上一步操作"""
        if self.history_index > 0:
            self.history_index -= 1
            self.setPixmap(QPixmap(self.history[self.history_index]))
            self.update()
            return True
        return False

    def redo(self):
        """重做下一步操作"""
        if self.history_index < len(self.history) - 1:
            self.history_index += 1
            self.setPixmap(QPixmap(self.history[self.history_index]))
            self.update()
            return True
        return False

    def increase_pen_width(self):
        """增加画笔宽度"""
        self.pen_width = min(self.pen_width + 1, 20)

    def decrease_pen_width(self):
        """减少画笔宽度"""
        self.pen_width = max(self.pen_width - 1, 1)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            # 开始绘制前保存状态（只保存一次）
            if not self.drawing:
                self.save_state()
            self.drawing = True
            self.last_point = event.pos()
            self.setFocus()  # 获取焦点以接收键盘事件

    def mouseMoveEvent(self, event):
        if self.drawing and (event.buttons() & Qt.LeftButton):
            # 获取当前pixmap，确保它存在且有效
            current_pixmap = self.pixmap()
            if current_pixmap.isNull():
                # 如果pixmap无效，创建一个新的
                current_pixmap = QPixmap(self.canvas_width, self.canvas_height)
                current_pixmap.fill(QColor(250, 250, 250))
                self.setPixmap(current_pixmap)

            # 创建pixmap的副本进行绘制，避免在绘制过程中修改原始pixmap
            pixmap_copy = QPixmap(current_pixmap)
            painter = QPainter(pixmap_copy)
            if not painter.isActive():
                return

            pen = QPen(self.pen_color, self.pen_width, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
            painter.setPen(pen)
            painter.drawLine(self.last_point, event.pos())
            painter.end()

            # 绘制完成后，更新pixmap
            self.setPixmap(pixmap_copy)
            self.last_point = event.pos()
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drawing = False
            # 绘制结束后保存状态（完成一次绘制操作）
            self.save_state()

    def keyPressEvent(self, event):
        """处理键盘快捷键"""
        # Ctrl+Z: 撤销
        if event.modifiers() == Qt.ControlModifier and event.key() == Qt.Key_Z:
            self.undo()
            event.accept()
            return

        # Ctrl+Y 或 Ctrl+Shift+Z: 重做
        if (event.modifiers() == Qt.ControlModifier and event.key() == Qt.Key_Y) or \
                (event.modifiers() == (Qt.ControlModifier | Qt.ShiftModifier) and event.key() == Qt.Key_Z):
            self.redo()
            event.accept()
            return

        # Delete 或 Backspace: 清空画布
        if event.key() in (Qt.Key_Delete, Qt.Key_Backspace):
            self.clear_canvas()
            event.accept()
            return

        # + 或 =: 增加画笔宽度
        if event.key() in (Qt.Key_Plus, Qt.Key_Equal):
            self.increase_pen_width()
            event.accept()
            return

        # - 或 _: 减少画笔宽度
        if event.key() in (Qt.Key_Minus, Qt.Key_Underscore):
            self.decrease_pen_width()
            event.accept()
            return

        super().keyPressEvent(event)

    def clear_canvas(self):
        """清空画布"""
        pixmap = QPixmap(self.canvas_width, self.canvas_height)
        pixmap.fill(QColor(250, 250, 250))
        self.setPixmap(pixmap)
        self.update()
        # 清空后保存状态
        self.save_state()

    def has_content(self):
        """检查画布是否有内容（不是空的）"""
        if not self.pixmap():
            return False

        pixmap = self.pixmap()
        if pixmap.isNull():
            return False

        # 创建空画布用于比较
        empty_pixmap = QPixmap(self.canvas_width, self.canvas_height)
        empty_pixmap.fill(QColor(250, 250, 250))

        # 比较两个pixmap是否相同
        pixmap_image = pixmap.toImage()
        empty_image = empty_pixmap.toImage()
        return pixmap_image != empty_image

    def save_canvas(self, file_path):
        """保存当前画布到文件（可用于后续扩展到控制图像等）"""
        self.pixmap().save(file_path)

    def load_image(self, image_path):
        """加载图片到画布上，自动缩放以适应画布大小"""
        if not os.path.exists(image_path):
            return False

        try:
            # 加载图片
            source_pixmap = QPixmap(image_path)
            if source_pixmap.isNull():
                return False

            # 保存状态（导入图片前）
            self.save_state()

            # 缩放图片以适应画布大小，保持宽高比
            scaled_pixmap = source_pixmap.scaled(
                self.canvas_width,
                self.canvas_height,
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )

            # 创建新的画布，填充背景色
            new_pixmap = QPixmap(self.canvas_width, self.canvas_height)
            new_pixmap.fill(QColor(250, 250, 250))

            # 将缩放后的图片居中绘制到画布上
            painter = QPainter(new_pixmap)
            x_offset = (self.canvas_width - scaled_pixmap.width()) // 2
            y_offset = (self.canvas_height - scaled_pixmap.height()) // 2
            painter.drawPixmap(x_offset, y_offset, scaled_pixmap)
            painter.end()

            # 设置新的画布
            self.setPixmap(new_pixmap)
            self.update()

            return True
        except Exception as e:
            print(f"加载图片失败: {e}")
            return False

    def dragEnterEvent(self, event):
        """拖拽进入事件"""
        # 检查拖拽的内容是否包含文件URL
        if event.mimeData().hasUrls():
            # 检查是否是图像文件
            urls = event.mimeData().urls()
            if urls:
                file_path = urls[0].toLocalFile()
                # 检查文件扩展名
                if file_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.webp')):
                    event.acceptProposedAction()
                    # 改变样式提示可以拖放
                    self.setStyleSheet("""
                        QLabel {
                            border: 3px dashed #6366f1;
                            border-radius: 12px;
                            background-color: #eef2ff;
                        }
                    """)
                    return
        event.ignore()

    def dragLeaveEvent(self, event):
        """拖拽离开事件"""
        # 恢复原始样式
        self.setStyleSheet("""
            QLabel {
                border: 2px solid #e0e0e0;
                border-radius: 12px;
                background-color: #fafafa;
            }
        """)
        super().dragLeaveEvent(event)

    def dropEvent(self, event):
        """拖放事件"""
        if event.mimeData().hasUrls():
            urls = event.mimeData().urls()
            if urls:
                file_path = urls[0].toLocalFile()
                # 检查文件扩展名
                if file_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.webp')):
                    # 加载图像
                    if self.load_image(file_path):
                        # 恢复原始样式
                        self.setStyleSheet("""
                            QLabel {
                                border: 2px solid #e0e0e0;
                                border-radius: 12px;
                                background-color: #fafafa;
                            }
                        """)
                        event.acceptProposedAction()
                        return
        # 恢复原始样式
        self.setStyleSheet("""
            QLabel {
                border: 2px solid #e0e0e0;
                border-radius: 12px;
                background-color: #fafafa;
            }
        """)
        event.ignore()


class SelectableThumbnail(QLabel):
    """现代化可选择的缩略图标签，点击时显示选中状态"""

    clicked = None  # 点击回调函数

    def __init__(self, image_path, parent=None):
        super().__init__(parent)
        self.image_path = image_path
        self.is_selected = False
        self.setAlignment(Qt.AlignCenter)
        self.setCursor(Qt.PointingHandCursor)
        self.deselect()  # 初始化样式

        # 加载并显示缩略图
        pix = QPixmap(image_path)
        if not pix.isNull():
            self.setPixmap(pix.scaled(160, 160, Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def mousePressEvent(self, event):
        """点击时触发选择事件"""
        if event.button() == Qt.LeftButton:
            if self.clicked:
                self.clicked(self.image_path, self)
            else:
                self.select()

    def select(self):
        """设置为选中状态"""
        self.is_selected = True
        self.setStyleSheet("""
            QLabel {
                border: 3px solid #6366f1;
                border-radius: 8px;
                padding: 2px;
                background-color: #eef2ff;
            }
        """)

    def deselect(self):
        """取消选中状态"""
        self.is_selected = False
        self.setStyleSheet("""
            QLabel {
                border: 2px solid #e5e7eb;
                border-radius: 8px;
                padding: 2px;
                background-color: #ffffff;
            }
            QLabel:hover {
                border: 2px solid #d1d5db;
                background-color: #f9fafb;
            }
        """)


class ClickableImageLabel(QLabel):
    """可点击的图像标签，点击时放大显示"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.image_path = None
        self.setCursor(Qt.PointingHandCursor)

    def set_image_path(self, image_path):
        """设置图像路径"""
        self.image_path = image_path

    def mousePressEvent(self, event):
        """点击时打开放大对话框"""
        if event.button() == Qt.LeftButton and self.image_path and os.path.exists(self.image_path):
            # 检查是否有pixmap（有图像显示）
            if self.pixmap() and not self.pixmap().isNull():
                dialog = ImageZoomDialog(self.image_path, self)
                dialog.show()  # 使用show()而不是exec_()，以便可以检测外部点击
                dialog.raise_()  # 将对话框置于最前
                dialog.activateWindow()  # 激活窗口


class EditPromptTextEdit(QTextEdit):
    """自定义编辑Prompt输入框，支持Enter键触发编辑"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.main_window = None  # 将在MainWindow中设置

    def set_main_window(self, main_window):
        """设置主窗口引用"""
        self.main_window = main_window

    def keyPressEvent(self, event):
        """处理键盘事件"""
        # 如果按下Enter键（不是Shift+Enter或Ctrl+Enter）
        if event.key() == Qt.Key_Return or event.key() == Qt.Key_Enter:
            # 检查是否按下了Ctrl或Shift（Ctrl+Enter或Shift+Enter用于换行）
            modifiers = event.modifiers()
            if modifiers == Qt.NoModifier or modifiers == Qt.KeypadModifier:
                # 纯Enter键，执行编辑
                if self.main_window:
                    # 获取当前文本
                    prompt = self.toPlainText().strip()
                    if prompt and self.main_window.selected_image_path:
                        # 执行编辑
                        self.main_window.on_edit_image()
                        # 如果启用了循环修改，编辑成功后会清空输入框
                        return  # 不处理Enter键的默认行为（插入换行）
                    elif not prompt:
                        # Prompt为空，不占用生成结果区域，直接返回
                        # 提示信息可以通过其他方式显示（如状态栏或消息框）
                        return
                    elif not self.main_window.selected_image_path:
                        # 没有选中图像，不占用生成结果区域，直接返回
                        # 提示信息可以通过其他方式显示（如状态栏或消息框）
                        return

        # 其他情况使用默认行为（包括Shift+Enter换行）
        super().keyPressEvent(event)


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Draft3D")
        self.resize(1200, 800)

        # 保存当前选中的图像路径和缩略图列表
        self.selected_image_path = None
        self.original_image_path = None  # 保存原始选中的图像路径（用于循环修改）
        self.thumbnail_widgets = []

        # 预设Prompt存储
        self.presets_file = os.path.join(os.getcwd(), "prompt_presets.json")
        self.prompt_presets = {}  # {name: prompt_text}
        self.load_presets()

        # 配置文件路径
        self.config_file = os.path.join(os.getcwd(), "gui_config.json")

        # 应用现代化样式
        self._apply_modern_style()
        self._init_ui()

        # 先连接信号（但加载配置时会暂时断开）
        self._connect_config_signals()

        # 加载保存的配置（加载时会暂时断开信号避免触发保存）
        self.load_config()

    def _apply_modern_style(self):
        """应用现代化样式表"""
        self.setStyleSheet("""
            QWidget {
                font-family: 'Segoe UI', 'Microsoft YaHei', sans-serif;
                font-size: 10pt;
                color: #1f2937;
            }

            QGroupBox {
                font-weight: 600;
                font-size: 11pt;
                border: 2px solid #e5e7eb;
                border-radius: 10px;
                margin-top: 12px;
                padding-top: 15px;
                background-color: #ffffff;
            }

            QGroupBox::title {
                subcontrol-origin: margin;
                left: 15px;
                padding: 0 8px;
                color: #4b5563;
            }

            QPushButton {
                background-color: #6366f1;
                color: white;
                border: none;
                border-radius: 8px;
                padding: 10px 20px;
                font-weight: 600;
                font-size: 10pt;
                min-height: 20px;
            }

            QPushButton:hover {
                background-color: #4f46e5;
            }

            QPushButton:pressed {
                background-color: #4338ca;
            }

            QPushButton:disabled {
                background-color: #9ca3af;
                color: #d1d5db;
            }

            QTextEdit {
                border: 2px solid #e5e7eb;
                border-radius: 8px;
                padding: 10px;
                background-color: #ffffff;
                selection-background-color: #eef2ff;
            }

            QTextEdit:focus {
                border: 2px solid #6366f1;
            }

            QSpinBox, QDoubleSpinBox {
                border: 2px solid #e5e7eb;
                border-radius: 6px;
                padding: 4px;
                background-color: #ffffff;
                min-width: 60px;
                font-size: 9pt;
                max-height: 28px;
            }

            QSpinBox:focus, QDoubleSpinBox:focus {
                border: 2px solid #6366f1;
            }

            QSpinBox::up-button, QDoubleSpinBox::up-button {
                border-left: 1px solid #d1d5db;
                border-top-right-radius: 6px;
                background-color: #e5e7eb;
                width: 16px;
            }

            QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover {
                background-color: #d1d5db;
            }

            QSpinBox::up-button, QDoubleSpinBox::up-button {
                border-left: 1px solid #d1d5db;
                border-top-right-radius: 6px;
                background-color: #e5e7eb;
                width: 16px;
            }

            QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover {
                background-color: #d1d5db;
            }

            QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {
                image: none;
                border: none;
                width: 0px;
                height: 0px;
            }

            QSpinBox::down-button, QDoubleSpinBox::down-button {
                border-left: 1px solid #d1d5db;
                border-bottom-right-radius: 6px;
                background-color: #e5e7eb;
                width: 16px;
            }

            QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover {
                background-color: #d1d5db;
            }

            QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {
                image: none;
                border: none;
                width: 0px;
                height: 0px;
            }

            QLabel {
                color: #374151;
            }
        """)

    def _init_ui(self):
        main_layout = QHBoxLayout(self)
        main_layout.setSpacing(20)
        main_layout.setContentsMargins(20, 20, 20, 20)

        # 设置窗口背景
        self.setStyleSheet(self.styleSheet() + """
            QWidget#main_window {
                background-color: #f3f4f6;
            }
        """)
        self.setObjectName("main_window")

        # 左侧：画布 + 按钮（支持滚动）
        left_content_widget = QWidget()
        left_content_widget.setStyleSheet("""
            QWidget {
                background-color: #ffffff;
                border-radius: 12px;
                padding: 15px;
            }
        """)
        left_layout = QVBoxLayout(left_content_widget)
        left_layout.setSpacing(15)
        left_layout.setContentsMargins(15, 15, 15, 15)

        # 创建滚动区域用于左边栏
        left_scroll = QScrollArea()
        left_scroll.setWidget(left_content_widget)
        left_scroll.setWidgetResizable(True)
        left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        left_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        left_scroll.setFrameShape(QScrollArea.NoFrame)
        left_scroll.setStyleSheet("""
            QScrollArea {
                border: none;
                background-color: transparent;
            }
            QScrollBar:vertical {
                border: none;
                background-color: #e5e7eb;
                width: 10px;
                border-radius: 5px;
                margin: 2px;
            }
            QScrollBar::handle:vertical {
                background-color: #9ca3af;
                border-radius: 5px;
                min-height: 20px;
                margin: 1px;
            }
            QScrollBar::handle:vertical:hover {
                background-color: #6b7280;
            }
            QScrollBar::handle:vertical:pressed {
                background-color: #4b5563;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0px;
            }
            QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {
                background: transparent;
            }
        """)

        # 创建外层widget用于设置样式
        left_widget = QWidget()
        left_widget.setStyleSheet("""
            QWidget {
                background-color: #ffffff;
                border-radius: 12px;
            }
        """)
        left_widget_layout = QVBoxLayout(left_widget)
        left_widget_layout.setContentsMargins(0, 0, 0, 0)
        left_widget_layout.addWidget(left_scroll)

        self.canvas = PaintCanvas(width=384, height=300)
        left_layout.addWidget(self.canvas)

        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(10)
        self.clear_btn = QPushButton("🗑️ Clear")
        self.save_sketch_btn = QPushButton("💾 Save")
        self.clear_btn.setStyleSheet("""
            QPushButton {
                background-color: #ef4444;
                font-size: 12pt;
                padding: 12px 24px;
                font-weight: 600;
            }
            QPushButton:hover {
                background-color: #dc2626;
            }
        """)
        self.save_sketch_btn.setStyleSheet("""
            QPushButton {
                background-color: #10b981;
                font-size: 12pt;
                padding: 12px 24px;
                font-weight: 600;
            }
            QPushButton:hover {
                background-color: #059669;
            }
        """)
        btn_layout.addWidget(self.clear_btn)
        btn_layout.addWidget(self.save_sketch_btn)

        self.open_canvas_window_btn = QPushButton("🪟 Edit Window")
        self.open_canvas_window_btn.setStyleSheet("""
            QPushButton {
                background-color: #8b5cf6;
                font-size: 12pt;
                padding: 12px 24px;
                font-weight: 600;
            }
            QPushButton:hover {
                background-color: #7c3aed;
            }
        """)
        self.open_canvas_window_btn.setToolTip("Open canvas in a separate window for editing")
        btn_layout.addWidget(self.open_canvas_window_btn)

        left_layout.addLayout(btn_layout)

        # 提示词输入区域
        prompt_label = QLabel("📝 Prompt")
        prompt_label.setStyleSheet("font-size: 11pt; font-weight: 600; color: #1f2937; padding-bottom: 5px;")
        left_layout.addWidget(prompt_label)

        self.prompt_edit = QTextEdit()
        # 设置默认提示词 - 耳机动画图像
        self.prompt_edit.setPlainText(
            "headphones, animated style, dynamic pose, white background, no lighting, high quality")
        self.prompt_edit.setMinimumHeight(60)
        self.prompt_edit.setMaximumHeight(120)
        self.prompt_edit.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        # 设置为可编辑
        self.prompt_edit.setReadOnly(False)
        self.prompt_edit.setStyleSheet("""
            QTextEdit {
                background-color: #ffffff;
                color: #374151;
                border: 1px solid #d1d5db;
                border-radius: 4px;
            }
        """)
        left_layout.addWidget(self.prompt_edit)

        # 生成参数（移到左边栏）
        param_group = QGroupBox("⚙️ Generation Parameters")
        param_group.setStyleSheet("""
            QGroupBox {
                font-weight: 600;
                font-size: 10pt;
                border: 2px solid #e5e7eb;
                border-radius: 8px;
                margin-top: 8px;
                padding-top: 10px;
                background-color: #ffffff;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 6px;
                color: #4b5563;
            }
        """)
        param_layout = QHBoxLayout()
        param_layout.setSpacing(6)
        param_layout.setContentsMargins(8, 5, 8, 8)

        self.seed_spin = CustomSpinBox()
        self.seed_spin.setRange(0, 2 ** 31 - 1)
        self.seed_spin.setValue(42)
        self.seed_spin.setPrefix("Seed: ")

        self.steps_spin = CustomSpinBox()
        self.steps_spin.setRange(1, 100)
        self.steps_spin.setValue(8)
        self.steps_spin.setPrefix("Steps: ")

        self.cfg_spin = CustomDoubleSpinBox()
        self.cfg_spin.setRange(0.1, 20.0)
        self.cfg_spin.setSingleStep(0.1)
        self.cfg_spin.setValue(2.0)
        self.cfg_spin.setPrefix("CFG: ")

        self.num_images_spin = CustomSpinBox()
        self.num_images_spin.setRange(1, 10)
        self.num_images_spin.setValue(1)
        self.num_images_spin.setPrefix("Quantity: ")

        param_layout.addWidget(self.seed_spin)
        param_layout.addWidget(self.steps_spin)
        param_layout.addWidget(self.cfg_spin)
        param_layout.addWidget(self.num_images_spin)

        param_group.setLayout(param_layout)
        left_layout.addWidget(param_group)

        # 图像分辨率设置（合并成一个滚动条，无标题）
        resolution_group = QWidget()
        resolution_group.setStyleSheet("""
            QWidget {
                background-color: transparent;
            }
        """)
        resolution_layout = QVBoxLayout()
        resolution_layout.setSpacing(5)  # 减少间距，更紧凑
        resolution_layout.setContentsMargins(8, 5, 8, 5)  # 减少边距

        # 分辨率标签
        resolution_label = QLabel("Resolution:")
        resolution_label.setStyleSheet("font-size: 10pt; font-weight: 500; color: #374151;")
        resolution_layout.addWidget(resolution_label)

        # 滚动条、数字和复选框放在同一行
        slider_value_checkbox_layout = QHBoxLayout()
        slider_value_checkbox_layout.setSpacing(8)

        # 统一的分辨率滑块
        self.image_resolution_slider = NoWheelSlider(Qt.Horizontal)
        self.image_resolution_slider.setRange(256, 2048)
        self.image_resolution_slider.setValue(512)
        self.image_resolution_slider.setSingleStep(64)
        self.image_resolution_slider.setPageStep(128)
        self.image_resolution_slider.setStyleSheet("""
            QSlider::groove:horizontal {
                border: 1px solid #d1d5db;
                height: 4px;
                background: #e5e7eb;
                border-radius: 2px;
            }
            QSlider::handle:horizontal {
                background: #6366f1;
                border: 2px solid #ffffff;
                width: 16px;
                height: 16px;
                border-radius: 8px;
                margin: -6px 0;
            }
            QSlider::handle:horizontal:hover {
                background: #4f46e5;
            }
            QSlider::handle:horizontal:pressed {
                background: #4338ca;
            }
        """)
        slider_value_checkbox_layout.addWidget(self.image_resolution_slider)

        # 分辨率数字显示（放在滚动条右侧）
        self.image_resolution_label = QLabel("512x512 px")
        self.image_resolution_label.setStyleSheet("font-size: 10pt; font-weight: 600; color: #6366f1; min-width: 80px;")
        slider_value_checkbox_layout.addWidget(self.image_resolution_label)

        # 添加锁定宽高比的复选框（默认锁定，保持正方形）
        self.lock_aspect_ratio = QCheckBox("Lock Aspect Ratio")
        self.lock_aspect_ratio.setChecked(True)
        self.lock_aspect_ratio.setEnabled(False)  # 禁用，因为现在总是正方形
        self.lock_aspect_ratio.setStyleSheet("""
            QCheckBox {
                font-size: 9pt;
                color: #6b7280;
            }
        """)
        slider_value_checkbox_layout.addWidget(self.lock_aspect_ratio)
        resolution_layout.addLayout(slider_value_checkbox_layout)

        # 为了兼容性，保留宽度和高度的内部变量（但不显示）
        self.image_width_slider = self.image_resolution_slider  # 使用同一个滑块
        self.image_height_slider = self.image_resolution_slider  # 使用同一个滑块
        self.image_width_label = self.image_resolution_label  # 兼容性引用
        self.image_height_label = self.image_resolution_label  # 兼容性引用

        resolution_group.setLayout(resolution_layout)
        left_layout.addWidget(resolution_group)

        # 连接滑块值改变事件
        self.image_resolution_slider.valueChanged.connect(self.on_resolution_slider_changed)

        # 初始化显示
        self.on_resolution_slider_changed(self.image_resolution_slider.value())

        # 生成按钮
        self.generate_btn = QPushButton("✨ Generate")
        self.generate_btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.generate_btn.setStyleSheet("""
            QPushButton {
                background-color: #6366f1;
                font-size: 12pt;
                padding: 12px 24px;
                font-weight: 600;
            }
            QPushButton:hover {
                background-color: #4f46e5;
            }
        """)
        left_layout.addWidget(self.generate_btn)

        # 中间列：图像画廊 + 生成结果 + 编辑功能
        middle_widget = QWidget()
        middle_widget.setStyleSheet("""
            QWidget {
                background-color: #ffffff;
                border-radius: 12px;
                padding: 15px;
            }
        """)
        middle_layout = QVBoxLayout(middle_widget)
        middle_layout.setSpacing(15)
        middle_layout.setContentsMargins(15, 15, 15, 15)

        # 图像画廊（支持点击选择）
        gallery_title = QLabel("📚 Image Gallery")
        gallery_title.setStyleSheet("""
            QLabel {
                font-size: 11pt;
                font-weight: 600;
                color: #4b5563;
                padding-top: 10px;
                padding-bottom: 5px;
            }
        """)
        middle_layout.addWidget(gallery_title)

        self.gallery_widget = QWidget()
        self.gallery_widget.setMinimumHeight(180)  # 为缩略图预留空间（160px图片 + 边框和padding）
        self.gallery_layout = QHBoxLayout(self.gallery_widget)
        self.gallery_layout.setSpacing(10)
        self.gallery_layout.setContentsMargins(0, 0, 0, 0)

        # 添加预设的占位符框（4个）- 使用耳机相关图标
        placeholder_icons = ["🎧", "🎧", "🎧", "🎧"]
        for i in range(4):
            placeholder = QLabel(placeholder_icons[i])
            placeholder.setFixedSize(160, 160)
            placeholder.setAlignment(Qt.AlignCenter)
            placeholder.setStyleSheet("""
                QLabel {
                    border: 2px dashed #d1d5db;
                    border-radius: 8px;
                    background-color: #f9fafb;
                    font-size: 48pt;
                    color: #9ca3af;
                }
            """)
            self.gallery_layout.addWidget(placeholder)

        middle_layout.addWidget(self.gallery_widget)

        # 生成结果和编辑结果并排显示
        result_container = QWidget()
        result_container_layout = QHBoxLayout(result_container)
        result_container_layout.setSpacing(15)
        result_container_layout.setContentsMargins(0, 0, 0, 0)

        # 左侧：生成结果
        result_left_widget = QWidget()
        result_left_layout = QVBoxLayout(result_left_widget)
        result_left_layout.setSpacing(5)
        result_left_layout.setContentsMargins(0, 0, 0, 0)

        result_title = QLabel("🖼️ Generated Results")
        result_title.setStyleSheet("""
            QLabel {
                font-size: 11pt;
                font-weight: 600;
                color: #1f2937;
                padding-bottom: 5px;
            }
        """)
        result_left_layout.addWidget(result_title)

        self.result_label = ClickableImageLabel()
        self.result_label.setText("Generated results will be displayed here")
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setFixedHeight(300)  # 固定高度
        self.result_label.setMinimumWidth(250)
        self.result_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.result_label.setScaledContents(False)
        self.result_label.setWordWrap(False)
        self.result_label.setStyleSheet("""
            QLabel {
                border: 2px dashed #d1d5db;
                border-radius: 10px;
                background-color: #f9fafb;
                color: #6b7280;
                font-size: 11pt;
                font-weight: 500;
            }
        """)
        result_left_layout.addWidget(self.result_label)

        result_container_layout.addWidget(result_left_widget)

        # 右侧：编辑结果
        result_right_widget = QWidget()
        result_right_layout = QVBoxLayout(result_right_widget)
        result_right_layout.setSpacing(5)
        result_right_layout.setContentsMargins(0, 0, 0, 0)

        edit_result_title = QLabel("✏️ Edited Results")
        edit_result_title.setStyleSheet("""
            QLabel {
                font-size: 11pt;
                font-weight: 600;
                color: #1f2937;
                padding-bottom: 5px;
            }
        """)
        result_right_layout.addWidget(edit_result_title)

        self.edit_result_label = ClickableImageLabel()
        self.edit_result_label.setText("Edited results will be displayed here")
        self.edit_result_label.setAlignment(Qt.AlignCenter)
        self.edit_result_label.setFixedHeight(300)  # 固定高度，与生成结果保持一致
        self.edit_result_label.setMinimumWidth(250)
        self.edit_result_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.edit_result_label.setScaledContents(False)
        self.edit_result_label.setWordWrap(False)
        self.edit_result_label.setStyleSheet("""
            QLabel {
                border: 2px dashed #d1d5db;
                border-radius: 10px;
                background-color: #f9fafb;
                color: #6b7280;
                font-size: 11pt;
                font-weight: 500;
            }
        """)
        result_right_layout.addWidget(self.edit_result_label)

        result_container_layout.addWidget(result_right_widget)

        middle_layout.addWidget(result_container)

        # 编辑图像Prompt输入区域
        edit_prompt_group = QGroupBox("✏️ Edit Image Prompt")
        edit_prompt_layout = QVBoxLayout()

        self.edit_prompt_edit = EditPromptTextEdit()
        self.edit_prompt_edit.set_main_window(self)
        self.edit_prompt_edit.setPlaceholderText(
            "Enter the Prompt for editing the image here...\nFor example: change headphones to red, add metallic texture, change background, etc.\n\nTip: Press Enter to edit the image directly (Shift+Enter for new line)")
        self.edit_prompt_edit.setMinimumHeight(80)
        self.edit_prompt_edit.setStyleSheet("""
            QTextEdit {
                border: 2px solid #f59e0b;
                border-radius: 8px;
                padding: 8px;
                background-color: #ffffff;
                selection-background-color: #fef3c7;
            }
            QTextEdit:focus {
                border: 2px solid #d97706;
            }
        """)
        edit_prompt_layout.addWidget(self.edit_prompt_edit)

        # 循环修改选项
        loop_edit_label = QLabel("🔄 Loop Modification Mode:")
        loop_edit_label.setStyleSheet("font-size: 10pt; font-weight: 500; color: #374151;")
        edit_prompt_layout.addWidget(loop_edit_label)

        self.loop_edit_group = QButtonGroup()
        loop_edit_layout = QHBoxLayout()
        loop_edit_layout.setSpacing(15)

        self.loop_edit_disable_radio = QRadioButton("Single Modification")
        self.loop_edit_disable_radio.setChecked(True)
        self.loop_edit_disable_radio.setToolTip("Each edit uses the original selected image as input")
        self.loop_edit_group.addButton(self.loop_edit_disable_radio, 0)
        loop_edit_layout.addWidget(self.loop_edit_disable_radio)

        self.loop_edit_enable_radio = QRadioButton("Loop Modification")
        self.loop_edit_enable_radio.setChecked(False)
        self.loop_edit_enable_radio.setToolTip(
            "Each edit uses the previously edited image as input, allowing iterative optimization")
        self.loop_edit_group.addButton(self.loop_edit_enable_radio, 1)
        loop_edit_layout.addWidget(self.loop_edit_enable_radio)

        loop_edit_layout.addStretch()

        radio_style = """
            QRadioButton {
                font-size: 10pt;
                font-weight: 500;
                color: #374151;
                padding: 5px;
            }
            QRadioButton::indicator {
                width: 18px;
                height: 18px;
                border: 2px solid #d1d5db;
                border-radius: 9px;
                background-color: #ffffff;
            }
            QRadioButton::indicator:checked {
                background-color: #f59e0b;
                border: 2px solid #d97706;
            }
        """
        self.loop_edit_disable_radio.setStyleSheet(radio_style)
        self.loop_edit_enable_radio.setStyleSheet(radio_style)

        edit_prompt_layout.addLayout(loop_edit_layout)
        edit_prompt_group.setLayout(edit_prompt_layout)
        middle_layout.addWidget(edit_prompt_group)

        # 操作按钮
        btn_action_layout = QHBoxLayout()
        btn_action_layout.setSpacing(10)

        self.save_selected_btn = QPushButton("💾 Save")
        self.save_selected_btn.setEnabled(False)
        self.save_selected_btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.save_selected_btn.setStyleSheet("""
            QPushButton {
                background-color: #10b981;
                font-size: 12pt;
                padding: 12px 24px;
                font-weight: 600;
            }
            QPushButton:hover {
                background-color: #059669;
            }
        """)
        btn_action_layout.addWidget(self.save_selected_btn)

        self.edit_image_btn = QPushButton("✏️ Edit")
        self.edit_image_btn.setEnabled(False)
        self.edit_image_btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.edit_image_btn.setStyleSheet("""
            QPushButton {
                background-color: #6366f1;
                font-size: 12pt;
                padding: 12px 24px;
                font-weight: 600;
            }
            QPushButton:hover {
                background-color: #4f46e5;
            }
            QPushButton:disabled {
                background-color: #9ca3af;
                color: #d1d5db;
            }
        """)
        self.edit_image_btn.setToolTip(
            "Edit image using Z-Image-Turbo ControlNet workflow\n\nThe Prompt will control the direction and content of editing:\n- Modify colors, materials, styles\n- Add or remove elements\n- Change composition and details\n\nPlease enter a new description in the \"Edit Image Prompt\" input box")
        btn_action_layout.addWidget(self.edit_image_btn)

        self.open_folder_btn = QPushButton("📁 Open")
        self.open_folder_btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.open_folder_btn.setStyleSheet("""
            QPushButton {
                background-color: #6366f1;
                font-size: 12pt;
                padding: 12px 24px;
                font-weight: 600;
            }
            QPushButton:hover {
                background-color: #4f46e5;
            }
        """)
        self.open_folder_btn.setToolTip("Open the folder where generated images are saved")
        btn_action_layout.addWidget(self.open_folder_btn)

        middle_layout.addLayout(btn_action_layout)

        # 右侧：参数 + 3D功能
        right_widget = QWidget()
        right_widget.setStyleSheet("""
            QWidget {
                background-color: #ffffff;
                border-radius: 12px;
                padding: 15px;
            }
        """)
        right_layout = QVBoxLayout(right_widget)
        right_layout.setSpacing(15)
        right_layout.setContentsMargins(15, 15, 15, 15)

        # 创建右侧列的内容布局（3D功能等）
        right_content_layout = QVBoxLayout()
        right_content_layout.setSpacing(15)
        right_content_layout.setContentsMargins(0, 0, 0, 0)

        # 3D模型显示区域
        model_3d_title = QLabel("🎲 3D Model Preview")
        model_3d_title.setStyleSheet("""
            QLabel {
                font-size: 11pt;
                font-weight: 600;
                color: #1f2937;
                padding-top: 10px;
                padding-bottom: 5px;
            }
        """)
        right_content_layout.addWidget(model_3d_title)

        # 3D模型显示容器
        self.model_3d_widget = QWidget()
        self.model_3d_layout = QVBoxLayout(self.model_3d_widget)
        self.model_3d_layout.setContentsMargins(0, 0, 0, 0)

        if PYVISTA_AVAILABLE:
            # 使用PyVista显示3D模型
            try:
                # 抑制OpenGL错误
                import warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    self.model_3d_plotter = QtInteractor(self)

                self.model_3d_plotter.setMinimumHeight(200)
                self.model_3d_plotter.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

                # 设置样式
                self.model_3d_plotter.interactor.setStyleSheet("""
                    QWidget {
                        border: 2px solid #e5e7eb;
                        border-radius: 10px;
                        background-color: #f9fafb;
                    }
                """)

                # 配置PyVista plotter（使用try-except捕获可能的OpenGL错误）
                try:
                    self.model_3d_plotter.set_background((0.06, 0.06, 0.12))
                    self.model_3d_plotter.enable_trackball_style()
                except Exception as e:
                    print(f"配置plotter时出现警告（可忽略）: {e}")

                self.model_3d_layout.addWidget(self.model_3d_plotter.interactor, stretch=1)
                print("PyVista 3D查看器已初始化")
            except Exception as e:
                print(f"初始化PyVista失败: {e}")
                import traceback
                traceback.print_exc()
                # 如果初始化失败，显示占位符
                self.model_3d_placeholder = QLabel(
                    f"PyVista initialization failed: {str(e)}\n\nPlease check if PyVista is installed correctly")
                self.model_3d_placeholder.setAlignment(Qt.AlignCenter)
                self.model_3d_placeholder.setMinimumHeight(200)
                self.model_3d_placeholder.setStyleSheet("""
                    QLabel {
                        border: 2px dashed #d1d5db;
                        border-radius: 10px;
                        background-color: #f9fafb;
                        color: #6b7280;
                        font-size: 10pt;
                        padding: 20px;
                    }
                """)
                self.model_3d_layout.addWidget(self.model_3d_placeholder)
        else:
            # 如果没有PyVista，显示占位符
            self.model_3d_placeholder = QLabel(
                "3D model preview requires PyVista\n\nPlease install: pip install pyvista pyvistaqt\n\nAfter generating a 3D model, you can also use an external 3D viewer to open .glb files")
            self.model_3d_placeholder.setAlignment(Qt.AlignCenter)
            self.model_3d_placeholder.setMinimumHeight(200)
            self.model_3d_placeholder.setStyleSheet("""
                QLabel {
                    border: 2px dashed #d1d5db;
                    border-radius: 10px;
                    background-color: #f9fafb;
                    color: #6b7280;
                    font-size: 10pt;
                    padding: 20px;
                }
            """)
            self.model_3d_layout.addWidget(self.model_3d_placeholder)

        # 初始化3D显示设置（保留变量，供独立窗口同步使用）
        self.model_3d_mesh = None
        self.model_3d_mesh_actor = None
        self.model_3d_color = (1.0, 1.0, 1.0)
        self.model_3d_show_edges = False
        self.model_3d_smooth_shading = True
        self.model_3d_ambient_light = 0.3
        self.model_3d_diffuse_light = 0.7

        # 分辨率设置（移到底部）
        resolution_group = QGroupBox("🎲 3D Generation Parameters")
        resolution_group.setStyleSheet("""
            QGroupBox {
                border: none;
                margin-top: 10px;
                padding-top: 0px;
                background-color: transparent;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 0px;
                padding: 0px;
            }
        """)
        resolution_group_layout = QVBoxLayout()

        resolution_layout = QVBoxLayout()
        resolution_label_layout = QHBoxLayout()
        resolution_label = QLabel("Resolution:")
        resolution_label.setStyleSheet("font-size: 10pt; font-weight: 500; color: #374151;")
        self.resolution_3d_label = QLabel("1024")
        self.resolution_3d_label.setStyleSheet("font-size: 10pt; font-weight: 600; color: #6366f1; min-width: 60px;")
        resolution_label_layout.addWidget(resolution_label)
        resolution_label_layout.addStretch()
        resolution_label_layout.addWidget(self.resolution_3d_label)
        resolution_layout.addLayout(resolution_label_layout)

        self.resolution_3d_slider = NoWheelSlider(Qt.Horizontal)
        self.resolution_3d_slider.setRange(512, 2048)
        self.resolution_3d_slider.setValue(1024)
        self.resolution_3d_slider.setSingleStep(256)
        self.resolution_3d_slider.setPageStep(512)
        self.resolution_3d_slider.setStyleSheet("""
            QSlider::groove:horizontal {
                border: 1px solid #d1d5db;
                height: 6px;
                background: #e5e7eb;
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: #8b5cf6;
                border: 2px solid #ffffff;
                width: 18px;
                height: 18px;
                border-radius: 9px;
                margin: -6px 0;
            }
            QSlider::handle:horizontal:hover {
                background: #7c3aed;
            }
            QSlider::handle:horizontal:pressed {
                background: #6d28d9;
            }
        """)
        resolution_layout.addWidget(self.resolution_3d_slider)
        resolution_group_layout.addLayout(resolution_layout)

        # 初始化3D分辨率标签显示
        self.on_resolution_3d_slider_changed(self.resolution_3d_slider.value())

        resolution_group.setLayout(resolution_group_layout)
        self.model_3d_layout.addWidget(resolution_group)

        # 3D模型控制按钮
        control_3d_layout = QHBoxLayout()
        control_3d_layout.setSpacing(5)

        # 3D生成按钮（移到底部按钮区域）
        self.generate_3d_btn = QPushButton("🎲 Generate 3D")
        self.generate_3d_btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.generate_3d_btn.setStyleSheet("""
            QPushButton {
                background-color: #8b5cf6;
                font-size: 12pt;
                padding: 12px 24px;
                font-weight: 600;
            }
            QPushButton:hover {
                background-color: #7c3aed;
            }
        """)
        self.generate_3d_btn.setToolTip("Generate 3D model (.glb file) based on the selected image")
        control_3d_layout.addWidget(self.generate_3d_btn)

        self.open_3d_external_btn = QPushButton("🔍 External")
        self.open_3d_external_btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.open_3d_external_btn.setStyleSheet("""
            QPushButton {
                background-color: #10b981;
                font-size: 12pt;
                padding: 12px 24px;
                font-weight: 600;
            }
            QPushButton:hover {
                background-color: #059669;
            }
        """)
        control_3d_layout.addWidget(self.open_3d_external_btn)

        self.open_3d_window_btn = QPushButton("🪟 Window")
        self.open_3d_window_btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.open_3d_window_btn.setStyleSheet("""
            QPushButton {
                background-color: #8b5cf6;
                font-size: 12pt;
                padding: 12px 24px;
                font-weight: 600;
            }
            QPushButton:hover {
                background-color: #7c3aed;
            }
        """)
        self.open_3d_window_btn.setToolTip("Open 3D model viewer in a separate window")
        control_3d_layout.addWidget(self.open_3d_window_btn)
        self.model_3d_layout.addLayout(control_3d_layout)

        right_content_layout.addWidget(self.model_3d_widget)

        # 保存当前3D模型路径
        self.current_3d_model_path = None

        # 创建右侧内容widget
        right_content_widget = QWidget()
        right_content_widget.setLayout(right_content_layout)

        # 将右侧内容添加到右侧布局
        right_layout.addWidget(right_content_widget)

        # 创建滚动区域用于中间列内容
        middle_scroll = QScrollArea()
        middle_scroll.setWidget(middle_widget)
        middle_scroll.setWidgetResizable(True)
        middle_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        middle_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        middle_scroll.setFrameShape(QScrollArea.NoFrame)
        middle_scroll.setStyleSheet("""
            QScrollArea {
                border: none;
                background-color: transparent;
            }
            QScrollBar:vertical {
                border: none;
                background-color: #e5e7eb;
                width: 10px;
                border-radius: 5px;
                margin: 2px;
            }
            QScrollBar::handle:vertical {
                background-color: #9ca3af;
                border-radius: 5px;
                min-height: 20px;
                margin: 1px;
            }
            QScrollBar::handle:vertical:hover {
                background-color: #6b7280;
            }
            QScrollBar::handle:vertical:pressed {
                background-color: #4b5563;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0px;
            }
            QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {
                background: transparent;
            }
        """)

        # 创建滚动区域用于右侧内容
        right_scroll = QScrollArea()
        right_scroll.setWidget(right_widget)
        right_scroll.setWidgetResizable(True)
        right_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        right_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        right_scroll.setFrameShape(QScrollArea.NoFrame)
        right_scroll.setStyleSheet("""
            QScrollArea {
                border: none;
                background-color: transparent;
            }
            QScrollBar:vertical {
                border: none;
                background-color: #e5e7eb;
                width: 10px;
                border-radius: 5px;
                margin: 2px;
            }
            QScrollBar::handle:vertical {
                background-color: #9ca3af;
                border-radius: 5px;
                min-height: 20px;
                margin: 1px;
            }
            QScrollBar::handle:vertical:hover {
                background-color: #6b7280;
            }
            QScrollBar::handle:vertical:pressed {
                background-color: #4b5563;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0px;
            }
            QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {
                background: transparent;
            }
        """)

        # 创建三列布局：使用两个分割器
        # 第一个分割器：分割左侧和（中间+右侧）
        self.main_splitter = QSplitter(Qt.Horizontal)

        # 第二个分割器：分割中间和右侧
        self.right_splitter = QSplitter(Qt.Horizontal)
        self.right_splitter.addWidget(middle_scroll)
        self.right_splitter.addWidget(right_scroll)
        self.right_splitter.setStretchFactor(0, 2)  # 中间可拉伸
        self.right_splitter.setStretchFactor(1, 1)  # 右侧可拉伸

        # 设置左右区域的最小宽度
        left_widget.setMinimumWidth(400)  # 左侧画布区域最小宽度
        middle_scroll.setMinimumWidth(350)  # 中间控制区域最小宽度
        right_scroll.setMinimumWidth(300)  # 右侧控制区域最小宽度

        self.main_splitter.addWidget(left_widget)
        self.main_splitter.addWidget(self.right_splitter)

        # 设置分割比例（左侧占40%，中间+右侧占60%）
        self.main_splitter.setStretchFactor(0, 2)  # 左侧可拉伸
        self.main_splitter.setStretchFactor(1, 3)  # 中间+右侧可拉伸

        # 使用main_splitter作为主分割器
        self.splitter = self.main_splitter

        # 设置初始大小比例（根据窗口大小自动计算）
        # self.splitter.setSizes([600, 400])  # 注释掉，让系统自动分配

        # 设置分割器样式（应用到两个分割器）
        splitter_style = """
            QSplitter::handle {
                background-color: #e5e7eb;
                width: 4px;
            }
            QSplitter::handle:hover {
                background-color: #6366f1;
            }
            QSplitter::handle:horizontal {
                margin: 0 2px;
            }
        """
        self.main_splitter.setStyleSheet(splitter_style)
        self.right_splitter.setStyleSheet(splitter_style)

        # 将分割器添加到主布局
        main_layout.addWidget(self.main_splitter)

        # 信号连接
        self.clear_btn.clicked.connect(self.canvas.clear_canvas)
        self.save_sketch_btn.clicked.connect(self.on_save_sketch)
        self.open_canvas_window_btn.clicked.connect(self.on_open_canvas_window)
        self.generate_btn.clicked.connect(self.on_generate)
        self.generate_3d_btn.clicked.connect(self.on_generate_3d)
        self.save_selected_btn.clicked.connect(self.on_save_selected)
        self.edit_image_btn.clicked.connect(self.on_edit_image)
        self.open_folder_btn.clicked.connect(self.on_open_folder)
        self.open_3d_external_btn.clicked.connect(self.on_open_3d_external)
        self.open_3d_window_btn.clicked.connect(self.on_open_3d_window)

        # 连接分割器移动信号，自动保存布局
        if QT_VERSION == "PySide6":
            from PySide6.QtCore import QTimer
        else:
            from PyQt5.QtCore import QTimer
        self.splitter_save_timer = QTimer()
        self.splitter_save_timer.setSingleShot(True)
        self.splitter_save_timer.timeout.connect(self.save_config)
        self.main_splitter.splitterMoved.connect(lambda: self.splitter_save_timer.start(500))  # 拖动停止500ms后保存
        self.right_splitter.splitterMoved.connect(lambda: self.splitter_save_timer.start(500))  # 拖动停止500ms后保存

        # 设置快捷键
        self._setup_shortcuts()

    def _setup_shortcuts(self):
        """设置全局快捷键"""
        # Ctrl+S: 保存手绘图
        save_shortcut = QShortcut(QKeySequence("Ctrl+S"), self)
        save_shortcut.activated.connect(self.on_save_sketch)

        # Ctrl+Z: 撤销（在画布上）
        undo_shortcut = QShortcut(QKeySequence("Ctrl+Z"), self.canvas)
        undo_shortcut.activated.connect(self.canvas.undo)

        # Ctrl+Y: 重做（在画布上）
        redo_shortcut = QShortcut(QKeySequence("Ctrl+Y"), self.canvas)
        redo_shortcut.activated.connect(self.canvas.redo)

        # Ctrl+Shift+Z: 重做（在画布上）
        redo_shortcut2 = QShortcut(QKeySequence("Ctrl+Shift+Z"), self.canvas)
        redo_shortcut2.activated.connect(self.canvas.redo)

        # Delete/Backspace: 清空画布（当画布有焦点时）
        # 这个已经在PaintCanvas的keyPressEvent中处理了

        # +/-: 调整画笔大小（当画布有焦点时）
        # 这个已经在PaintCanvas的keyPressEvent中处理了

    def on_import_image(self):
        """导入图片到画布"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "选择要导入的图片",
            "",
            "图像文件 (*.png *.jpg *.jpeg *.bmp *.gif);;所有文件 (*.*)"
        )
        if file_path:
            success = self.canvas.load_image(file_path)
            if success:
                # 在结果区域显示成功信息
                self.result_label.setText(
                    f"✅ 图片已导入到画布:\n{os.path.basename(file_path)}\n\n你可以在画布上继续绘制或修改")
                self.result_label.setStyleSheet("""
                    QLabel {
                        border: 2px solid #10b981;
                        border-radius: 10px;
                        background-color: #ecfdf5;
                        color: #059669;
                        font-size: 9pt;
                        padding: 10px;
                    }
                """)
            else:
                # 显示错误信息
                self.result_label.setText(f"❌ 导入图片失败:\n{os.path.basename(file_path)}\n\n请确保文件格式正确")
                self.result_label.setStyleSheet("""
                    QLabel {
                        border: 2px solid #ef4444;
                        border-radius: 10px;
                        background-color: #fef2f2;
                        color: #dc2626;
                        font-size: 9pt;
                        padding: 10px;
                    }
                """)

    def on_save_sketch(self):
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Sketch", "sketch.png", "PNG Image (*.png);;JPG Image (*.jpg)"
        )
        if file_path:
            self.canvas.save_canvas(file_path)
            # 不弹窗，改为在标签上提示
            self.result_label.setText(f"✅ Sketch saved to:\n{file_path}")
            self.result_label.setStyleSheet("""
                QLabel {
                    border: 2px solid #10b981;
                    border-radius: 10px;
                    background-color: #ecfdf5;
                    color: #059669;
                    font-size: 9pt;
                    padding: 10px;
                }
            """)

    def on_open_canvas_window(self):
        """打开独立画布编辑窗口"""
        if not hasattr(self, 'canvas_window') or not self.canvas_window or not self.canvas_window.isVisible():
            self.canvas_window = CanvasEditWindow(self.canvas, self)
            self.canvas_window.setWindowFlags(Qt.Window | Qt.WindowCloseButtonHint | Qt.WindowMinMaxButtonsHint)
            self.canvas_window.show()
            self.canvas_window.raise_()
            self.canvas_window.activateWindow()
        else:
            # 如果窗口已经打开，将其置于前台
            self.canvas_window.raise_()
            self.canvas_window.activateWindow()

    def on_generate(self):
        prompt = self.prompt_edit.toPlainText().strip()
        if not prompt:
            # 不弹窗，直接在结果区域提示
            self.result_label.setText("⚠️ Please enter a prompt first")
            self.result_label.setStyleSheet("""
                QLabel {
                    border: 2px solid #f59e0b;
                    border-radius: 10px;
                    background-color: #fffbeb;
                    color: #d97706;
                    font-size: 10pt;
                    padding: 10px;
                }
            """)
            return

        num_images = int(self.num_images_spin.value())

        seed = int(self.seed_spin.value())
        steps = int(self.steps_spin.value())
        cfg = float(self.cfg_spin.value())

        # 获取图像分辨率设置（使用统一的分辨率滑块）
        resolution = int(self.image_resolution_slider.value())
        width = resolution
        height = resolution

        # 自动检测画布是否有内容，如果有就使用手绘图
        use_sketch = False
        sketch_path = None

        if self.canvas.has_content():
            # 画布有内容，自动使用手绘图
            use_sketch = True
            # 保存手绘图到临时文件
            temp_sketch_path = os.path.join(os.getcwd(), "temp_sketch.png")
            self.canvas.save_canvas(temp_sketch_path)
            sketch_path = temp_sketch_path
            print(f"检测到画布有内容，自动使用手绘图生成，手绘图路径: {sketch_path}")
        else:
            print("画布为空，仅使用Prompt生成")

        self.generate_btn.setEnabled(False)
        self.generate_btn.setText("Generating...")

        # 清空旧缩略图和选择状态（在生成开始前清空）
        self.selected_image_path = None
        self.thumbnail_widgets = []
        while self.gallery_layout.count():
            item = self.gallery_layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()

        QApplication.processEvents()

        # 用于跟踪是否是第一张图像
        first_image_processed = [False]  # 使用列表以便在回调中修改

        def on_image_saved_callback(image_path, image_index, total_images):
            """每保存一张图像时调用的回调函数"""
            try:
                pix = QPixmap(image_path)
                if pix.isNull():
                    return

                # 创建可选择的缩略图
                thumb_label = SelectableThumbnail(image_path, self)
                # 设置点击回调
                thumb_label.clicked = lambda p=image_path, t=thumb_label: self.on_select_image(p, t)
                self.gallery_layout.addWidget(thumb_label)
                self.thumbnail_widgets.append(thumb_label)

                # 第一张图像自动选中并显示在大图区域
                if not first_image_processed[0]:
                    thumb_label.select()
                    self.selected_image_path = image_path
                    # 如果是第一次选择图像，设置为原始图像路径
                    if self.original_image_path is None:
                        self.original_image_path = image_path
                    # 使用统一的方法更新图像显示
                    QApplication.processEvents()  # 确保界面已更新
                    self.update_result_image(image_path)
                    self.save_selected_btn.setEnabled(True)
                    self.edit_image_btn.setEnabled(True)
                    first_image_processed[0] = True

                # 更新界面
                QApplication.processEvents()
            except Exception as e:
                print(f"添加图像到画廊失败: {e}")

        try:
            save_paths = generate_image(
                prompt=prompt,
                seed=seed,
                steps=steps,
                cfg=cfg,
                width=width,
                height=height,
                batch_size=num_images,
                use_sketch=use_sketch,
                sketch_path=sketch_path,
                on_image_saved=on_image_saved_callback,
            )
        finally:
            self.generate_btn.setEnabled(True)
            self.generate_btn.setText("✨ Generate")

        if not save_paths:
            # 不弹窗，结果区域提示
            self.result_label.setText(
                "❌ Image generation failed\nPlease check if ComfyUI is running and the workflow is correct.")
            self.result_label.setStyleSheet("""
                QLabel {
                    border: 2px solid #ef4444;
                    border-radius: 10px;
                    background-color: #fef2f2;
                    color: #dc2626;
                    font-size: 10pt;
                    padding: 10px;
                }
            """)
            return

        # 显示保存文件夹信息
        if save_paths:
            output_folder = os.path.dirname(save_paths[0])
            print(f"✅ 所有图像已保存到文件夹: {output_folder}")
            print(f"   共生成 {len(save_paths)} 张图像")

        # 注意：图像已经在生成过程中通过回调函数添加到画廊了
        # 这里只需要检查是否有图像成功显示
        if not first_image_processed[0] and save_paths:
            self.result_label.setText("Image file saved, but could not be loaded and displayed in the interface.")
            self.result_label.setStyleSheet("""
                QLabel {
                    border: 2px dashed #d1d5db;
                    border-radius: 10px;
                    background-color: #f9fafb;
                    color: #9ca3af;
                }
            """)
            self.save_selected_btn.setEnabled(False)

    def _clear_result_pixmap(self):
        """清除结果标签的pixmap"""
        self.result_label.setPixmap(QPixmap())

    def update_result_image(self, image_path):
        """更新结果显示区域的图像"""
        if not image_path or not os.path.exists(image_path):
            return

        pix = QPixmap(image_path)
        if pix.isNull():
            return

        # 清除之前的文本内容和pixmap
        self.result_label.setText("")
        self.result_label.setWordWrap(False)

        # 获取label的实际尺寸
        # 先强制更新布局，确保尺寸已计算
        self.result_label.updateGeometry()
        QApplication.processEvents()

        label_size = self.result_label.size()
        label_width = label_size.width()
        label_height = label_size.height()

        # 如果尺寸还没有初始化，使用geometry获取
        if label_width <= 0 or label_height <= 0:
            label_geometry = self.result_label.geometry()
            label_width = label_geometry.width()
            label_height = label_geometry.height()

        # 如果仍然无效，尝试从父widget获取
        if label_width <= 0 or label_height <= 0:
            parent = self.result_label.parentWidget()
            if parent:
                parent_size = parent.size()
                if parent_size.width() > 0:
                    label_width = max(parent_size.width() - 40, 400)  # 减去边距，使用更大的默认值
                if parent_size.height() > 0:
                    label_height = max(500, label_height)  # 使用更大的默认高度

        # 如果仍然无效，使用默认值（增大默认尺寸）
        if label_width <= 0:
            label_width = 500
        if label_height <= 0:
            label_height = 500

        # 减去内边距和边框（padding 5px + border 2px * 2 = 约14px，留一些余量）
        available_width = max(label_width - 30, 200)  # 增加可用宽度
        available_height = max(label_height - 30, 200)  # 增加可用高度

        # 计算缩放比例，确保图像完整显示在可用区域内
        pix_width = pix.width()
        pix_height = pix.height()

        if pix_width <= 0 or pix_height <= 0:
            return

        # 计算缩放比例，使用较小的比例确保完整显示
        # 移除1.0的限制，允许图像缩小以适应空间
        width_ratio = available_width / pix_width
        height_ratio = available_height / pix_height
        scale_ratio = min(width_ratio, height_ratio)  # 允许缩小以适应空间

        # 计算新尺寸
        new_width = int(pix_width * scale_ratio)
        new_height = int(pix_height * scale_ratio)

        # 确保尺寸有效
        new_width = max(new_width, 1)
        new_height = max(new_height, 1)

        # 缩放图像
        scaled_pix = pix.scaled(
            new_width,
            new_height,
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )

        # 设置pixmap并确保对齐方式正确
        self.result_label.setPixmap(scaled_pix)
        self.result_label.setAlignment(Qt.AlignCenter)
        # 保存图像路径以便点击放大
        self.result_label.set_image_path(image_path)

    def update_edit_result_image(self, image_path):
        """更新编辑结果显示区域的图像"""
        if not image_path or not os.path.exists(image_path):
            return

        pix = QPixmap(image_path)
        if pix.isNull():
            return

        # 清除之前的文本内容和pixmap
        self.edit_result_label.setText("")
        self.edit_result_label.setWordWrap(False)

        # 获取label的实际尺寸
        self.edit_result_label.updateGeometry()
        QApplication.processEvents()

        label_size = self.edit_result_label.size()
        label_width = label_size.width()
        label_height = label_size.height()

        # 如果尺寸还没有初始化，使用geometry获取
        if label_width <= 0 or label_height <= 0:
            label_geometry = self.edit_result_label.geometry()
            label_width = label_geometry.width()
            label_height = label_geometry.height()

        # 如果仍然无效，尝试从父widget获取
        if label_width <= 0 or label_height <= 0:
            parent = self.edit_result_label.parentWidget()
            if parent:
                parent_size = parent.size()
                if parent_size.width() > 0:
                    label_width = max(parent_size.width() - 40, 400)
                if parent_size.height() > 0:
                    label_height = max(500, label_height)

        # 如果仍然无效，使用默认值
        if label_width <= 0:
            label_width = 500
        if label_height <= 0:
            label_height = 500

        # 减去内边距和边框
        available_width = max(label_width - 30, 200)
        available_height = max(label_height - 30, 200)

        # 计算缩放比例
        pix_width = pix.width()
        pix_height = pix.height()

        if pix_width <= 0 or pix_height <= 0:
            return

        width_ratio = available_width / pix_width
        height_ratio = available_height / pix_height
        scale_ratio = min(width_ratio, height_ratio)

        # 计算新尺寸
        new_width = int(pix_width * scale_ratio)
        new_height = int(pix_height * scale_ratio)

        # 确保尺寸有效
        new_width = max(new_width, 1)
        new_height = max(new_height, 1)

        # 缩放图像
        scaled_pix = pix.scaled(
            new_width,
            new_height,
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )

        # 设置pixmap并确保对齐方式正确
        self.edit_result_label.setPixmap(scaled_pix)
        self.edit_result_label.setAlignment(Qt.AlignCenter)
        # 保存图像路径以便点击放大
        self.edit_result_label.set_image_path(image_path)
        self.result_label.setStyleSheet("""
            QLabel {
                border: 2px solid #6366f1;
                border-radius: 10px;
                background-color: #ffffff;
                padding: 5px;
            }
        """)

    def resizeEvent(self, event):
        """窗口大小改变时，重新缩放当前显示的图像"""
        super().resizeEvent(event)
        # 如果有选中的图像，重新显示以适应新尺寸
        if self.selected_image_path:
            # 使用QTimer延迟更新，确保布局已完成
            if QT_VERSION == "PySide6":
                from PySide6.QtCore import QTimer
            else:
                from PyQt5.QtCore import QTimer
            QTimer.singleShot(100, lambda: self.update_result_image(self.selected_image_path))

    def on_select_image(self, image_path, thumbnail):
        """处理缩略图点击选择"""
        # 取消所有缩略图的选中状态
        for thumb in self.thumbnail_widgets:
            thumb.deselect()

        # 选中当前点击的缩略图
        thumbnail.select()
        self.selected_image_path = image_path

        # 如果是第一次选择，或者用户重新选择了新图像（不是编辑后的图像），保存为原始图像
        # 判断：如果original_image_path为None，或者选择的图像不在编辑后的图像列表中，则认为是新选择的原始图像
        if self.original_image_path is None:
            self.original_image_path = image_path
        # 如果用户从画廊中选择了不同的图像，也更新原始图像路径
        # 这里可以通过检查文件名是否包含"_edited_"来判断是否是编辑后的图像
        if "_edited_" not in os.path.basename(image_path):
            # 如果选择的不是编辑后的图像，更新原始图像路径
            self.original_image_path = image_path

        # 在大图区域显示选中的图像
        self.update_result_image(image_path)

        # 启用保存和编辑按钮
        self.save_selected_btn.setEnabled(True)
        self.edit_image_btn.setEnabled(True)

    def on_save_selected(self):
        """保存选中的图像"""
        if not self.selected_image_path:
            self.result_label.setText("⚠️ 请先选择一张图像")
            self.result_label.setStyleSheet("""
                QLabel {
                    border: 2px solid #f59e0b;
                    border-radius: 10px;
                    background-color: #fffbeb;
                    color: #d97706;
                    font-size: 10pt;
                    padding: 10px;
                }
            """)
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self, "保存选中图像", os.path.basename(self.selected_image_path),
            "PNG 图像 (*.png);;JPG 图像 (*.jpg);;所有文件 (*.*)"
        )

        if file_path:
            try:
                shutil.copy2(self.selected_image_path, file_path)
                self.result_label.setText(f"✅ Selected image saved to:\n{file_path}")
                self.result_label.setStyleSheet("""
                    QLabel {
                        border: 2px solid #10b981;
                        border-radius: 10px;
                        background-color: #ecfdf5;
                        color: #059669;
                        font-size: 9pt;
                        padding: 10px;
                    }
                """)
            except Exception as e:
                self.result_label.setText(f"❌ 保存失败:\n{str(e)}")
                self.result_label.setStyleSheet("""
                    QLabel {
                        border: 2px solid #ef4444;
                        border-radius: 10px;
                        background-color: #fef2f2;
                        color: #dc2626;
                        font-size: 10pt;
                        padding: 10px;
                    }
                """)

    def on_resolution_slider_changed(self, value):
        """当分辨率滑块改变时，更新标签显示（宽度和高度相同）"""
        self.image_resolution_label.setText(f"{value}x{value} px")

    def on_width_slider_changed(self, value):
        """兼容性函数：当宽度滑块改变时，更新标签显示"""
        # 现在使用统一的分辨率滑块，这个函数保留用于兼容性
        if hasattr(self, 'image_resolution_label'):
            self.image_resolution_label.setText(f"{value}x{value} px")

    def on_height_slider_changed(self, value):
        """兼容性函数：当高度滑块改变时，更新标签显示"""
        # 现在使用统一的分辨率滑块，这个函数保留用于兼容性
        if hasattr(self, 'image_resolution_label'):
            self.image_resolution_label.setText(f"{value}x{value} px")

    def on_width_changed(self, value):
        """兼容性函数：当宽度改变时的处理"""
        # 现在使用统一的分辨率滑块，宽度和高度总是相同
        pass

    def on_height_changed(self, value):
        """兼容性函数：当高度改变时的处理"""
        # 现在使用统一的分辨率滑块，宽度和高度总是相同
        pass

    def on_resolution_3d_slider_changed(self, value):
        """当3D分辨率滑块改变时，更新标签显示"""
        # 滑块值已经是512的倍数（因为singleStep=512），直接显示
        self.resolution_3d_label.setText(f"{value}")

    def on_copy_main_prompt(self):
        """将主Prompt复制到编辑Prompt输入框"""
        main_prompt = self.prompt_edit.toPlainText().strip()
        if main_prompt:
            self.edit_prompt_edit.setPlainText(main_prompt)
            # 显示提示信息
            self.result_label.setText(
                f"✅ Main Prompt copied to Edit Prompt input box\n\nPrompt: {main_prompt[:100]}{'...' if len(main_prompt) > 100 else ''}")
            self.result_label.setStyleSheet("""
                QLabel {
                    border: 2px solid #10b981;
                    border-radius: 10px;
                    background-color: #ecfdf5;
                    color: #059669;
                    font-size: 9pt;
                    padding: 10px;
                }
            """)
        else:
            self.result_label.setText("⚠️ Main Prompt input box is empty, nothing to copy")
            self.result_label.setStyleSheet("""
                QLabel {
                    border: 2px solid #f59e0b;
                    border-radius: 10px;
                    background-color: #fffbeb;
                    color: #d97706;
                    font-size: 10pt;
                    padding: 10px;
                }
            """)

    def on_edit_image(self):
        """编辑选中的图像"""
        if not self.selected_image_path:
            self.result_label.setText("⚠️ 请先选择一张图像")
            self.result_label.setStyleSheet("""
                QLabel {
                    border: 2px solid #f59e0b;
                    border-radius: 10px;
                    background-color: #fffbeb;
                    color: #d97706;
                    font-size: 10pt;
                    padding: 10px;
                }
            """)
            return

        # 使用专门的编辑Prompt输入框
        prompt = self.edit_prompt_edit.toPlainText().strip()
        if not prompt:
            self.result_label.setText(
                "⚠️ Please enter a prompt in the Edit Image Prompt input box\n\nThe prompt will control the direction and content of image editing\nFor example: change headphones to red, add metallic texture, change background, etc.")
            self.result_label.setStyleSheet("""
                QLabel {
                    border: 2px solid #f59e0b;
                    border-radius: 10px;
                    background-color: #fffbeb;
                    color: #d97706;
                    font-size: 10pt;
                    padding: 10px;
                }
            """)
            return

        # 根据循环修改选项决定使用哪个图像作为输入
        enable_loop_edit = self.loop_edit_enable_radio.isChecked()
        if enable_loop_edit:
            # 启用循环修改：使用当前选中的图像（可能是上次编辑的结果）
            input_image_path = self.selected_image_path
            edit_mode = "循环修改（基于上次编辑结果）"
        else:
            # 禁用循环修改：使用原始选中的图像
            if self.original_image_path and os.path.exists(self.original_image_path):
                input_image_path = self.original_image_path
                edit_mode = "单次修改（基于原始图像）"
            else:
                # 如果没有原始图像路径，使用当前选中的图像
                input_image_path = self.selected_image_path
                edit_mode = "单次修改（基于当前图像）"

        # 获取参数（编辑功能只对单张图像进行编辑，不使用批量参数）
        seed = int(self.seed_spin.value())
        steps = int(self.steps_spin.value())
        cfg = float(self.cfg_spin.value())

        # 禁用编辑按钮
        self.edit_image_btn.setEnabled(False)
        self.edit_image_btn.setText("Editing...")

        # 不显示编辑状态信息，保持编辑结果区域为空，只用于显示图像
        # 清空编辑结果区域的文字，只显示图像
        self.edit_result_label.setText("")

        QApplication.processEvents()

        try:
            # 使用 Z-Image-Turbo ControlNet 工作流编辑图像
            # 只对选中的单张图像进行编辑，batch_size固定为1
            # 根据循环修改选项使用不同的输入图像
            save_paths = edit_image(
                prompt=prompt,
                image_path=input_image_path,
                seed=seed if seed >= 0 else -1,  # 如果seed为0，使用-1表示随机
                steps=max(steps, 9) if steps < 9 else steps,  # Z-Image-Turbo建议至少9步
                cfg=cfg,
                control_strength=0.85,  # ControlNet强度，默认0.85
                canny_low=0.1,  # Canny低阈值
                canny_high=0.32,  # Canny高阈值
                batch_size=1  # 编辑功能只处理单张图像
            )
        finally:
            self.edit_image_btn.setEnabled(True)
            self.edit_image_btn.setText("✏️ Edit")

        if not save_paths:
            # 编辑失败时不显示文字信息，保持编辑结果区域为空
            # 可以通过其他方式提示用户（如消息框或状态栏）
            return

        # 显示编辑后的图像（只显示在编辑结果区域，不更新生成结果区域）
        first_ok = False
        for path in save_paths:
            pix = QPixmap(path)
            if pix.isNull():
                continue

            if not first_ok:
                # 第一张图像只显示在编辑结果区域
                # 如果启用了循环修改，更新原始图像路径为当前编辑后的图像（用于下次编辑）
                if self.loop_edit_enable_radio.isChecked():
                    self.original_image_path = path
                    # 更新selected_image_path用于下次编辑，但不更新生成结果区域的显示
                    self.selected_image_path = path
                # 使用统一的方法更新编辑结果图像显示
                QApplication.processEvents()  # 确保界面已更新
                self.update_edit_result_image(path)
                self.save_selected_btn.setEnabled(True)
                self.edit_image_btn.setEnabled(True)
                # 保留编辑Prompt输入框的内容，方便继续编辑或修改
                first_ok = True

        if not first_ok:
            self.edit_result_label.setText("Image file saved, but could not be loaded and displayed in the interface.")
            self.edit_result_label.setStyleSheet("""
                QLabel {
                    border: 2px dashed #d1d5db;
                    border-radius: 10px;
                    background-color: #f9fafb;
                    color: #9ca3af;
                }
            """)
            self.save_selected_btn.setEnabled(False)
            self.edit_image_btn.setEnabled(False)

    def on_open_folder(self):
        """Open the folder where generated images are saved"""
        # 获取当前日期的输出文件夹
        folder_path = get_output_folder()

        if not os.path.exists(folder_path):
            # 如果文件夹不存在，创建它
            os.makedirs(folder_path, exist_ok=True)

        # 打开文件夹
        if open_folder(folder_path):
            self.result_label.setText(f"✅ Folder opened:\n{folder_path}")
            self.result_label.setStyleSheet("""
                QLabel {
                    border: 2px solid #10b981;
                    border-radius: 10px;
                    background-color: #ecfdf5;
                    color: #059669;
                    font-size: 9pt;
                    padding: 10px;
                }
            """)
        else:
            self.result_label.setText(f"❌ Failed to open folder:\n{folder_path}")
            self.result_label.setStyleSheet("""
                QLabel {
                    border: 2px solid #ef4444;
                    border-radius: 10px;
                    background-color: #fef2f2;
                    color: #dc2626;
                    font-size: 10pt;
                    padding: 10px;
                }
            """)

    def on_generate_3d(self):
        """Generate 3D model"""
        # 检查是否有选中的图像
        if not self.selected_image_path:
            # 不占用生成结果区域显示提示信息
            # 可以通过其他方式提示用户（如消息框）
            return

        # 获取参数
        seed = int(self.seed_spin.value())
        steps = 30  # 3D生成默认30步
        cfg = 5.0  # 3D生成默认CFG 5.0
        resolution = int(self.resolution_3d_slider.value())

        self.generate_3d_btn.setEnabled(False)
        self.generate_3d_btn.setText("Generating 3D...")

        QApplication.processEvents()

        try:
            save_path = generate_3d_model(
                image_path=self.selected_image_path,
                seed=seed,
                steps=steps,
                cfg=cfg,
                resolution=resolution,
                remove_bg=True  # 明确启用背景移除，只保留前景图像
            )
        finally:
            self.generate_3d_btn.setEnabled(True)
            self.generate_3d_btn.setText("🎲 Generate 3D")

        if not save_path:
            # 生成失败时不占用生成结果区域显示错误信息
            # 可以通过其他方式提示用户（如消息框）
            return

        # 保存3D模型路径并自动加载显示
        self.current_3d_model_path = save_path
        self.load_3d_model(save_path)

        # 保存配置，以便下次启动时自动加载3D模型
        self.save_config()

        # 不占用生成结果区域显示成功信息，保持该区域只用于显示图像

    def load_3d_model(self, glb_path=None):
        """加载并显示3D模型"""
        if not glb_path:
            # 如果没有提供路径，打开文件选择对话框
            glb_path, _ = QFileDialog.getOpenFileName(
                self, "选择3D模型文件", "", "GLB文件 (*.glb);;所有文件 (*.*)"
            )

        if not glb_path or not os.path.exists(glb_path):
            return

        self.current_3d_model_path = glb_path

        # 保存配置，以便下次启动时自动加载3D模型
        self.save_config()

        if PYVISTA_AVAILABLE:
            # 检查model_3d_plotter是否存在
            if not hasattr(self, 'model_3d_plotter') or self.model_3d_plotter is None:
                print("错误: model_3d_plotter未初始化")
                return

            try:
                # 清除之前的模型
                self.model_3d_plotter.clear()

                # 加载GLB模型
                print(f"正在加载3D模型: {glb_path}")
                self.model_3d_mesh = pv.read(glb_path)
                # 过滤掉外部盒子/墙
                self.model_3d_mesh = filter_external_boxes(self.model_3d_mesh)

                # 使用更新方法添加模型
                self.update_model_3d_display()

                # 重置相机并设置等轴视图
                try:
                    self.model_3d_plotter.reset_camera()
                    self.model_3d_plotter.view_isometric()
                    # 确保不显示坐标系
                    remove_axes_from_plotter(self.model_3d_plotter)
                except Exception as e:
                    if "wglMakeCurrent" not in str(e) and "句柄无效" not in str(e):
                        print(f"相机设置警告（可忽略）: {e}")

                print("3D模型加载成功")

            except Exception as e:
                print(f"加载3D模型失败: {e}")
                import traceback
                traceback.print_exc()

                # 显示错误信息
                self.model_3d_plotter.add_text(
                    f"加载失败：{str(e)}\n路径：{glb_path}",
                    position='upper_left',
                    font_size=14,
                    color='red'
                )
        else:
            # 如果没有PyVista，更新占位符
            if hasattr(self, 'model_3d_placeholder'):
                self.model_3d_placeholder.setText(
                    f"3D模型已加载: {os.path.basename(glb_path)}\n\n"
                    "请安装PyVista以在GUI中查看: pip install pyvista pyvistaqt\n\n"
                    "或点击'外部查看'按钮使用外部3D查看器打开"
                )

    def create_simple_3d_viewer_html(self, glb_path):
        """创建简化的3D模型查看HTML（使用更稳定的Three.js版本）"""
        # 转换路径为URL
        abs_path = os.path.abspath(glb_path)
        if platform.system() == "Windows":
            # Windows: 使用file:///协议
            glb_url = "file:///" + abs_path.replace('\\', '/')
        else:
            # Linux/macOS
            glb_url = "file://" + abs_path

        print(f"GLB文件路径: {abs_path}")
        print(f"GLB URL: {glb_url}")

        html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>3D Viewer</title>
    <style>
        body {{ margin: 0; padding: 0; overflow: hidden; background-color: #f0f0f0; }}
        #info {{ position: absolute; top: 10px; left: 10px; color: white; background: rgba(0,0,0,0.5); padding: 10px; border-radius: 5px; font-family: Arial, sans-serif; font-size: 12px; z-index: 100; }}
    </style>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/loaders/GLTFLoader.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
</head>
<body>
    <div id="info">加载3D模型...</div>
    <script>
        // 基础Three.js场景
        let scene, camera, renderer, controls;

        function init() {{
            // 场景
            scene = new THREE.Scene();
            scene.background = new THREE.Color(0xf0f0f0);

            // 相机
            camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
            camera.position.set(0, 0, 5);

            // 渲染器
            renderer = new THREE.WebGLRenderer({{ antialias: true }});
            renderer.setSize(window.innerWidth, window.innerHeight);
            renderer.shadowMap.enabled = true;
            document.body.appendChild(renderer.domElement);

            // 控制器
            controls = new OrbitControls(camera, renderer.domElement);
            controls.enableDamping = true;
            controls.dampingFactor = 0.05;

            // 灯光
            const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
            scene.add(ambientLight);

            const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
            directionalLight.position.set(5, 5, 5);
            scene.add(directionalLight);

            // 加载模型
            loadModel();

            // 窗口大小调整
            window.addEventListener('resize', onWindowResize);

            // 动画循环
            animate();
        }}

        function loadModel() {{
            const loader = new GLTFLoader();
            const glbUrl = '{glb_url}';

            console.log('正在加载: ' + glbUrl);

            loader.load(
                glbUrl,
                function(gltf) {{
                    console.log('模型加载成功');
                    const model = gltf.scene;
                    scene.add(model);

                    // 调整相机位置
                    const box = new THREE.Box3().setFromObject(model);
                    const center = box.getCenter(new THREE.Vector3());
                    const size = box.getSize(new THREE.Vector3());
                    const maxDim = Math.max(size.x, size.y, size.z);

                    camera.position.copy(center);
                    camera.position.z += maxDim * 1.5;
                    controls.target.copy(center);
                    controls.update();

                    document.getElementById('info').innerHTML = '✓ 3D模型已加载<br>鼠标拖动旋转，滚轮缩放';
                }},
                function(xhr) {{
                    const percent = Math.round(xhr.loaded / xhr.total * 100);
                    document.getElementById('info').innerHTML = `加载中... ${{percent}}%`;
                }},
                function(error) {{
                    console.error('加载失败:', error);
                    document.getElementById('info').innerHTML = '加载失败: ' + error;
                }}
            );
        }}

        function onWindowResize() {{
            camera.aspect = window.innerWidth / window.innerHeight;
            camera.updateProjectionMatrix();
            renderer.setSize(window.innerWidth, window.innerHeight);
        }}

        function animate() {{
            requestAnimationFrame(animate);
            controls.update();
            renderer.render(scene, camera);
        }}

        // 初始化
        init();
    </script>
</body>
</html>"""
        return html

    # 以下方法已废弃，保留用于兼容性（现在使用PyVista）
    def _load_glb_after_page_ready(self, glb_path):
        """已废弃：页面加载完成后，通过JavaScript加载GLB（现在使用PyVista）"""
        pass

    def _inject_glb_loading(self, glb_url):
        """已废弃：注入JavaScript代码加载GLB（现在使用PyVista）"""
        pass

    def _try_alternative_3d_viewer(self, glb_path):
        """已废弃：尝试使用备用3D查看器（现在使用PyVista）"""
        pass

    def test_webengine(self):
        """已废弃：测试WebEngine是否正常工作（现在使用PyVista）"""
        if PYVISTA_AVAILABLE and hasattr(self, 'model_3d_plotter'):
            print("PyVista可用")
        else:
            print("PyVista不可用")

    def update_model_3d_display(self):
        """更新主窗口3D预览的显示设置"""
        if not PYVISTA_AVAILABLE or self.model_3d_mesh is None or not hasattr(self, 'model_3d_plotter'):
            return

        try:
            # 抑制OpenGL警告
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                # 清除之前的mesh
                if self.model_3d_mesh_actor is not None:
                    try:
                        self.model_3d_plotter.remove_actor(self.model_3d_mesh_actor)
                    except:
                        pass

                # 添加新的mesh
                self.model_3d_mesh_actor = self.model_3d_plotter.add_mesh(
                    self.model_3d_mesh,
                    color=self.model_3d_color,
                    smooth_shading=self.model_3d_smooth_shading,
                    show_edges=self.model_3d_show_edges,
                    interpolate_before_map=True,
                    pbr=False,  # 禁用PBR以便更好地控制颜色和光线
                    ambient=self.model_3d_ambient_light,
                    diffuse=self.model_3d_diffuse_light,
                    specular=0.3,
                    specular_power=30.0
                )
                # 确保不显示坐标系
                remove_axes_from_plotter(self.model_3d_plotter)

                # 安全地执行渲染
                safe_render(self.model_3d_plotter)
        except Exception as e:
            # 只打印非OpenGL相关的错误
            if "wglMakeCurrent" not in str(e) and "句柄无效" not in str(e):
                print(f"更新3D预览显示失败: {e}")
                import traceback
                traceback.print_exc()

    def on_model_3d_color_changed(self):
        """主窗口3D预览颜色选择器回调"""
        color = QColorDialog.getColor(
            QColor(int(self.model_3d_color[0] * 255), int(self.model_3d_color[1] * 255),
                   int(self.model_3d_color[2] * 255)),
            self, "选择表面颜色"
        )
        if color.isValid():
            self.model_3d_color = (color.red() / 255.0, color.green() / 255.0, color.blue() / 255.0)
            self.model_3d_color_button.setStyleSheet(
                f"background-color: rgb({color.red()}, {color.green()}, {color.blue()}); "
                f"border: 2px solid #d1d5db; border-radius: 4px;"
            )
            self.update_model_3d_display()
            # 保存配置
            self.save_config()

    def on_model_3d_edges_changed(self, state):
        """主窗口3D预览网格线显示切换"""
        self.model_3d_show_edges = (state == Qt.Checked)
        self.update_model_3d_display()
        # 保存配置
        self.save_config()

    def on_model_3d_smooth_changed(self, state):
        """主窗口3D预览平滑着色切换"""
        self.model_3d_smooth_shading = (state == Qt.Checked)
        self.update_model_3d_display()
        # 保存配置
        self.save_config()

    def on_model_3d_ambient_label_changed(self, value):
        """主窗口3D预览环境光强度改变（仅更新标签显示）"""
        ambient_value = value / 100.0
        self.model_3d_ambient_label.setText(f"{ambient_value:.2f}")

    def on_model_3d_ambient_changed(self):
        """主窗口3D预览环境光强度改变（松开滑块时更新效果）"""
        value = self.model_3d_ambient_slider.value()
        self.model_3d_ambient_light = value / 100.0
        self.model_3d_ambient_label.setText(f"{self.model_3d_ambient_light:.2f}")
        self.update_model_3d_display()
        # 保存配置
        self.save_config()

    def on_model_3d_diffuse_label_changed(self, value):
        """主窗口3D预览漫反射光强度改变（仅更新标签显示）"""
        diffuse_value = value / 100.0
        self.model_3d_diffuse_label.setText(f"{diffuse_value:.2f}")

    def on_model_3d_diffuse_changed(self):
        """主窗口3D预览漫反射光强度改变（松开滑块时更新效果）"""
        value = self.model_3d_diffuse_slider.value()
        self.model_3d_diffuse_light = value / 100.0
        self.model_3d_diffuse_label.setText(f"{self.model_3d_diffuse_light:.2f}")
        self.update_model_3d_display()
        # 保存配置
        self.save_config()

    def on_save_3d_model_main(self):
        """主窗口保存编辑后的3D模型"""
        if not PYVISTA_AVAILABLE or not hasattr(self, 'model_3d_mesh') or self.model_3d_mesh is None:
            QMessageBox.warning(self, "保存失败", "3D模型未加载或PyVista不可用")
            return

        if not NUMPY_AVAILABLE:
            QMessageBox.warning(self, "保存失败", "需要安装numpy库: pip install numpy")
            return

        if not hasattr(self, 'current_3d_model_path') or not self.current_3d_model_path:
            QMessageBox.warning(self, "保存失败", "没有可保存的3D模型")
            return

        try:
            # 创建带颜色的mesh副本
            mesh_to_save = self.model_3d_mesh.copy()

            # 处理MultiBlock情况：如果是MultiBlock，需要合并为单个mesh
            def extract_single_mesh(mesh):
                """递归提取单个mesh，处理MultiBlock"""
                if not hasattr(mesh, 'n_blocks'):
                    # 已经是单个mesh
                    return mesh

                if mesh.n_blocks == 0:
                    raise ValueError("MultiBlock中没有块")

                # 收集所有有效的块
                blocks = []
                for i in range(mesh.n_blocks):
                    block = mesh[i]
                    if block is not None:
                        # 递归处理嵌套的MultiBlock
                        if hasattr(block, 'n_blocks') and block.n_blocks > 0:
                            block = extract_single_mesh(block)
                        # 确保是单个mesh且有顶点
                        if hasattr(block, 'n_points') and block.n_points > 0:
                            blocks.append(block)

                if not blocks:
                    raise ValueError("MultiBlock中没有有效的块")

                # 合并所有块
                if len(blocks) == 1:
                    return blocks[0]
                else:
                    # 使用merge合并，然后确保结果是单个mesh
                    merged = pv.merge(blocks)
                    # 如果merge后还是MultiBlock，使用extract_geometry
                    if hasattr(merged, 'n_blocks') and merged.n_blocks > 0:
                        return merged.extract_geometry()
                    return merged

            # 提取单个mesh
            mesh_to_save = extract_single_mesh(mesh_to_save)

            # 确保mesh_to_save是单个mesh对象
            if not hasattr(mesh_to_save, 'n_points'):
                raise ValueError(f"无法处理mesh类型: {type(mesh_to_save)}")

            # 将颜色应用到mesh的顶点
            # 创建一个颜色数组，每个顶点使用相同的颜色
            n_points = mesh_to_save.n_points
            colors = np.array([self.model_3d_color] * n_points) * 255  # 转换为0-255范围
            colors = colors.astype(np.uint8)

            # 设置顶点颜色
            mesh_to_save['colors'] = colors
            mesh_to_save['RGB'] = colors

            # 生成保存路径
            output_folder = get_output_folder()
            base_name = os.path.splitext(os.path.basename(self.current_3d_model_path))[0]
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_filename = f"{base_name}_colored_{timestamp}.glb"
            save_path = os.path.join(output_folder, save_filename)

            # 保存为GLB文件
            try:
                # 尝试使用PyVista保存
                mesh_to_save.save(save_path)
                QMessageBox.information(
                    self,
                    "保存成功",
                    f"3D模型已保存到:\n{save_path}\n\n"
                    f"文件大小: {os.path.getsize(save_path) / (1024 * 1024):.2f} MB"
                )
                print(f"3D模型已保存到: {save_path}")
            except Exception as e:
                # 如果GLB保存失败，尝试保存为PLY格式
                print(f"保存GLB失败，尝试保存为PLY格式: {e}")
                save_path_ply = save_path.replace('.glb', '.ply')
                mesh_to_save.save(save_path_ply)
                QMessageBox.information(
                    self,
                    "保存成功",
                    f"3D模型已保存为PLY格式:\n{save_path_ply}\n\n"
                    f"文件大小: {os.path.getsize(save_path_ply) / (1024 * 1024):.2f} MB\n\n"
                    f"注意: 由于GLB格式限制，已保存为PLY格式"
                )
                print(f"3D模型已保存为PLY格式: {save_path_ply}")

        except Exception as e:
            error_msg = f"保存3D模型失败: {str(e)}"
            print(error_msg)
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self, "保存失败", error_msg)

    def on_open_3d_external(self):
        """使用外部程序打开3D模型"""
        if not self.current_3d_model_path or not os.path.exists(self.current_3d_model_path):
            # 如果没有当前模型，先让用户选择
            self.load_3d_model()
            if not self.current_3d_model_path:
                return

        # 使用系统默认程序打开
        try:
            if platform.system() == "Windows":
                os.startfile(self.current_3d_model_path)
            elif platform.system() == "Darwin":  # macOS
                subprocess.Popen(["open", self.current_3d_model_path])
            else:  # Linux
                subprocess.Popen(["xdg-open", self.current_3d_model_path])
        except Exception as e:
            print(f"打开3D模型失败: {e}")

    def on_open_3d_window(self):
        """在独立窗口中打开3D模型查看器"""
        if not self.current_3d_model_path or not os.path.exists(self.current_3d_model_path):
            # 如果没有当前模型，先让用户选择
            self.load_3d_model()
            if not self.current_3d_model_path:
                return

        # 创建并显示独立窗口
        if not hasattr(self, 'model_3d_window') or not self.model_3d_window or not self.model_3d_window.isVisible():
            self.model_3d_window = Model3DViewerDialog(self.current_3d_model_path, self)
            self.model_3d_window.setWindowFlags(Qt.Window | Qt.WindowCloseButtonHint | Qt.WindowMinMaxButtonsHint)
            self.model_3d_window.show()
            self.model_3d_window.raise_()
            self.model_3d_window.activateWindow()
        else:
            # 如果窗口已存在，将其置于最前
            self.model_3d_window.raise_()
            self.model_3d_window.activateWindow()

    def load_presets(self):
        """从文件加载预设Prompt"""
        if os.path.exists(self.presets_file):
            try:
                with open(self.presets_file, 'r', encoding='utf-8') as f:
                    self.prompt_presets = json.load(f)
                # 如果预设文件已存在，检查是否有耳机预设，如果没有则添加
                if "Headphones" not in self.prompt_presets:
                    self.prompt_presets[
                        "Headphones"] = "headphones, animated style, dynamic pose, white background, no lighting, high quality, detailed, 3D render style"
                    self.save_presets()
            except Exception as e:
                print(f"加载预设失败: {e}")
                self.prompt_presets = {}
                # 加载失败时，创建默认预设
                self.prompt_presets = {
                    "Default Example": "A beautiful landscape with mountains and lakes, cinematic lighting, high quality",
                    "Portrait": "Portrait of a person, professional photography, detailed face, soft lighting",
                    "Sci-Fi Scene": "Futuristic cityscape, neon lights, cyberpunk style, detailed, 4k",
                    "Headphones": "headphones, animated style, dynamic pose, white background, no lighting, high quality, detailed, 3D render style",
                }
                self.save_presets()
        else:
            # 添加一些默认预设
            self.prompt_presets = {
                "Default Example": "A beautiful landscape with mountains and lakes, cinematic lighting, high quality",
                "Portrait": "Portrait of a person, professional photography, detailed face, soft lighting",
                "Sci-Fi Scene": "Futuristic cityscape, neon lights, cyberpunk style, detailed, 4k",
                "Headphones": "headphones, animated style, dynamic pose, white background, no lighting, high quality, detailed, 3D render style",
            }
            self.save_presets()

    def save_presets(self):
        """保存预设Prompt到文件"""
        try:
            with open(self.presets_file, 'w', encoding='utf-8') as f:
                json.dump(self.prompt_presets, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"保存预设失败: {e}")

    def load_config(self):
        """加载保存的配置"""
        if not os.path.exists(self.config_file):
            return

        try:
            # 暂时断开信号，避免加载时触发保存
            self.seed_spin.blockSignals(True)
            self.steps_spin.blockSignals(True)
            self.cfg_spin.blockSignals(True)
            self.num_images_spin.blockSignals(True)
            self.resolution_3d_slider.blockSignals(True)
            # 已移除use_sketch_checkbox，不再需要
            self.loop_edit_enable_radio.blockSignals(True)
            self.loop_edit_disable_radio.blockSignals(True)
            if hasattr(self, 'image_width_slider'):
                self.image_width_slider.blockSignals(True)
                self.image_height_slider.blockSignals(True)

            with open(self.config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)

            # 加载生成参数
            if 'seed' in config:
                self.seed_spin.setValue(int(config['seed']))
            if 'steps' in config:
                self.steps_spin.setValue(int(config['steps']))
            if 'cfg' in config:
                self.cfg_spin.setValue(float(config['cfg']))
            if 'num_images' in config:
                self.num_images_spin.setValue(int(config['num_images']))

            # 加载提示词（不使用信号，直接设置）
            if 'prompt' in config:
                self.prompt_edit.blockSignals(True)
                self.prompt_edit.setPlainText(config['prompt'])
                self.prompt_edit.blockSignals(False)

            # 已移除use_sketch_checkbox，自动检测画布内容，不再需要加载配置

            # 加载循环修改选项
            if 'enable_loop_edit' in config:
                if bool(config['enable_loop_edit']):
                    self.loop_edit_enable_radio.setChecked(True)
                else:
                    self.loop_edit_disable_radio.setChecked(True)

            # 加载3D参数
            if 'resolution' in config:
                resolution_value = int(config['resolution'])
                self.resolution_3d_slider.setValue(resolution_value)
                self.on_resolution_3d_slider_changed(resolution_value)

            # 加载3D模型显示设置（颜色、光照等）
            if 'model_3d_color' in config:
                color_list = config['model_3d_color']
                if isinstance(color_list, list) and len(color_list) >= 3:
                    self.model_3d_color = tuple(color_list[:3])
                    # 更新颜色按钮显示
                    if hasattr(self, 'model_3d_color_button'):
                        color = QColor(int(self.model_3d_color[0] * 255), int(self.model_3d_color[1] * 255),
                                       int(self.model_3d_color[2] * 255))
                        self.model_3d_color_button.setStyleSheet(
                            f"background-color: rgb({color.red()}, {color.green()}, {color.blue()}); "
                            f"border: 2px solid #d1d5db; border-radius: 4px;"
                        )
                    # 如果3D模型已加载，更新显示
                    if hasattr(self, 'model_3d_mesh') and self.model_3d_mesh is not None:
                        self.update_model_3d_display()
            if 'model_3d_show_edges' in config:
                self.model_3d_show_edges = bool(config['model_3d_show_edges'])
                if hasattr(self, 'model_3d_edges_checkbox'):
                    self.model_3d_edges_checkbox.blockSignals(True)
                    self.model_3d_edges_checkbox.setChecked(self.model_3d_show_edges)
                    self.model_3d_edges_checkbox.blockSignals(False)
            if 'model_3d_smooth_shading' in config:
                self.model_3d_smooth_shading = bool(config['model_3d_smooth_shading'])
                if hasattr(self, 'model_3d_smooth_checkbox'):
                    self.model_3d_smooth_checkbox.blockSignals(True)
                    self.model_3d_smooth_checkbox.setChecked(self.model_3d_smooth_shading)
                    self.model_3d_smooth_checkbox.blockSignals(False)
            if 'model_3d_ambient_light' in config:
                self.model_3d_ambient_light = float(config['model_3d_ambient_light'])
                if hasattr(self, 'model_3d_ambient_slider'):
                    self.model_3d_ambient_slider.blockSignals(True)
                    self.model_3d_ambient_slider.setValue(int(self.model_3d_ambient_light * 100))
                    self.model_3d_ambient_slider.blockSignals(False)
                if hasattr(self, 'model_3d_ambient_label'):
                    self.model_3d_ambient_label.setText(f"{self.model_3d_ambient_light:.2f}")
            if 'model_3d_diffuse_light' in config:
                self.model_3d_diffuse_light = float(config['model_3d_diffuse_light'])
                if hasattr(self, 'model_3d_diffuse_slider'):
                    self.model_3d_diffuse_slider.blockSignals(True)
                    self.model_3d_diffuse_slider.setValue(int(self.model_3d_diffuse_light * 100))
                    self.model_3d_diffuse_slider.blockSignals(False)
                if hasattr(self, 'model_3d_diffuse_label'):
                    self.model_3d_diffuse_label.setText(f"{self.model_3d_diffuse_light:.2f}")

            # 加载图像分辨率设置
            # 加载图像分辨率设置（使用统一的分辨率滑块）
            if hasattr(self, 'image_resolution_slider'):
                if 'image_width' in config:
                    width_value = int(config['image_width'])
                    self.image_resolution_slider.setValue(width_value)
                    self.on_resolution_slider_changed(width_value)
                elif 'image_height' in config:
                    height_value = int(config['image_height'])
                    self.image_resolution_slider.setValue(height_value)
                    self.on_resolution_slider_changed(height_value)
                if 'lock_aspect_ratio' in config:
                    self.lock_aspect_ratio.setChecked(bool(config['lock_aspect_ratio']))
                # 宽高比总是1:1（正方形）
                self._last_aspect_ratio = 1.0

            # 加载3D模型显示设置（颜色、光照等）
            if 'model_3d_color' in config:
                color_list = config['model_3d_color']
                if isinstance(color_list, list) and len(color_list) == 3:
                    self.model_3d_color = tuple(color_list)
                    # 更新颜色按钮显示
                    if hasattr(self, 'model_3d_color_button'):
                        color = QColor(int(self.model_3d_color[0] * 255), int(self.model_3d_color[1] * 255),
                                       int(self.model_3d_color[2] * 255))
                        self.model_3d_color_button.setStyleSheet(
                            f"background-color: rgb({color.red()}, {color.green()}, {color.blue()}); "
                            f"border: 2px solid #d1d5db; border-radius: 4px;"
                        )
            if 'model_3d_show_edges' in config:
                self.model_3d_show_edges = bool(config['model_3d_show_edges'])
                if hasattr(self, 'model_3d_edges_checkbox'):
                    self.model_3d_edges_checkbox.blockSignals(True)
                    self.model_3d_edges_checkbox.setChecked(self.model_3d_show_edges)
                    self.model_3d_edges_checkbox.blockSignals(False)
            if 'model_3d_smooth_shading' in config:
                self.model_3d_smooth_shading = bool(config['model_3d_smooth_shading'])
                if hasattr(self, 'model_3d_smooth_checkbox'):
                    self.model_3d_smooth_checkbox.blockSignals(True)
                    self.model_3d_smooth_checkbox.setChecked(self.model_3d_smooth_shading)
                    self.model_3d_smooth_checkbox.blockSignals(False)
            if 'model_3d_ambient_light' in config:
                self.model_3d_ambient_light = float(config['model_3d_ambient_light'])
                if hasattr(self, 'model_3d_ambient_slider'):
                    self.model_3d_ambient_slider.blockSignals(True)
                    self.model_3d_ambient_slider.setValue(int(self.model_3d_ambient_light * 100))
                    self.model_3d_ambient_slider.blockSignals(False)
                if hasattr(self, 'model_3d_ambient_label'):
                    self.model_3d_ambient_label.setText(f"{self.model_3d_ambient_light:.2f}")
            if 'model_3d_diffuse_light' in config:
                self.model_3d_diffuse_light = float(config['model_3d_diffuse_light'])
                if hasattr(self, 'model_3d_diffuse_slider'):
                    self.model_3d_diffuse_slider.blockSignals(True)
                    self.model_3d_diffuse_slider.setValue(int(self.model_3d_diffuse_light * 100))
                    self.model_3d_diffuse_slider.blockSignals(False)
                if hasattr(self, 'model_3d_diffuse_label'):
                    self.model_3d_diffuse_label.setText(f"{self.model_3d_diffuse_light:.2f}")

            # 加载窗口状态
            if 'window_geometry' in config:
                geometry = config['window_geometry']
                if 'x' in geometry and 'y' in geometry and 'width' in geometry and 'height' in geometry:
                    self.setGeometry(
                        int(geometry['x']),
                        int(geometry['y']),
                        int(geometry['width']),
                        int(geometry['height'])
                    )

            # 加载分割器布局状态
            if 'splitter_sizes' in config and hasattr(self, 'main_splitter'):
                splitter_sizes = config['splitter_sizes']
                if isinstance(splitter_sizes, list):
                    # 延迟设置，确保窗口已经显示
                    if QT_VERSION == "PySide6":
                        from PySide6.QtCore import QTimer
                    else:
                        from PyQt5.QtCore import QTimer
                    if len(splitter_sizes) == 3:
                        # 三列布局：左侧、中间、右侧
                        QTimer.singleShot(100, lambda: self.main_splitter.setSizes([
                            int(splitter_sizes[0]),
                            int(splitter_sizes[1] + splitter_sizes[2])  # 中间+右侧的总宽度
                        ]))
                        QTimer.singleShot(150, lambda: self.right_splitter.setSizes([
                            int(splitter_sizes[1]),
                            int(splitter_sizes[2])
                        ]))
                    elif len(splitter_sizes) == 2:
                        # 兼容旧的两列布局
                        QTimer.singleShot(100, lambda: self.main_splitter.setSizes([
                            int(splitter_sizes[0]),
                            int(splitter_sizes[1])
                        ]))

            # 提示词可编辑，默认值为"耳机,白色背景,不要有任何光线"

            # 加载3D模型路径（但不自动显示，用户需要手动加载）
            # 注释掉自动加载，启动时不显示3D模型
            # if 'current_3d_model_path' in config:
            #     model_path = config['current_3d_model_path']
            #     if model_path and os.path.exists(model_path):
            #         self.current_3d_model_path = model_path
            #         # 延迟加载，确保界面已完全初始化
            #         if QT_VERSION == "PySide6":
            #             from PySide6.QtCore import QTimer
            #         else:
            #             from PyQt5.QtCore import QTimer
            #         QTimer.singleShot(500, lambda: self.load_3d_model(model_path))

            # 只保存路径，不自动加载显示
            if 'current_3d_model_path' in config:
                model_path = config['current_3d_model_path']
                if model_path and os.path.exists(model_path):
                    self.current_3d_model_path = model_path
                    # 不自动加载，用户需要手动点击"加载3D模型"按钮

            # 恢复信号连接
            self.seed_spin.blockSignals(False)
            self.steps_spin.blockSignals(False)
            self.cfg_spin.blockSignals(False)
            self.num_images_spin.blockSignals(False)
            self.resolution_3d_slider.blockSignals(False)
            # 已移除use_sketch_checkbox，不再需要
            self.loop_edit_enable_radio.blockSignals(False)
            self.loop_edit_disable_radio.blockSignals(False)
            if hasattr(self, 'image_width_slider'):
                self.image_width_slider.blockSignals(False)
                self.image_height_slider.blockSignals(False)
        except Exception as e:
            print(f"加载配置失败: {e}")
            # 确保信号恢复
            self.seed_spin.blockSignals(False)
            self.steps_spin.blockSignals(False)
            self.cfg_spin.blockSignals(False)
            self.num_images_spin.blockSignals(False)
            self.resolution_3d_slider.blockSignals(False)
            # 已移除use_sketch_checkbox，不再需要
            self.loop_edit_enable_radio.blockSignals(False)
            self.loop_edit_disable_radio.blockSignals(False)
            if hasattr(self, 'image_width_slider'):
                self.image_width_slider.blockSignals(False)
                self.image_height_slider.blockSignals(False)

    def save_config(self):
        """保存当前配置到文件"""
        try:
            config = {
                'seed': int(self.seed_spin.value()),
                'steps': int(self.steps_spin.value()),
                'cfg': float(self.cfg_spin.value()),
                'num_images': int(self.num_images_spin.value()),
                'prompt': self.prompt_edit.toPlainText(),
                # 已移除use_sketch_checkbox，自动检测画布内容
                'enable_loop_edit': self.loop_edit_enable_radio.isChecked(),  # 循环修改选项
                'resolution': int(self.resolution_3d_slider.value()),
                'window_geometry': {
                    'x': self.geometry().x(),
                    'y': self.geometry().y(),
                    'width': self.geometry().width(),
                    'height': self.geometry().height(),
                },
            }

            # 保存3D模型路径（如果存在）
            if hasattr(self, 'current_3d_model_path') and self.current_3d_model_path and os.path.exists(
                    self.current_3d_model_path):
                config['current_3d_model_path'] = self.current_3d_model_path

            # 保存3D模型显示设置（颜色、光照等）
            if hasattr(self, 'model_3d_color'):
                config['model_3d_color'] = list(self.model_3d_color)  # 转换为列表以便JSON序列化
            if hasattr(self, 'model_3d_show_edges'):
                config['model_3d_show_edges'] = self.model_3d_show_edges
            if hasattr(self, 'model_3d_smooth_shading'):
                config['model_3d_smooth_shading'] = self.model_3d_smooth_shading
            if hasattr(self, 'model_3d_ambient_light'):
                config['model_3d_ambient_light'] = self.model_3d_ambient_light
            if hasattr(self, 'model_3d_diffuse_light'):
                config['model_3d_diffuse_light'] = self.model_3d_diffuse_light

            # 保存图像分辨率设置（使用统一的分辨率滑块）
            if hasattr(self, 'image_resolution_slider'):
                resolution = int(self.image_resolution_slider.value())
                config['image_width'] = resolution
                config['image_height'] = resolution
                config['lock_aspect_ratio'] = self.lock_aspect_ratio.isChecked()

            # 保存分割器布局状态（三列：左侧、中间、右侧）
            if hasattr(self, 'main_splitter') and hasattr(self, 'right_splitter'):
                main_sizes = self.main_splitter.sizes()
                right_sizes = self.right_splitter.sizes()
                # 保存为三列：[左侧, 中间, 右侧]
                config['splitter_sizes'] = [
                    main_sizes[0],  # 左侧
                    right_sizes[0],  # 中间
                    right_sizes[1]  # 右侧
                ]

            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(config, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"保存配置失败: {e}")

    def _connect_config_signals(self):
        """连接参数改变信号，自动保存配置"""
        # 参数改变时自动保存
        self.seed_spin.valueChanged.connect(self.save_config)
        self.steps_spin.valueChanged.connect(self.save_config)
        self.cfg_spin.valueChanged.connect(self.save_config)
        self.num_images_spin.valueChanged.connect(self.save_config)
        self.resolution_3d_slider.valueChanged.connect(self.on_resolution_3d_slider_changed)
        self.resolution_3d_slider.valueChanged.connect(self.save_config)

        # 提示词改变时保存（使用定时器避免频繁保存）
        if QT_VERSION == "PySide6":
            from PySide6.QtCore import QTimer
        else:
            from PyQt5.QtCore import QTimer
        self.prompt_save_timer = QTimer()
        self.prompt_save_timer.setSingleShot(True)
        self.prompt_save_timer.timeout.connect(self.save_config)
        self.prompt_edit.textChanged.connect(lambda: self.prompt_save_timer.start(1000))  # 1秒后保存

        # 手绘图选项改变时保存
        # 已移除use_sketch_checkbox，自动检测画布内容

        # 循环修改选项改变时保存
        self.loop_edit_enable_radio.toggled.connect(self.save_config)
        self.loop_edit_disable_radio.toggled.connect(self.save_config)

        # 图像分辨率改变时保存（使用统一的分辨率滑块）
        if hasattr(self, 'image_resolution_slider'):
            self.image_resolution_slider.valueChanged.connect(self.save_config)
            self.lock_aspect_ratio.stateChanged.connect(self.save_config)

    def closeEvent(self, event):
        """窗口关闭时保存配置"""
        self.save_config()
        super().closeEvent(event)

    def update_preset_combo(self):
        """更新预设下拉框"""
        self.preset_combo.blockSignals(True)
        current_text = self.preset_combo.currentText()
        self.preset_combo.clear()
        self.preset_combo.addItem("-- 选择预设 --")
        for name in sorted(self.prompt_presets.keys()):
            self.preset_combo.addItem(name)
        # 恢复之前的选择
        index = self.preset_combo.findText(current_text)
        if index >= 0:
            self.preset_combo.setCurrentIndex(index)
        else:
            self.preset_combo.setCurrentIndex(0)
        self.preset_combo.blockSignals(False)

    def on_preset_selected(self, text):
        """当选择预设时"""
        if text and text != "-- 选择预设 --" and text in self.prompt_presets:
            self.prompt_edit.setPlainText(self.prompt_presets[text])

    def on_save_preset(self):
        """保存当前Prompt为预设"""
        prompt_text = self.prompt_edit.toPlainText().strip()
        if not prompt_text:
            QMessageBox.warning(self, "Tip", "Please enter Prompt content first")
            return

        name, ok = QInputDialog.getText(
            self, "Save Preset", "Please enter preset name:",
            text="New Preset"
        )

        if ok and name:
            name = name.strip()
            if not name:
                QMessageBox.warning(self, "Tip", "Preset name cannot be empty")
                return

            self.prompt_presets[name] = prompt_text
            self.save_presets()
            self.update_preset_combo()
            # 选中新添加的预设
            index = self.preset_combo.findText(name)
            if index >= 0:
                self.preset_combo.setCurrentIndex(index)

    def on_manage_presets(self):
        """管理预设Prompt"""
        if not self.prompt_presets:
            QMessageBox.information(self, "Tip", "No preset Prompts available")
            return

        # 创建管理对话框
        dialog = QDialog(self)
        dialog.setWindowTitle("Manage Preset Prompts")
        dialog.setMinimumSize(500, 400)
        dialog.setStyleSheet("""
            QWidget {
                background-color: #ffffff;
            }
            QPushButton {
                background-color: #6366f1;
                color: white;
                border: none;
                border-radius: 6px;
                padding: 8px 16px;
                font-weight: 600;
            }
            QPushButton:hover {
                background-color: #4f46e5;
            }
            QListWidget {
                border: 2px solid #e5e7eb;
                border-radius: 8px;
                padding: 5px;
            }
            QListWidget::item {
                padding: 8px;
                border-bottom: 1px solid #f3f4f6;
            }
            QListWidget::item:selected {
                background-color: #eef2ff;
                color: #6366f1;
            }
        """)

        layout = QVBoxLayout(dialog)

        list_widget = QListWidget()
        for name in sorted(self.prompt_presets.keys()):
            item = QListWidgetItem(name)
            item.setData(Qt.UserRole, name)
            list_widget.addItem(item)

        layout.addWidget(QLabel("Preset List:"))
        layout.addWidget(list_widget)

        btn_layout = QHBoxLayout()

        edit_btn = QPushButton("Edit")

        def on_edit():
            item = list_widget.currentItem()
            if item:
                name = item.data(Qt.UserRole)
                new_text, ok = QInputDialog.getMultiLineText(
                    dialog, "Edit Preset", f"Edit preset '{name}':",
                    text=self.prompt_presets[name]
                )
                if ok and new_text.strip():
                    self.prompt_presets[name] = new_text.strip()
                    self.save_presets()
                    QMessageBox.information(dialog, "Success", "Preset updated")

        edit_btn.clicked.connect(on_edit)

        delete_btn = QPushButton("Delete")

        def on_delete():
            item = list_widget.currentItem()
            if item:
                name = item.data(Qt.UserRole)
                reply = QMessageBox.question(
                    dialog, "Confirm Delete", f"Are you sure you want to delete preset '{name}'?",
                    QMessageBox.Yes | QMessageBox.No
                )
                if reply == QMessageBox.Yes:
                    del self.prompt_presets[name]
                    self.save_presets()
                    self.update_preset_combo()
                    list_widget.takeItem(list_widget.currentRow())
                    if list_widget.count() == 0:
                        dialog.close()
                    QMessageBox.information(dialog, "Success", "Preset deleted")

        delete_btn.setStyleSheet("background-color: #ef4444;")
        delete_btn.clicked.connect(on_delete)

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dialog.close)

        btn_layout.addWidget(edit_btn)
        btn_layout.addWidget(delete_btn)
        btn_layout.addStretch()
        btn_layout.addWidget(close_btn)

        layout.addLayout(btn_layout)

        dialog.exec_()


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.showMaximized()  # 启动时最大化窗口
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
