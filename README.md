# Draft3D

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

Draft3D is an open-source local software platform for **sketch-conditioned multi-stage 2D-to-3D concept generation**, built on top of ComfyUI.

The Python-based implementation provides a cross-platform implementation basis. After installing the required dependencies, the software can run on Windows, Ubuntu, and macOS. The main workflow has been additionally tested on Ubuntu and macOS. Platform-specific dependency differences (GPU drivers, CUDA/Metal support, GUI stack, and model components) may still affect deployment experience.

## Key Features

- **Sketch → image → 3D**: sketch-conditioned image generation (Z-Image-Turbo) and image-to-3D generation (Hunyuan3D)
- **One-click launch**: platform-specific scripts start ComfyUI backend + Draft3D GUI together
- **Interactive preview**: image gallery + 3D preview for inspecting results

## Quick Start

The following steps are organized into two usage paths:

- **General users (recommended)**: directly use the one-click launch scripts (Windows: `RunAll.bat`; Ubuntu/macOS: `RunAll.sh`)
- **Developers / research reproducibility**: clone via git (with submodules), then run the one-click script for your platform

### Method A: One-click launch for general users (recommended)

1. **Get the code**
   - Method 1: on the GitHub page, click `Code -> Download ZIP`, then extract to an **English-only path** (avoid non-ASCII characters, spaces, and overly deep directories)
   - Method 2: use git clone as shown in "Method B" below

2. **One-click start (Windows)**
   - Double-click to run:

```bat
RunAll.bat
```

Expected behavior on first run:

- Automatically create the virtual environment `venv/`
- Automatically install Draft3D and ComfyUI dependencies (the first run may take several minutes depending on network and disk speed)
- Automatically launch ComfyUI (default `http://127.0.0.1:8188`) and then start the Draft3D GUI

If the GUI opens normally and your browser can access `http://127.0.0.1:8188`, the one-click launch has succeeded.

**Model and node dependencies:** Draft3D delegates image generation and 3D reconstruction to your local ComfyUI instance, so the required **model weight files** must be placed in `ComfyUI/models/` (and required **custom nodes** must be installed for the workflows). The one-click scripts only install Python dependencies and start services; they **do not automatically download large models**. Before your first full Sketch -> Image -> 3D run, prepare models and nodes according to [`docs/COMFYUI_SETUP.md`](docs/COMFYUI_SETUP.md). If generation fails or reports missing models, troubleshoot using that document and [`docs/TROUBLESHOOTING.md`](docs/TROUBLESHOOTING.md).

On Ubuntu/macOS (bash):

```bash
bash RunAll.sh
```

### Method B: Developer clone (with submodules)

1. **Clone with submodules**:

```bat
git clone --recurse-submodules https://github.com/tian-works/Draft3D.git
cd Draft3D
```

If you already cloned without submodules:

```bat
git submodule update --init --recursive
```

2. **Create & activate a virtual environment**

```bat
python -m venv venv
venv\Scripts\activate
```

3. **Install Python dependencies**

```bat
python -m pip install -r requirements.txt
```

4. **Prepare ComfyUI models & custom nodes**

- **Model files** (place under `ComfyUI/models/` following ComfyUI conventions):
  - Hunyuan3D: `hunyuan_3d_v2.1.safetensors`
  - Z-Image-Turbo: `z_image_turbo_bf16.safetensors`, `ae.safetensors`, `qwen_3_4b.safetensors`, `Z-Image-Turbo-Fun-Controlnet-Union.safetensors`, `lumina2.safetensors`
- **Custom nodes**: install the node packs needed by the workflows. Using **ComfyUI-Manager** is the easiest way; if ComfyUI reports “missing node / unknown class_type”, install the missing custom nodes.

5. **Run (one-click launch recommended)**:

Windows:

```bat
RunAll.bat
```

Ubuntu/macOS:

```bash
bash RunAll.sh
```

## Testability and Main-Feature Reproducibility

This repository provides **locally executable test entry points** that can reproduce the main functionality directly (no extra coding required). The section below provides minimal reproducible steps, sample inputs, and expected outputs/artifact locations for straightforward verification by reviewers and readers.

### Test Entry Points (Windows / Ubuntu / macOS)

- **Recommended: one-click launch (Backend + GUI)**:

Windows:

```bat
RunAll.bat
```

Ubuntu/macOS:

```bash
bash RunAll.sh
```

Expected behavior (normal case):

- After the ComfyUI backend starts, it attempts to open `http://127.0.0.1:8188` in a browser
- The Draft3D GUI starts automatically and connects to the local ComfyUI backend (default port: 8188)

- **Manual launch (for troubleshooting)**:
  - Windows:
    - Terminal 1: launch ComfyUI (run `python main.py` under `ComfyUI/`, or run `LaunchComfyUI.bat`)
    - Terminal 2: launch GUI (run `LaunchGUI.bat` or `python GUI.py`)
  - Ubuntu/macOS:
    - Terminal 1: launch ComfyUI (run `bash LaunchComfyUI.sh` in the project root, or run `python main.py` under `ComfyUI/`)
    - Terminal 2: launch GUI (run `bash LaunchGUI.sh` in the project root, or run `python GUI.py`)

For more detailed installation/startup instructions, see [`docs/INSTALL.md`](docs/INSTALL.md).

### Minimal Reproducible Test Case (covers core functionality)

This test case covers Draft3D's primary workflow: **Sketch -> Image -> 3D**.

1. **Launch the software**
   - Windows: run `RunAll.bat`
   - Ubuntu/macOS: run `bash RunAll.sh`
   - Confirm ComfyUI is reachable by opening `http://127.0.0.1:8188` in your browser

2. **Sketch input (example input)**
   - Draw a simple contour on the GUI canvas (for example: headphone outer contour + headband)
   - The software automatically detects whether the canvas is non-empty; when non-empty, the sketch is automatically used as conditioning input for image generation (no extra switch required)
   - You can also save the sketch as a local PNG (the GUI provides a `Save Sketch` feature) to archive input samples

3. **Generate image (Image Generation)**
   - Enter a simple prompt in the prompt input box, for example:
     - `a studio product photo of a headphone, high detail, white background`
   - Recommended quick validation settings (to reduce VRAM usage):
     - width/height: 512x512
     - steps: 4 (or lower)
     - batch: 1
   - Click generate and wait for results to appear in the image gallery

4. **Generate 3D (Image-to-3D)**
   - Select one generated image in the gallery
   - Click **Generate 3D**, then wait for the 3D mesh to be generated and visualized in the 3D preview panel

### Expected Outputs

- **Output directory**: generated images and 3D meshes are saved under `generated_images/` in the repository root (organized by date).
- **3D result validation**: after one successful 3D generation, you should find a `.glb` file under `generated_images/`, typically with prefix `ComfyUI_Hunyuan3D` (see Quick verification in [`docs/COMFYUI_SETUP.md`](docs/COMFYUI_SETUP.md)).

If these artifacts are not produced, the most common causes are missing model files, missing ComfyUI custom nodes, or VRAM configuration issues.

### FAQ and Troubleshooting

- **One-click launch has no response / closes immediately**: run platform scripts from a terminal first to inspect errors (Windows: `RunAll.bat`; Ubuntu/macOS: `bash RunAll.sh`). You can also run backend and GUI launch scripts separately to isolate issues (Windows: `LaunchComfyUI.bat` / `LaunchGUI.bat`; Ubuntu/macOS: `bash LaunchComfyUI.sh` / `bash LaunchGUI.sh`).
- **Cannot connect to backend / ComfyUI connection errors**: confirm ComfyUI is running and reachable at `http://127.0.0.1:8188`
- **Missing nodes / unknown `class_type`**: install missing custom nodes via ComfyUI-Manager, then restart ComfyUI
- **Model not found / filename mismatch**: verify model filenames and locations (see [`docs/COMFYUI_SETUP.md`](docs/COMFYUI_SETUP.md))
- **Out of memory / OOM**: lower resolution, steps, and batch first, and close other GPU-intensive programs

For a more complete troubleshooting checklist, see [`docs/TROUBLESHOOTING.md`](docs/TROUBLESHOOTING.md).

## Project Structure

Core project layout (excluding third-party `ComfyUI/` internals):

- `GUI.py`: desktop entry point and UI orchestration
- `src/draft3d/operations.py`: core generation operations (image/edit/remove-bg/3D)
- `src/draft3d/io_utils.py`: output path and cross-platform folder opening utilities
- `src/draft3d/config.py`: runtime path/config helpers
- `RunAll.bat` / `RunAll.sh`: one-click launcher for backend + GUI
- `LaunchComfyUI.bat` / `LaunchComfyUI.sh`: backend-only launcher
- `LaunchGUI.bat` / `LaunchGUI.sh`: GUI-only launcher

## Workflow Overview

![Draft3D GUI workflow overview](docs/assets/images/diagram.png)

1. **Sketch input**
2. **Image gallery selection**
3. **Result confirmation**
4. **3D generation & preview**

## Dependencies

Installed by the setup script (see `requirements.txt`):

- `PySide6` (primary) / `PyQt5` (fallback)
- `numpy`, `opencv-python`, `requests`
- `pyvista`, `pyvistaqt`, `vtk`

## Documentation

- **Install & setup**: [`docs/INSTALL.md`](docs/INSTALL.md)
- **ComfyUI setup (models & custom nodes)**: [`docs/COMFYUI_SETUP.md`](docs/COMFYUI_SETUP.md)
- **User guide (GUI workflow)**: [`docs/USAGE.md`](docs/USAGE.md)
- **Troubleshooting**: [`docs/TROUBLESHOOTING.md`](docs/TROUBLESHOOTING.md)
- **Development**: [`docs/DEVELOPMENT.md`](docs/DEVELOPMENT.md)

## How to Cite

**BibTeX:**

```bibtex
@software{draft3d_2026,
  author = {Song, Jiatian and Zhang, Jianmin},
  title  = {{Draft3D}: An open-source local software platform for sketch-conditioned multi-stage 2D-to-3D concept generation},
  year   = {2026},
  url    = {https://github.com/tian-works/Draft3D},
  note   = {Research software repository}
}
```

## License

MIT License. See [LICENSE.txt](LICENSE.txt).

## Acknowledgments

- Built on top of [ComfyUI](https://github.com/comfyanonymous/ComfyUI)
