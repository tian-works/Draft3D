# Installation

This page contains more detailed setup steps than the project homepage (`README.md`).

## Prerequisites

- Windows 10+ (Linux/macOS can work with script adaptations)
- Python **3.10+**
- NVIDIA GPU recommended (CPU-only is possible but slower)

## Get the code

Recommended (includes ComfyUI submodule):

```bat
git clone --recurse-submodules https://github.com/tian-works/Draft3D.git
cd Draft3D
```

If you already cloned without submodules:

```bat
git submodule update --init --recursive
```

## One-click launch

```bat
RunAll.bat
```

The script will create a virtual environment, install dependencies, start ComfyUI, and launch the GUI.

## Manual launch (two terminals)

Terminal 1 (ComfyUI):

```bat
call venv\Scripts\activate.bat
cd ComfyUI
python main.py
```

Terminal 2 (GUI):

```bat
call venv\Scripts\activate.bat
cd ..
python GUI.py
```

## Next

Proceed to the ComfyUI setup page for model/custom-node requirements:

- [`docs/COMFYUI_SETUP.md`](COMFYUI_SETUP.md)

