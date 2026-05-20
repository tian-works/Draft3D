# Installation

This page contains more detailed setup steps than the project homepage (`README.md`).

## Prerequisites

- Windows 10/11, Ubuntu, or macOS
- Python **3.10+**
- NVIDIA GPU recommended (CPU-only is possible but slower)

## Cross-platform notes

Draft3D is implemented mainly in Python, and its core workflow does not rely on Windows-specific interfaces. Therefore, the software has a cross-platform implementation basis. In the original submission, we mainly reported validation on Windows 10 and Windows 11 because these were the complete testing environments available at that time. In response to the reviewer comment, we further tested Draft3D on Ubuntu and macOS. The additional tests confirmed that, after installing the required dependencies, Draft3D can be launched and its main workflow can be executed on Windows, Ubuntu, and macOS.

Nevertheless, platform-specific differences in GPU drivers, CUDA or Metal support, deep learning library versions, GUI environments, local path conventions, and external model components may still affect the deployment experience. Please check environment-specific notes when troubleshooting.

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

The script will create a virtual environment (if missing), install dependencies on first run, start ComfyUI, and launch the GUI.

On Ubuntu/macOS (bash):

```bash
bash RunAll.sh
```

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

