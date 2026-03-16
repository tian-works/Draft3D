# Draft3D

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

Draft3D is an open-source local software platform for **sketch-conditioned multi-stage 2D-to-3D concept generation**, built on top of ComfyUI.

## Key Features

- **Sketch → image → 3D**: sketch-conditioned image generation (Z-Image-Turbo) and image-to-3D generation (Hunyuan3D)
- **One-click launch**: batch scripts start ComfyUI backend + Draft3D GUI together
- **Interactive preview**: image gallery + 3D preview for inspecting results

## Quick Start (Windows)

1. **Clone with submodules**:

```bat
git clone --recurse-submodules https://github.com/tian-works/Draft3D.git
cd Draft3D
```

If you already cloned without submodules:

```bat
git submodule update --init --recursive
```

2. **Prepare ComfyUI models & custom nodes**

- **Model files** (place under `ComfyUI/models/` following ComfyUI conventions):
  - Hunyuan3D: `hunyuan_3d_v2.1.safetensors`
  - Z-Image-Turbo: `z_image_turbo_bf16.safetensors`, `ae.safetensors`, `qwen_3_4b.safetensors`, `Z-Image-Turbo-Fun-Controlnet-Union.safetensors`, `lumina2.safetensors`
- **Custom nodes**: install the node packs needed by the workflows. Using **ComfyUI-Manager** is the easiest way; if ComfyUI reports “missing node / unknown class_type”, install the missing custom nodes.

3. **Run**:

```bat
RunAll.bat
```

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
