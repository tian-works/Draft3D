# Troubleshooting

## ComfyUI connection errors

- Confirm ComfyUI is running and reachable at `http://127.0.0.1:8188`.
- Confirm `ComfyUI/` exists and contains `main.py`.

## “Missing node” / unknown `class_type`

- Install **ComfyUI-Manager** and use it to install the missing custom nodes.
- Restart ComfyUI after installing new nodes.

## Model file not found / wrong filename

- Confirm the model file names match what Draft3D workflows reference (see [`docs/COMFYUI_SETUP.md`](COMFYUI_SETUP.md)).

## Out of memory (CUDA / VRAM)

- Lower image resolution and/or reduce batch size.
- Close other GPU-heavy applications.

