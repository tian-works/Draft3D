# ComfyUI Setup (Models & Custom Nodes)

Draft3D uses ComfyUI as the backend and expects specific workflows to run successfully.

## Required model files

Place model files under `ComfyUI/models/` according to ComfyUI conventions (exact subfolders depend on the model type and your ComfyUI setup).

- **Hunyuan3D (image-to-3D)**:
  - `hunyuan_3d_v2.1.safetensors`

- **Z-Image-Turbo (sketch-conditioned image generation)**:
  - `z_image_turbo_bf16.safetensors`
  - `ae.safetensors`
  - `qwen_3_4b.safetensors`
  - `Z-Image-Turbo-Fun-Controlnet-Union.safetensors`
  - `lumina2.safetensors`

## Required custom nodes

Draft3D workflows reference multiple ComfyUI node types (for example: AuraFlow-related nodes and Hunyuan3D v2 nodes such as `ModelSamplingAuraFlow`, `UNETLoader`, `ModelPatchLoader`, `EmptyLatentHunyuan3Dv2`, `VAEDecodeHunyuan3D`, `SaveGLB`, etc.).

Because custom-node packs can vary, the most robust approach is:

1. Install **ComfyUI-Manager**
2. Start ComfyUI once
3. If ComfyUI reports **“missing node”** or **unknown `class_type`**, install the missing custom nodes through the manager

## Quick verification

1. Start ComfyUI and confirm it launches without errors.
2. Launch Draft3D GUI and generate a small image (low resolution, few steps).
3. Run **Generate 3D** once and confirm a `.glb` appears under `generated_images/` with prefix `ComfyUI_Hunyuan3D`.

