# Development

## Environment

```bat
python -m venv venv
call venv\Scripts\activate.bat
pip install -r requirements.txt
pip install -e .
```

## Run the GUI (development)

```bat
call venv\Scripts\activate.bat
python GUI.py
```

## Project layout (high level)

- `src/draft3d/`: core logic (ComfyUI client + workflow builders + operations)
- `src/draft3d_gui/`: refactored GUI components
- `GUI.py`: legacy GUI entry point (still used as main entry)

