# Photo Colorizer

Neural-network photo colorization with reference-image colour transfer.

Upload black & white photos alongside a colour reference image. The app
colorizes using DDColor (or Zhang et al. ECCV 2016 as fallback), then blends
the output toward a palette extracted from the reference image using LAB
colour-space matching.

## Features

- **AI colorization** — DDColor (tiny) or SIGGRAPH17 / ECCV16 fallback
- **Reference palette** — k-means palette extraction from a user-provided colour image
- **LAB blending** — adjustable strength slider (0–100 %) to control colour influence
- **Region sampler** — pull colours from full image, top/bottom/left/right half, or centre crop
- **Batch processing** — upload multiple B&W photos against one reference
- **GPU accelerated** — CUDA 12.1 with automatic CPU fallback
- **Gradio UI** — side-by-side before/after preview, palette display, device info banner

## Quick Start (Windows)

```
cd photo_colorizer
setup.bat
```

The script creates a virtual environment, installs CUDA 12.1 PyTorch, and all
dependencies. After setup:

```
venv\Scripts\activate.bat
python app.py
```

Open http://localhost:7860 in your browser.

## Manual Setup

```bash
python -m venv venv
source venv/bin/activate          # Linux/macOS
# venv\Scripts\activate.bat       # Windows

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

# Optional: DDColor (falls back to colorizers if unavailable)
pip install ddcolor

python app.py
```

## GPU Setup — NVIDIA RTX 3070

The RTX 3070 uses the Ampere architecture with 8 GB VRAM and works with
CUDA 12.1. Make sure you have:

1. **NVIDIA driver ≥ 530** — download from https://www.nvidia.com/drivers
2. **CUDA Toolkit 12.1** (optional; PyTorch ships its own runtime)
3. **PyTorch with cu121** — installed via the index URL above

### Verify CUDA is working

```python
import torch
print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("VRAM:", torch.cuda.get_device_properties(0).total_mem // 1024**2, "MB")
```

Expected output:

```
PyTorch: 2.x.x+cu121
CUDA available: True
GPU: NVIDIA GeForce RTX 3070
VRAM: 8192 MB
```

### VRAM Notes

- Model loading + a single 1080p image stays well under 4 GB.
- Batch processing is sequential per image, so VRAM usage does not scale
  with the number of uploaded files.
- If you process very large images (>4K), peak VRAM may approach 6–7 GB.
  The code does not explicitly cap resolution, but the model internally
  resizes to 256×256 or 512×512 for inference and up-samples back, so the
  GPU cost is bounded by the model's internal resolution, not the input size.

## Project Structure

```
photo_colorizer/
├── app.py              # Gradio web interface
├── colorize_engine.py  # Model loading, device selection, inference
├── palette.py          # Palette extraction (k-means) and LAB blending
├── requirements.txt    # Python dependencies
├── setup.bat           # Windows one-click setup
├── output/             # Saved colorized images
└── README.md
```

## Supported Formats

- **Input:** JPG, JPEG, PNG
- **Output:** Same format as input (saved to `output/` folder)
