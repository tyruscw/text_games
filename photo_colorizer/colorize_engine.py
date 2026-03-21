"""
Core colorization engine.

Tries DDColor first, then falls back to Zhang et al. ECCV 2016 (colorizers).
Handles model loading, GPU/CPU selection, and single-image inference.
"""

import os
import warnings
import numpy as np
import cv2
import torch
from PIL import Image

# ---------------------------------------------------------------------------
# Device selection
# ---------------------------------------------------------------------------

def get_device_info() -> dict:
    """Return dict with keys: device (torch.device), name (str), vram_mb (int|None)."""
    if torch.cuda.is_available():
        dev = torch.device("cuda", 0)
        name = torch.cuda.get_device_name(0)
        vram_mb = torch.cuda.get_device_properties(0).total_mem // (1024 * 1024)
        return {"device": dev, "name": name, "vram_mb": vram_mb}
    return {"device": torch.device("cpu"), "name": "CPU", "vram_mb": None}


DEVICE_INFO = get_device_info()
DEVICE = DEVICE_INFO["device"]


def device_label() -> str:
    info = DEVICE_INFO
    if info["vram_mb"]:
        return f"{info['name']}  ({info['vram_mb']} MB VRAM)"
    return info["name"]


# ---------------------------------------------------------------------------
# Model wrapper
# ---------------------------------------------------------------------------

class ColorizerBackend:
    """Unified interface around whichever model we can load."""

    def __init__(self):
        self.backend = None  # "ddcolor" | "eccv16" | "siggraph17"
        self.model = None
        self._load_model()

    # -- loading ------------------------------------------------------------

    def _load_model(self):
        # Attempt 1 – DDColor
        if self._try_ddcolor():
            return
        # Attempt 2 – colorizers (Zhang ECCV16 / SIGGRAPH17)
        if self._try_colorizers():
            return
        raise RuntimeError(
            "No colorization backend available. "
            "Install 'ddcolor' or 'colorizers' (pip install colorizers)."
        )

    def _try_ddcolor(self) -> bool:
        try:
            from ddcolor import DDColor as DDColorModel  # type: ignore
            self.model = DDColorModel.from_pretrained("piddnad/ddcolor_paper_tiny")
            self.model = self.model.to(DEVICE).eval()
            self.backend = "ddcolor"
            print("[colorize_engine] Loaded DDColor (tiny) on", DEVICE)
            return True
        except Exception as exc:
            warnings.warn(f"DDColor not available ({exc}); trying fallback.")
            return False

    def _try_colorizers(self) -> bool:
        try:
            from colorizers import eccv16, siggraph17, load_img, preprocess_img, postprocess_tens  # type: ignore  # noqa: E501
            # SIGGRAPH17 generally better, but keep ECCV16 as last resort
            try:
                self.model = siggraph17(pretrained=True).to(DEVICE).eval()
                self.backend = "siggraph17"
                print("[colorize_engine] Loaded SIGGRAPH17 on", DEVICE)
            except Exception:
                self.model = eccv16(pretrained=True).to(DEVICE).eval()
                self.backend = "eccv16"
                print("[colorize_engine] Loaded ECCV16 on", DEVICE)
            return True
        except Exception as exc:
            warnings.warn(f"colorizers not available ({exc}).")
            return False

    # -- inference ----------------------------------------------------------

    @torch.no_grad()
    def colorize(self, img_bgr: np.ndarray) -> np.ndarray:
        """
        Accept a BGR uint8 image (any size).
        Return a BGR uint8 colorized image at the *original* resolution.
        """
        if self.backend == "ddcolor":
            return self._run_ddcolor(img_bgr)
        return self._run_colorizers(img_bgr)

    def _run_ddcolor(self, img_bgr: np.ndarray) -> np.ndarray:
        from ddcolor import DDColor as DDColorModel  # type: ignore
        h, w = img_bgr.shape[:2]
        # DDColor expects RGB float32 tensor [1,3,H,W] resized to 512
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        inp = cv2.resize(img_rgb, (512, 512))
        tensor = torch.from_numpy(inp).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
        out = self.model(tensor)  # [1,3,512,512] RGB float
        out = out.squeeze(0).permute(1, 2, 0).cpu().numpy()
        out = np.clip(out * 255, 0, 255).astype(np.uint8)
        out = cv2.resize(out, (w, h))
        return cv2.cvtColor(out, cv2.COLOR_RGB2BGR)

    def _run_colorizers(self, img_bgr: np.ndarray) -> np.ndarray:
        from colorizers import load_img, preprocess_img, postprocess_tens  # type: ignore
        h, w = img_bgr.shape[:2]

        # colorizers expects a PIL-style RGB or a file path; we pass numpy.
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        # Convert to the format preprocess_img expects (float, 0-1, LAB internally)
        img_l = img_rgb.astype(np.float32) / 255.0

        from skimage.color import rgb2lab, lab2rgb  # shipped with colorizers deps
        lab = rgb2lab(img_l)  # H x W x 3
        l_channel = lab[:, :, 0]

        # Resize L to 256x256 for model input
        l_resized = cv2.resize(l_channel, (256, 256))
        # Model expects [1,1,256,256] tensor
        tens_l = torch.from_numpy(l_resized).unsqueeze(0).unsqueeze(0).float().to(DEVICE)
        # Normalize L from [0,100] to [-50,50] (model convention)
        tens_l = tens_l - 50.0

        out_ab = self.model(tens_l).cpu()  # [1,2,256,256]

        # postprocess_tens handles resizing ab back and combining with L
        result = postprocess_tens(
            torch.from_numpy(l_channel).unsqueeze(0).unsqueeze(0),
            out_ab,
        )
        result = (np.clip(result, 0, 1) * 255).astype(np.uint8)
        # result is H x W x 3 RGB
        result = cv2.resize(result, (w, h))
        return cv2.cvtColor(result, cv2.COLOR_RGB2BGR)


# ---------------------------------------------------------------------------
# Convenience: single-image colorize from file path
# ---------------------------------------------------------------------------

_ENGINE: ColorizerBackend | None = None


def get_engine() -> ColorizerBackend:
    """Lazy-load singleton so Gradio import doesn't block on model download."""
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = ColorizerBackend()
    return _ENGINE


def colorize_file(path: str) -> np.ndarray:
    """Read an image file and return colorized BGR ndarray."""
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Cannot read image: {path}")
    return get_engine().colorize(img)
