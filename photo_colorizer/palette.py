"""
Reference-image colour palette extraction and LAB-space blending.

Pipeline:
1. Crop a region from the reference image (full, top-half, bottom-half, or
   user-defined bounding box).
2. Run k-means on the cropped pixels in LAB space → dominant palette.
3. For each pixel of the model-colorized image, find the nearest palette
   colour (in LAB) and blend toward it according to `strength` (0-1).
"""

import numpy as np
import cv2
from sklearn.cluster import MiniBatchKMeans


# ---------------------------------------------------------------------------
# Region cropping helpers
# ---------------------------------------------------------------------------

REGION_PRESETS = {
    "Full image": None,
    "Top half": (0.0, 0.0, 1.0, 0.5),
    "Bottom half": (0.0, 0.5, 1.0, 1.0),
    "Left half": (0.0, 0.0, 0.5, 1.0),
    "Right half": (0.5, 0.0, 1.0, 1.0),
    "Center crop": (0.25, 0.25, 0.75, 0.75),
}


def crop_region(img_bgr: np.ndarray, region: str | tuple | None) -> np.ndarray:
    """
    Crop *img_bgr* according to *region*.
    - None / "Full image" → full image
    - str key in REGION_PRESETS → use fractional box
    - (x1_frac, y1_frac, x2_frac, y2_frac) tuple → custom crop
    """
    if region is None or region == "Full image":
        return img_bgr

    if isinstance(region, str):
        box = REGION_PRESETS.get(region)
        if box is None:
            return img_bgr
    else:
        box = region

    h, w = img_bgr.shape[:2]
    x1 = int(box[0] * w)
    y1 = int(box[1] * h)
    x2 = int(box[2] * w)
    y2 = int(box[3] * h)
    return img_bgr[y1:y2, x1:x2]


# ---------------------------------------------------------------------------
# Palette extraction (k-means in LAB space)
# ---------------------------------------------------------------------------

def extract_palette(
    img_bgr: np.ndarray,
    n_colors: int = 6,
    region: str | tuple | None = None,
    max_pixels: int = 50_000,
) -> np.ndarray:
    """
    Return an (n_colors, 3) float32 array of dominant LAB colours
    extracted from the given region of *img_bgr*.
    """
    crop = crop_region(img_bgr, region)
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB).astype(np.float32)
    pixels = lab.reshape(-1, 3)

    # Subsample for speed
    if len(pixels) > max_pixels:
        idx = np.random.default_rng(42).choice(len(pixels), max_pixels, replace=False)
        pixels = pixels[idx]

    km = MiniBatchKMeans(n_clusters=n_colors, random_state=42, batch_size=1024)
    km.fit(pixels)
    return km.cluster_centers_.astype(np.float32)  # shape (n_colors, 3) in LAB


# ---------------------------------------------------------------------------
# LAB blending: push colorized image toward the reference palette
# ---------------------------------------------------------------------------

def blend_with_palette(
    colorized_bgr: np.ndarray,
    palette_lab: np.ndarray,
    strength: float = 0.5,
) -> np.ndarray:
    """
    For each pixel in *colorized_bgr*, find its nearest palette colour in LAB
    and blend the a,b channels toward that colour.

    Parameters
    ----------
    strength : float  0 → pure model output, 1 → full palette replacement of chrominance.
    """
    if strength <= 0.0 or palette_lab is None or len(palette_lab) == 0:
        return colorized_bgr

    strength = min(strength, 1.0)

    lab = cv2.cvtColor(colorized_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    h, w = lab.shape[:2]
    pixels = lab.reshape(-1, 3)

    # Compute distances in LAB (all 3 channels) to each palette centre
    # pixels: (N,3), palette_lab: (K,3) → dists: (N,K)
    # Use chunked computation to keep memory reasonable for large images
    chunk = 200_000
    nearest_ab = np.empty((len(pixels), 2), dtype=np.float32)

    for start in range(0, len(pixels), chunk):
        end = min(start + chunk, len(pixels))
        p = pixels[start:end]  # (C, 3)
        dists = np.linalg.norm(
            p[:, np.newaxis, :] - palette_lab[np.newaxis, :, :], axis=2
        )  # (C, K)
        idx = dists.argmin(axis=1)  # (C,)
        nearest_ab[start:end] = palette_lab[idx, 1:3]  # a,b of nearest colour

    # Blend a,b channels
    pixels[:, 1] = pixels[:, 1] * (1 - strength) + nearest_ab[:, 0] * strength
    pixels[:, 2] = pixels[:, 2] * (1 - strength) + nearest_ab[:, 1] * strength

    lab = pixels.reshape(h, w, 3)
    lab = np.clip(lab, 0, 255).astype(np.uint8)
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


# ---------------------------------------------------------------------------
# High-level helper used by the Gradio app
# ---------------------------------------------------------------------------

def apply_reference_colors(
    colorized_bgr: np.ndarray,
    reference_bgr: np.ndarray,
    strength: float = 0.5,
    n_colors: int = 6,
    region: str | tuple | None = None,
) -> np.ndarray:
    """Extract palette from the reference and blend it into the colorized image."""
    palette = extract_palette(reference_bgr, n_colors=n_colors, region=region)
    return blend_with_palette(colorized_bgr, palette, strength=strength)


def palette_preview(
    reference_bgr: np.ndarray,
    n_colors: int = 6,
    region: str | tuple | None = None,
    swatch_size: int = 64,
) -> np.ndarray:
    """Return a small BGR image showing the extracted palette swatches."""
    palette_lab = extract_palette(reference_bgr, n_colors=n_colors, region=region)
    # Convert each LAB centre to BGR for display
    swatches = []
    for lab_val in palette_lab:
        patch = np.full((swatch_size, swatch_size, 3), lab_val, dtype=np.float32)
        patch = np.clip(patch, 0, 255).astype(np.uint8)
        bgr_patch = cv2.cvtColor(patch, cv2.COLOR_LAB2BGR)
        swatches.append(bgr_patch)
    return np.hstack(swatches)
