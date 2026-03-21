"""
Photo Colorizer — Gradio web application.

Upload one or more B&W photos and an optional reference image, then colorize
with neural-network models enhanced by reference-palette colour transfer.
"""

import os
import time
import uuid
import numpy as np
import cv2
import gradio as gr
from PIL import Image

from colorize_engine import get_engine, device_label, DEVICE_INFO
from palette import (
    apply_reference_colors,
    palette_preview,
    REGION_PRESETS,
)

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def pil_to_bgr(pil_img: Image.Image) -> np.ndarray:
    arr = np.array(pil_img.convert("RGB"))
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def bgr_to_pil(bgr: np.ndarray) -> Image.Image:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def save_output(pil_img: Image.Image, src_name: str) -> str:
    base, ext = os.path.splitext(os.path.basename(src_name))
    if ext.lower() not in (".jpg", ".jpeg", ".png"):
        ext = ".png"
    fname = f"{base}_colorized_{uuid.uuid4().hex[:6]}{ext}"
    path = os.path.join(OUTPUT_DIR, fname)
    pil_img.save(path)
    return path


def make_side_by_side(original_pil: Image.Image, colorized_pil: Image.Image) -> Image.Image:
    """Create a side-by-side comparison image."""
    w1, h1 = original_pil.size
    w2, h2 = colorized_pil.size
    h = max(h1, h2)
    # Scale both to the same height
    if h1 != h:
        w1 = int(w1 * h / h1)
        original_pil = original_pil.resize((w1, h), Image.LANCZOS)
    if h2 != h:
        w2 = int(w2 * h / h2)
        colorized_pil = colorized_pil.resize((w2, h), Image.LANCZOS)
    canvas = Image.new("RGB", (w1 + w2 + 10, h), (40, 40, 40))
    canvas.paste(original_pil.convert("RGB"), (0, 0))
    canvas.paste(colorized_pil.convert("RGB"), (w1 + 10, 0))
    return canvas


# ---------------------------------------------------------------------------
# Core processing function
# ---------------------------------------------------------------------------

def process(
    bw_images: list,
    reference_image,
    strength: float,
    region: str,
    n_colors: int,
):
    """
    Main Gradio callback.

    Parameters
    ----------
    bw_images : list of uploaded file paths (Gradio file component)
    reference_image : single uploaded PIL image or None
    strength : 0-100 slider value
    region : region preset name
    n_colors : number of palette colours
    """
    if not bw_images:
        raise gr.Error("Please upload at least one B&W photo.")

    engine = get_engine()
    strength_frac = strength / 100.0

    ref_bgr = None
    if reference_image is not None:
        ref_bgr = pil_to_bgr(reference_image)

    comparisons = []
    colorized_outputs = []
    saved_paths = []

    for file_obj in bw_images:
        # Gradio gives file paths for the File component
        file_path = file_obj if isinstance(file_obj, str) else file_obj.name if hasattr(file_obj, "name") else str(file_obj)
        original_bgr = cv2.imread(file_path, cv2.IMREAD_COLOR)
        if original_bgr is None:
            continue

        # Colorize
        colorized_bgr = engine.colorize(original_bgr)

        # Apply reference palette if available
        if ref_bgr is not None and strength_frac > 0:
            colorized_bgr = apply_reference_colors(
                colorized_bgr,
                ref_bgr,
                strength=strength_frac,
                n_colors=int(n_colors),
                region=region,
            )

        original_pil = bgr_to_pil(original_bgr)
        colorized_pil = bgr_to_pil(colorized_bgr)

        # Save
        path = save_output(colorized_pil, file_path)
        saved_paths.append(path)

        # Build comparison
        comparison = make_side_by_side(original_pil, colorized_pil)
        comparisons.append(comparison)
        colorized_outputs.append(colorized_pil)

    if not comparisons:
        raise gr.Error("Could not read any of the uploaded images.")

    palette_img = None
    if ref_bgr is not None:
        swatch_bgr = palette_preview(ref_bgr, n_colors=int(n_colors), region=region)
        palette_img = bgr_to_pil(swatch_bgr)

    status = f"Colorized {len(saved_paths)} image(s). Saved to /output folder."

    # Return first comparison as the preview, the gallery, palette, and status
    return (
        comparisons[0] if comparisons else None,
        colorized_outputs,
        palette_img,
        status,
    )


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

def build_ui() -> gr.Blocks:
    with gr.Blocks(
        title="Photo Colorizer",
        theme=gr.themes.Soft(),
    ) as demo:
        gr.Markdown("# Photo Colorizer")
        gr.Markdown(
            "Upload black & white photos and an optional colour reference image. "
            "The app colorizes using a neural network and blends toward the reference palette."
        )

        # Device info banner
        dev = DEVICE_INFO
        if dev["vram_mb"]:
            device_text = f"**Device:** {dev['name']}  |  **VRAM:** {dev['vram_mb']} MB"
        else:
            device_text = "**Device:** CPU  (install CUDA-enabled PyTorch for GPU acceleration)"
        gr.Markdown(device_text)

        with gr.Row():
            with gr.Column(scale=1):
                bw_input = gr.File(
                    label="B&W Photo(s)",
                    file_count="multiple",
                    file_types=["image"],
                    type="filepath",
                )
                ref_input = gr.Image(
                    label="Reference colour image (optional)",
                    type="pil",
                )
                strength_slider = gr.Slider(
                    minimum=0,
                    maximum=100,
                    value=50,
                    step=1,
                    label="Reference colour strength (0 = pure model, 100 = full reference)",
                )
                region_dropdown = gr.Dropdown(
                    choices=list(REGION_PRESETS.keys()),
                    value="Full image",
                    label="Region sampler — area of reference image to pull colours from",
                )
                n_colors_slider = gr.Slider(
                    minimum=2,
                    maximum=16,
                    value=6,
                    step=1,
                    label="Number of palette colours (k-means clusters)",
                )
                run_btn = gr.Button("Colorize", variant="primary")

            with gr.Column(scale=2):
                comparison_output = gr.Image(label="Before / After (side by side)")
                gallery_output = gr.Gallery(label="Colorized results", columns=2)
                palette_output = gr.Image(label="Extracted palette")
                status_output = gr.Textbox(label="Status", interactive=False)

        run_btn.click(
            fn=process,
            inputs=[bw_input, ref_input, strength_slider, region_dropdown, n_colors_slider],
            outputs=[comparison_output, gallery_output, palette_output, status_output],
        )

    return demo


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    demo = build_ui()
    demo.launch(server_name="0.0.0.0", server_port=7860)
