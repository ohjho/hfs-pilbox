"""Gradio app: annotate an image with pascal_voc bounding boxes via pilbox.

Web-UI counterpart of ``annotate_cli.py`` — paste a list of objects (each with a
``boundingBox`` dict ``{x0, y0, x1, y1}``) and get the annotated image back.
"""

import json
from pathlib import Path

import gradio as gr
from loguru import logger

import pilbox

ASSETS = Path(__file__).parent / "assets"


def _load_example():
    """Return ``(image_path, boxes_json_text)`` for the demo example.

    The example assets (``example_0_in.jpg`` + ``example_0.json``) live in
    ``assets/`` but are git-ignored and not deployed to the Space, so both are
    optional. When either is missing the app runs without a preloaded example.

    Returns:
        ``(image_path, boxes_json_text)`` when both assets are present, else
        ``(None, "[]")``.
    """
    image_path = ASSETS / "example_0_in.jpg"
    boxes_path = ASSETS / "example_0.json"
    if not (image_path.exists() and boxes_path.exists()):
        logger.info("demo example assets not found in {}; running without one", ASSETS)
        return None, "[]"
    try:
        return str(image_path), boxes_path.read_text()
    except OSError as e:
        logger.warning(f"could not read example assets: {e}")
        return None, "[]"


EXAMPLE_IMAGE, EXAMPLE_JSON = _load_example()


def annotate_image(
    image, boxes_json, label_key, color_key, mask_key, mask_alpha, width, font_size
):
    """Draw pascal_voc bounding boxes (and optional segmentation masks) onto an image and return the annotated image.

    boxes_json is a JSON list of object dicts. Each object holds its box under a "boundingBox"
    key as {"x0", "y0", "x1", "y1"}: (x0, y0) is the top-left corner and (x1, y1) is the
    bottom-right corner, in absolute pixels of the input image (the "pascal_voc" format,
    documented at
    https://albumentations.ai/docs/3-basic-usage/bounding-boxes-augmentations/#bounding-box-formats ).
    NOT normalized to 0-1 and NOT [x, y, width, height]. Each object may also carry the label_key
    and color_key fields, plus a mask_key field holding a base64-encoded PNG mask. When an object
    has a mask, it is drawn as a translucent colored overlay beneath the box, using the SAME color
    as that object's box (both derived from color_key). The output is the input image with every
    mask and box drawn on it.

    Args:
        image: The RGB image to annotate.
        boxes_json: JSON text — a list of object dicts, each with a "boundingBox" {x0, y0, x1, y1} and optional label/color/mask fields.
        label_key: Name of the object field whose value is drawn as each box's text label.
        color_key: Name of the object field used to color-group boxes and masks (each distinct value gets its own color).
        mask_key: Name of the object field holding a base64-encoded PNG mask; leave empty to disable mask drawing.
        mask_alpha: Mask overlay opacity from 0.0 (invisible) to 1.0 (solid color).
        width: Box outline width in pixels.
        font_size: Label font size in points.
    """
    if image is None:
        raise gr.Error("Please provide an input image.")
    try:
        objects = json.loads(boxes_json)
    except json.JSONDecodeError as e:
        raise gr.Error(f"Invalid JSON: {e}")

    font = pilbox.load_pil_font(size=int(font_size))
    return pilbox.annotate(
        image,
        objects,
        label_key=label_key,
        color_key=color_key,
        mask_key=mask_key,
        mask_alpha=float(mask_alpha),
        width=int(width),
        font=font,
    )


def crop_image(image, x0, y0, x1, y1):
    """Crop an image to the pascal_voc box and return only that region as a new image.

    The crop box is given as absolute pixel coordinates in the "pascal_voc" format: (x0, y0) is
    the top-left corner and (x1, y1) is the bottom-right corner, measured in pixels of the input
    image (documented at
    https://albumentations.ai/docs/3-basic-usage/bounding-boxes-augmentations/#bounding-box-formats ).
    The box must be non-empty (x1 > x0 and y1 > y0) and lie fully within the image; otherwise an
    error is returned. NOT normalized to 0-1 and NOT [x, y, width, height]. The output is the
    cropped RGB image of size (x1 - x0) by (y1 - y0).

    Args:
        image: The RGB image to crop.
        x0: Left edge of the crop box, in absolute pixels from the left.
        y0: Top edge of the crop box, in absolute pixels from the top.
        x1: Right edge of the crop box, in absolute pixels from the left; must be greater than x0.
        y1: Bottom edge of the crop box, in absolute pixels from the top; must be greater than y0.
    """
    if image is None:
        raise gr.Error("Please provide an input image.")
    try:
        return pilbox.crop(image, x0, y0, x1, y1)
    except ValueError as e:
        raise gr.Error(str(e))


annotate_interface = gr.Interface(
    fn=annotate_image,
    inputs=[
        gr.Image(type="pil", label="Input Image"),
        gr.Code(
            language="json",
            label="Bounding Boxes (pascal_voc JSON)",
            value=EXAMPLE_JSON,
        ),
        gr.Textbox(value="object_id", label="Label key"),
        gr.Textbox(value="object_id", label="Color key"),
        gr.Textbox(value="b64_mask", label="Mask key"),
        gr.Slider(0, 1, value=0.5, step=0.05, label="Mask opacity"),
        gr.Slider(1, 10, value=3, step=1, label="Box width"),
        gr.Slider(8, 60, value=20, step=1, label="Font size"),
    ],
    outputs=gr.Image(type="pil", label="Annotated Image"),
    examples=(
        [[EXAMPLE_IMAGE, EXAMPLE_JSON, "object_id", "object_id", "b64_mask", 0.5, 3, 20]]
        if EXAMPLE_IMAGE
        else None
    ),
    title="PILBox — Bounding Box Annotator",
    description="Draw pascal_voc bounding boxes on an image using numpy + Pillow.",
    api_name="annotate",
)

crop_interface = gr.Interface(
    fn=crop_image,
    inputs=[
        gr.Image(type="pil", label="Input Image"),
        gr.Number(value=0, precision=0, label="x0 (left)"),
        gr.Number(value=0, precision=0, label="y0 (top)"),
        gr.Number(value=100, precision=0, label="x1 (right)"),
        gr.Number(value=100, precision=0, label="y1 (bottom)"),
    ],
    outputs=gr.Image(type="pil", label="Cropped Image"),
    examples=(
        [[EXAMPLE_IMAGE, 0, 0, 100, 100]] if EXAMPLE_IMAGE else None
    ),
    title="PILBox — Image Cropper",
    description="Crop an image to a pascal_voc box (x0, y0, x1, y1) using numpy + Pillow.",
    api_name="crop",
)

app = gr.TabbedInterface(
    [annotate_interface, crop_interface],
    ["Annotate", "Crop"],
    title="PILBox",
)

if __name__ == "__main__":
    app.launch(
        mcp_server=True, app_kwargs={"docs_url": "/docs"}  # FastAPI Swagger API Docs
    )
