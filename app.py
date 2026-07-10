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
    """Return ``(image_path, boxes_json_text)`` for the demo example, if present."""
    try:
        image_path = ASSETS / "example_0_in.jpg"
        boxes_text = (ASSETS / "example_0.json").read_text()
        if image_path.exists():
            return str(image_path), boxes_text
    except OSError as e:
        logger.warning(f"could not load example assets: {e}")
    return None, "[]"


EXAMPLE_IMAGE, EXAMPLE_JSON = _load_example()


def annotate_image(image, boxes_json, label_key, color_key, width, font_size):
    """Draw the bounding boxes described by ``boxes_json`` onto ``image``.

    Args:
        image: Input ``PIL.Image.Image`` from the UI.
        boxes_json: JSON text: a list of object dicts, each with a pascal_voc
            ``boundingBox`` ``{x0, y0, x1, y1}``.
        label_key: Object key drawn as each box's label.
        color_key: Object key used to color-group boxes.
        width: Box outline width in pixels.
        font_size: Label font size in points.

    Returns:
        The annotated ``PIL.Image.Image``.
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
        width=int(width),
        font=font,
    )


app = gr.Interface(
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
        gr.Slider(1, 10, value=3, step=1, label="Box width"),
        gr.Slider(8, 60, value=20, step=1, label="Font size"),
    ],
    outputs=gr.Image(type="pil", label="Annotated Image"),
    examples=(
        [[EXAMPLE_IMAGE, EXAMPLE_JSON, "object_id", "object_id", 3, 20]]
        if EXAMPLE_IMAGE
        else None
    ),
    title="PILBox — Bounding Box Annotator",
    description="Draw pascal_voc bounding boxes on an image using numpy + Pillow.",
    api_name="annotate",
)

if __name__ == "__main__":
    app.launch(
        mcp_server=True, app_kwargs={"docs_url": "/docs"}  # FastAPI Swagger API Docs
    )
