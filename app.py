"""Gradio app: annotate an image with pascal_voc bounding boxes via pilbox.

Web-UI counterpart of ``annotate_cli.py`` — paste a list of objects (each with a
``boundingBox`` dict ``{x0, y0, x1, y1}``) and get the annotated image back.
"""

import json
import tempfile
from pathlib import Path

import gradio as gr
from loguru import logger
from PIL import ImageColor

import boxer
import pilbox
import vidbox

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


def _example_mask():
    """Return the first object's base64 PNG mask from the demo JSON, if any.

    Used to preload the Mask tab's example. Returns an empty string when the
    example assets are absent (e.g. on the Space) or carry no mask.
    """
    try:
        objects = json.loads(EXAMPLE_JSON)
        for obj in objects:
            if obj.get("b64_mask"):
                return obj["b64_mask"]
    except (json.JSONDecodeError, AttributeError, TypeError):
        pass
    return ""


EXAMPLE_MASK = _example_mask()


def _load_video_example():
    """Return ``(video_path, detections_json_path)`` for the video demo example.

    Both assets live in ``assets/`` but are git-ignored and not deployed to the
    Space, so both are optional; returns ``(None, None)`` when either is absent.
    """
    video_path = ASSETS / "17078229_3222904.mp4"
    json_path = ASSETS / "17078229_3222904-SAM2_tiny_ZeroGPU-with_mask.json"
    if not (video_path.exists() and json_path.exists()):
        logger.info("video example assets not found in {}; running without one", ASSETS)
        return None, None
    return str(video_path), str(json_path)


EXAMPLE_VIDEO, EXAMPLE_VIDEO_JSON = _load_video_example()


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


def _rgb_from_css(color: str):
    """Convert a CSS color string ("#rrggbb" or "rgba(r,g,b,a)") to an (r, g, b) tuple."""
    color = (color or "#000000").strip()
    if color.startswith("rgba") or color.startswith("rgb"):
        nums = color[color.index("(") + 1 : color.index(")")].split(",")
        return tuple(int(float(n)) for n in nums[:3])
    return ImageColor.getrgb(color)


def mask_image(image, b64_mask, bg_color):
    """Cut out an image's foreground using a base64-encoded PNG mask and return it on a solid background.

    The mask is a base64-encoded PNG string the SAME pixel size as the input image: pixels that are
    non-zero (white) mark the foreground to keep, and zero (black) pixels are the background. The
    output keeps the foreground pixels unchanged and replaces every background pixel with bg_color,
    so the subject is "masked out" of its scene onto a flat backdrop. bg_color is a CSS hex color
    string like "#000000" (the default, black); "#ff0000" would put the foreground on red.

    Args:
        image: The RGB image to mask.
        b64_mask: Base64-encoded PNG mask, same width and height as image; non-zero pixels are the foreground to keep.
        bg_color: Background fill as a CSS hex color string like "#rrggbb"; defaults to black "#000000".
    """
    if image is None:
        raise gr.Error("Please provide an input image.")
    try:
        rgb = _rgb_from_css(bg_color)
        return pilbox.apply_mask(image, b64_mask, bg_rgb_tup=rgb)
    except ValueError as e:
        raise gr.Error(str(e))


def annotate_video(
    video_path,
    boxes_json_file,
    bbox_format,
    coord_keys,
    label_key,
    color_key,
    mask_key,
    mask_alpha,
    width,
    font_size,
) -> str:
    """Draw per-frame bounding boxes (and optional masks) onto every frame of a video and return the annotated video.

    Takes a video plus a detections JSON file — a flat JSON list of per-frame detection objects. Each
    object holds a 0-based frame index under the "frame" key, the box coordinates under the four keys
    named by coord_keys (default "x,y,w,h", read in that order), and optionally a label value, a color
    value, and a base64-encoded PNG mask the same pixel size as the video frame. Every detection is
    drawn on its frame; each distinct color_key value keeps ONE stable color across the whole video (so
    a track id is one consistent color), and any mask is drawn as a translucent overlay beneath the box
    in that same color. The output is a new silent video at the source resolution and frame rate with
    all boxes, labels, and masks burned in. The box coordinates are interpreted per bbox_format, one of:
    "pascal_voc" = [x0, y0, x1, y1] absolute pixels; "albumentations" = [x0, y0, x1, y1] normalized 0-1;
    "coco" = [x0, y0, width, height] absolute pixels; "coco_normalized" = [x0, y0, width, height]
    normalized 0-1 — as documented at
    https://albumentations.ai/docs/3-basic-usage/bounding-boxes-augmentations/#bounding-box-formats .

    Args:
        video_path: The input video file to annotate.
        boxes_json_file: A .json file holding a flat list of per-frame detection objects (see above).
        bbox_format: Box coordinate convention: one of "pascal_voc", "albumentations", "coco", or "coco_normalized".
        coord_keys: Comma-separated names of the four object keys holding the box values, in order (default "x,y,w,h").
        label_key: Name of the object field whose value is drawn as each box's text label.
        color_key: Name of the object field used to color-group boxes and masks; each distinct value gets one stable color across all frames.
        mask_key: Name of the object field holding a base64-encoded PNG mask; leave empty to disable mask drawing.
        mask_alpha: Mask overlay opacity from 0.0 (invisible) to 1.0 (solid color).
        width: Box outline width in pixels.
        font_size: Label font size in points.
    """
    if video_path is None:
        raise gr.Error("Please provide an input video.")
    if not boxes_json_file:
        raise gr.Error("Please provide a detections JSON file.")
    try:
        with open(boxes_json_file) as f:
            detections = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        raise gr.Error(f"Could not read detections JSON: {e}")

    out_path = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
    try:
        return vidbox.annotate_video(
            video_path,
            detections,
            out_path,
            bbox_format=bbox_format,
            coord_keys=tuple(k.strip() for k in coord_keys.split(",")),
            label_key=label_key,
            color_key=color_key,
            mask_key=mask_key,
            mask_alpha=float(mask_alpha),
            width=int(width),
            font_size=int(font_size),
        )
    except (ValueError, KeyError) as e:
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

annotate_video_interface = gr.Interface(
    fn=annotate_video,
    inputs=[
        gr.Video(label="Input Video"),
        gr.File(label="Detections JSON", file_types=[".json"], type="filepath"),
        gr.Dropdown(
            choices=list(boxer.BBOX_FORMATS),
            value="coco_normalized",
            label="Box format",
        ),
        gr.Textbox(value="x,y,w,h", label="Coordinate keys (comma-separated)"),
        gr.Textbox(value="track_id", label="Label key"),
        gr.Textbox(value="track_id", label="Color key"),
        gr.Textbox(value="mask_b64", label="Mask key"),
        gr.Slider(0, 1, value=0.5, step=0.05, label="Mask opacity"),
        gr.Slider(1, 10, value=3, step=1, label="Box width"),
        gr.Slider(8, 60, value=20, step=1, label="Font size"),
    ],
    outputs=gr.Video(label="Annotated Video"),
    examples=(
        [
            [
                EXAMPLE_VIDEO,
                EXAMPLE_VIDEO_JSON,
                "coco_normalized",
                "x,y,w,h",
                "track_id",
                "track_id",
                "mask_b64",
                0.5,
                3,
                20,
            ]
        ]
        if EXAMPLE_VIDEO and EXAMPLE_VIDEO_JSON
        else None
    ),
    title="PILBox — Video Annotator",
    description="Draw per-frame bounding boxes and masks over a video (pascal_voc / albumentations / coco / coco_normalized).",
    api_name="annotate_video",
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

mask_interface = gr.Interface(
    fn=mask_image,
    inputs=[
        gr.Image(type="pil", label="Input Image"),
        gr.Textbox(lines=4, label="Mask (base64-encoded PNG)", value=EXAMPLE_MASK),
        gr.ColorPicker(value="#000000", label="Background color"),
    ],
    outputs=gr.Image(type="pil", label="Masked Image"),
    examples=(
        [[EXAMPLE_IMAGE, EXAMPLE_MASK, "#000000"]]
        if EXAMPLE_IMAGE and EXAMPLE_MASK
        else None
    ),
    title="PILBox — Background Masker",
    description="Cut out an image's foreground with a base64 PNG mask, over a solid background color.",
    api_name="mask",
)

app = gr.TabbedInterface(
    [annotate_interface, annotate_video_interface, crop_interface, mask_interface],
    ["Annotate", "Annotate Video", "Crop", "Mask"],
    title="PILBox",
)

if __name__ == "__main__":
    app.launch(
        mcp_server=True, app_kwargs={"docs_url": "/docs"}  # FastAPI Swagger API Docs
    )
