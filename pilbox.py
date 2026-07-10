"""PILBox: ultra-lite bounding-box annotation using only numpy + Pillow.

Bounding boxes use the **pascal_voc** convention: absolute pixel coordinates
``(x0, y0, x1, y1)`` for the top-left and bottom-right corners, matching the
``boundingBox`` dict found in ``assets/example_0.json``.

The only third-party dependencies are ``numpy``, ``Pillow`` and ``loguru``.
"""

import os
import json
import colorsys

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from loguru import logger


# --------------------------------------------------------------------------- #
# Image helpers (kept for future features)
# --------------------------------------------------------------------------- #
def im_crop(im_rgb_array, x0, y0, x1, y1):
    """Crop an RGB numpy image to the pascal_voc box ``(x0, y0, x1, y1)``."""
    return im_rgb_array[y0:y1, x0:x1, :]


def im_center_crop(im_rgb_array, w, h):
    """Center-crop an RGB numpy image to width ``w`` and height ``h``."""
    h_, w_, _ = im_rgb_array.shape
    assert (
        w_ >= w and h_ >= h
    ), f"target width ({w}) and height ({h}) must be less then input image width ({w_}), height ({h_})"
    x0 = int((w_ - w) / 2)
    x1 = x0 + w
    y0 = int((h_ - h) / 2)
    y1 = y0 + h
    return im_crop(im_rgb_array, x0=x0, y0=y0, x1=x1, y1=y1)


def load_pil_font(
    preferred_font_name: str = "arial.ttf", size: int = 12, fallback_to_any: bool = True
):
    """Attempt to load a TrueType font, with fallbacks.

    Args:
        preferred_font_name: The name or relative path of the font file
            (e.g. ``'arial.ttf'``).
        size: The font size in points.
        fallback_to_any: If True, tries a list of common fonts before falling
            back to Pillow's default font.

    Returns:
        PIL.ImageFont.FreeTypeFont | PIL.ImageFont.ImageFont: The loaded font.
    """
    any_fonts = ["DejaVuSans.ttf", "LiberationSans-Regular.ttf", "Arial Unicode.ttf"]
    # 1. Try absolute path relative to script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    possible_paths = [
        script_dir,
        os.path.join(
            script_dir,
            "fonts",
        ),  # Common practice to put fonts in a 'fonts' folder
    ]

    # 2. Try known system paths (less portable, but useful if system fonts are preferred)
    if os.name == "nt":  # Windows
        possible_paths.append(
            os.path.join(os.environ.get("WINDIR", "C:\\Windows"), "Fonts")
        )
    elif os.name == "posix":  # Linux/macOS
        possible_paths.extend(
            [
                "/usr/share/fonts/truetype",
                "/usr/local/share/fonts/truetype",
                os.path.join(
                    os.path.expanduser("~"),
                    ".local/share/fonts",
                ),  # User-specific fonts
                "/Library/Fonts",  # macOS
                "/System/Library/Fonts",  # macOS
            ]
        )

    fonts = [preferred_font_name]
    fonts += any_fonts if fallback_to_any else []
    for font in fonts:
        for d in possible_paths:
            if os.path.exists(os.path.join(d, font)):
                try:
                    if font in any_fonts:
                        logger.warning(f"using fallback font {font}")
                    return ImageFont.truetype(os.path.join(d, font), size=size)
                except OSError as e:
                    logger.warning(
                        f"load_pil_font: Failed to load {font} from {d}: {e}"
                    )
                    continue  # Try next path

    logger.warning("load_pil_font: Falling back to Pillow's default font.")
    return ImageFont.load_default(size=size)


# --------------------------------------------------------------------------- #
# Color palette (unbounded, generated — no matplotlib)
# --------------------------------------------------------------------------- #
# Golden-angle in the unit hue circle; stepping by it spreads hues as far apart
# as possible for any count, so consecutive indices never collide visually.
_GOLDEN_RATIO_CONJUGATE = 0.6180339887498949


def palette_color(index: int, saturation: float = 0.65, value: float = 0.95) -> str:
    """Return the ``index``-th visually distinct color as a ``#rrggbb`` string.

    Hues are spaced by the golden angle so any number of colors stay far apart,
    meaning the palette never runs out (no fixed list, no matplotlib).

    Note:
        If you prefer fixed *named* colors instead, ``PIL.ImageColor.colormap``
        exposes ~148 built-in CSS/X11 color names (e.g. ``"lime"``, ``"cyan"``).

    Args:
        index: Zero-based color index.
        saturation: HSV saturation in ``[0, 1]``.
        value: HSV value/brightness in ``[0, 1]``.

    Returns:
        Hex color string usable directly by Pillow, e.g. ``"#f23a1b"``.

    >>> palette_color(0)
    '#f25454'
    >>> palette_color(0) != palette_color(1)
    True
    """
    hue = (index * _GOLDEN_RATIO_CONJUGATE) % 1.0
    r, g, b = colorsys.hsv_to_rgb(hue, saturation, value)
    return "#{:02x}{:02x}{:02x}".format(int(r * 255), int(g * 255), int(b * 255))


def color_for(key, mapping: dict) -> str:
    """Return a stable palette color for ``key``, assigning a new one if unseen.

    ``mapping`` is mutated in place to remember which palette index each distinct
    key was given (first-seen order), so repeated calls are deterministic.

    Args:
        key: Any hashable value to color by (e.g. an object id or class name).
        mapping: Dict tracking ``key -> palette index`` across calls.

    Returns:
        A ``#rrggbb`` color string.

    >>> m = {}
    >>> color_for("a", m) == color_for("a", m)  # stable
    True
    >>> color_for("a", m) != color_for("b", m)  # distinct
    True
    """
    if key not in mapping:
        mapping[key] = len(mapping)
    return palette_color(mapping[key])


# --------------------------------------------------------------------------- #
# Drawing
# --------------------------------------------------------------------------- #
def im_draw_bbox(
    pil_im,
    x0,
    y0,
    x1,
    y1,
    color="black",
    width=3,
    caption=None,
    caption_font=None,
    text_color="black",
):
    """Draw a bounding box (and optional label tab) on ``pil_im`` in place.

    The caption is rendered as a filled tab in ``color`` sitting just above the
    box's top-left corner, with the text drawn in ``text_color`` — matching the
    look of ``assets/example_0_out.jpg``.

    Args:
        pil_im: Target ``PIL.Image.Image`` (drawn on in place).
        x0, y0, x1, y1: pascal_voc box coordinates (floats are coerced to int).
        color: Box outline / label-tab color, as read by ``PIL.ImageColor``.
        width: Outline width in pixels.
        caption: Optional label text. If falsy, no label is drawn.
        caption_font: Optional ``PIL.ImageFont``; defaults to Pillow's default.
        text_color: Color of the caption text drawn on the tab.
    """
    if any(isinstance(i, float) for i in [x0, y0, x1, y1]):
        x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)

    if caption_font is None:
        caption_font = ImageFont.load_default()

    draw = ImageDraw.Draw(pil_im)
    draw.rectangle([(x0, y0), (x1, y1)], outline=color, width=width)

    if caption:
        caption = str(caption)
        # Measure the text so the tab hugs it.
        left, top, right, bottom = draw.textbbox((0, 0), caption, font=caption_font)
        tw, th = right - left, bottom - top
        pad = 2
        tab_h = th + 2 * pad
        # Sit the tab above the box; if there's no room, place it inside.
        tab_top = y0 - tab_h
        if tab_top < 0:
            tab_top = y0
        draw.rectangle(
            [(x0, tab_top), (x0 + tw + 2 * pad, tab_top + tab_h)],
            fill=color,
        )
        draw.text(
            (x0 + pad - left, tab_top + pad - top),
            text=caption,
            fill=text_color,
            font=caption_font,
        )


def im_draw_point(
    pil_im: Image.Image,
    x: int,
    y: int,
    caption: str = None,
    size: int = 10,
    width: int = 2,
    color: str = "red",
) -> Image.Image:
    """Return a copy of ``pil_im`` with a cross marker drawn at ``(x, y)``."""
    # Initialize variable for drawing image
    im_draw = pil_im.copy()
    draw = ImageDraw.Draw(im_draw)

    # Get params for drawing cross
    im_w, im_h = im_draw.size

    # Ensure the cross stays within image bounds
    draw_x = max(size, min(x, im_w - size))
    draw_y = max(size, min(y, im_h - size))

    # Draw cross
    draw.line((draw_x - size, draw_y, draw_x + size, draw_y), fill=color, width=width)
    draw.line((draw_x, draw_y - size, draw_x, draw_y + size), fill=color, width=width)
    if caption:
        draw.text(
            xy=(draw_x + size + size / 2, draw_y - size / 2),
            text=caption,
            fill=color,
            font_size=width * 5,
        )

    return im_draw


def annotate(
    image: Image.Image,
    objects,
    *,
    label_key: str = "object_id",
    color_key: str = "object_id",
    bbox_key: str = "boundingBox",
    width: int = 3,
    font=None,
) -> Image.Image:
    """Draw every object's pascal_voc bounding box on a copy of ``image``.

    Args:
        image: Source ``PIL.Image.Image`` (never mutated; a copy is returned).
        objects: Iterable of dicts. Each must hold a pascal_voc box under
            ``bbox_key`` as ``{"x0", "y0", "x1", "y1"}`` and, optionally, values
            under ``label_key`` and ``color_key``.
        label_key: Object key whose value is drawn as the box label.
        color_key: Object key whose value groups boxes into colors (each distinct
            value gets its own palette color).
        bbox_key: Object key holding the pascal_voc box dict.
        width: Box outline width in pixels.
        font: Optional ``PIL.ImageFont`` for labels; defaults to Pillow's default.

    Returns:
        A new annotated ``PIL.Image.Image``.

    >>> im = Image.new("RGB", (100, 100), "white")
    >>> objs = [{"object_id": 0, "boundingBox": {"x0": 10, "y0": 10, "x1": 40, "y1": 60}}]
    >>> out = annotate(im, objs)
    >>> out.size
    (100, 100)
    >>> out is im
    False
    """
    out = image.copy()
    color_map: dict = {}
    for obj in objects:
        box = obj[bbox_key]
        color = color_for(obj.get(color_key), color_map)
        caption = obj.get(label_key)
        im_draw_bbox(
            out,
            x0=box["x0"],
            y0=box["y0"],
            x1=box["x1"],
            y1=box["y1"],
            color=color,
            width=width,
            caption=None if caption is None else str(caption),
            caption_font=font,
        )
    return out


def annotate_file(
    image_path: str,
    annotations_path: str,
    output_path: str,
    *,
    label_key: str = "object_id",
    color_key: str = "object_id",
    width: int = 3,
    font_size: int = 20,
) -> str:
    """Annotate ``image_path`` using boxes from a JSON file and save the result.

    Args:
        image_path: Path to the source image.
        annotations_path: Path to a JSON file holding a list of object dicts
            (see :func:`annotate`).
        output_path: Where to write the annotated image.
        label_key: Object key drawn as each box's label.
        color_key: Object key used to group boxes into colors.
        width: Box outline width in pixels.
        font_size: Label font size in points.

    Returns:
        ``output_path``.
    """
    with open(annotations_path) as f:
        objects = json.load(f)
    image = Image.open(image_path).convert("RGB")
    font = load_pil_font(size=font_size)
    out = annotate(
        image,
        objects,
        label_key=label_key,
        color_key=color_key,
        width=width,
        font=font,
    )
    out.save(output_path)
    return output_path
