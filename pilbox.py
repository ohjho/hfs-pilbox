"""PILBox: ultra-lite bounding-box annotation using only numpy + Pillow.

Bounding boxes use the **pascal_voc** convention: absolute pixel coordinates
``(x0, y0, x1, y1)`` for the top-left and bottom-right corners, matching the
``boundingBox`` dict found in ``assets/example_0.json``.

The only third-party dependencies are ``numpy``, ``Pillow`` and ``loguru``.
"""

import os
import io
import json
import base64
import colorsys

import numpy as np
from PIL import Image, ImageColor, ImageDraw, ImageFilter, ImageFont, ImageOps
from loguru import logger


# --------------------------------------------------------------------------- #
# Image helpers (kept for future features)
# --------------------------------------------------------------------------- #
def im_crop(im_rgb_array, x0, y0, x1, y1):
    """Crop an RGB numpy image to the pascal_voc box ``(x0, y0, x1, y1)``."""
    return im_rgb_array[y0:y1, x0:x1, :]


def crop(image: Image.Image, x0, y0, x1, y1) -> Image.Image:
    """Return the pascal_voc region ``(x0, y0, x1, y1)`` of ``image`` as a new image.

    A PIL-friendly wrapper around :func:`im_crop`: it validates the box, converts
    to RGB, and returns a new ``PIL.Image.Image`` without mutating the input.

    Args:
        image: Source ``PIL.Image.Image`` (never mutated; a new image is returned).
        x0, y0, x1, y1: pascal_voc box — top-left ``(x0, y0)`` and bottom-right
            ``(x1, y1)`` corners in absolute pixels (floats are coerced to int).

    Returns:
        A new ``PIL.Image.Image`` containing only the cropped region.

    Raises:
        ValueError: If the box is empty/inverted (``x1 <= x0`` or ``y1 <= y0``) or
            falls outside the image bounds.

    >>> im = Image.new("RGB", (100, 100), "white")
    >>> out = crop(im, 10, 20, 40, 80)
    >>> out.size
    (30, 60)
    >>> out is im
    False
    """
    x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
    w, h = image.size
    if x1 <= x0 or y1 <= y0:
        raise ValueError(
            f"crop box is empty: need x1 > x0 and y1 > y0, got "
            f"(x0={x0}, y0={y0}, x1={x1}, y1={y1})"
        )
    if x0 < 0 or y0 < 0 or x1 > w or y1 > h:
        raise ValueError(
            f"crop box ({x0}, {y0}, {x1}, {y1}) falls outside the "
            f"image bounds ({w}x{h})"
        )
    arr = np.asarray(image.convert("RGB"))
    cropped = im_crop(arr, x0=x0, y0=y0, x1=x1, y1=y1)
    return Image.fromarray(cropped)


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


def im_color_mask(
    im_rgb_array, mask_array, rgb_tup=(91, 86, 188), alpha=0.5, get_pil_im=False
):
    """Overlay a translucent solid-color mask onto an RGB image.

    Pixels where ``mask_array`` is truthy are blended toward ``rgb_tup`` by
    ``alpha``; the rest are left untouched.

    Args:
        im_rgb_array: RGB image as a ``(H, W, 3)`` ``uint8`` numpy array.
        mask_array: Boolean/0-1 mask, shape ``(H, W)`` matching the image.
        rgb_tup: Fill color as an ``(r, g, b)`` tuple.
        alpha: Blend strength in ``[0, 1]`` — ``0`` fully transparent (image
            unchanged), ``1`` fully opaque (solid fill over the mask).
        get_pil_im: If True, return a ``PIL.Image.Image``; otherwise a numpy array.

    Returns:
        The composited image, as a ``PIL.Image.Image`` when ``get_pil_im`` else a
        ``(H, W, 3)`` ``uint8`` numpy array.

    Raises:
        ValueError: If the image and mask height/width do not match.

    >>> import numpy as np
    >>> img = np.zeros((2, 2, 3), dtype=np.uint8)  # black
    >>> mask = np.array([[True, False], [False, False]])
    >>> out = im_color_mask(img, mask, rgb_tup=(255, 0, 0), alpha=0.5)
    >>> tuple(int(v) for v in out[0, 0])  # masked pixel blended halfway to red
    (127, 0, 0)
    >>> tuple(int(v) for v in out[1, 1])  # unmasked pixel untouched
    (0, 0, 0)
    """
    if im_rgb_array.shape[:2] != mask_array.shape[:2]:
        raise ValueError(
            f"im_color_mask: image is shape {im_rgb_array.shape[:2]} which is different than mask shape {mask_array.shape[:2]}"
        )

    bg_im = np.zeros(im_rgb_array.shape, dtype=np.uint8)  # create color
    bg_im[:, :] = rgb_tup
    # Coerce the mask to a uint8 alpha layer; a bool*int product is int64,
    # which Image.fromarray rejects.
    mask_l = (mask_array.astype(bool) * int(alpha * 255)).astype(np.uint8)
    im = Image.composite(
        Image.fromarray(bg_im),
        Image.fromarray(im_rgb_array),
        Image.fromarray(mask_l),
    )
    return im if get_pil_im else np.array(im)


def im_apply_mask(
    im_rgb_array,
    mask_array,
    get_pil_im=False,
    bg_rgb_tup=None,
    bg_blur_radius=None,
    bg_greyscale=False,
    mask_gblur_radius=0,
):
    """Keep the masked foreground and replace/transform the background.

    Pixels where ``mask_array`` is truthy keep their original value; the rest (the
    background) are handled per the mutually-exclusive ``bg_*`` options below —
    checked in order ``bg_rgb_tup`` → ``bg_blur_radius`` → ``bg_greyscale``, and if
    none is set the background is made transparent (an RGBA result).

    Args:
        im_rgb_array: RGB image as a ``(H, W, 3)`` ``uint8`` numpy array.
        mask_array: Foreground mask, shape ``(H, W)`` matching the image
            (bool or 0/255); truthy = keep the original pixel.
        get_pil_im: If True return a ``PIL.Image.Image``; otherwise a numpy array.
        bg_rgb_tup: If given, fill the background with this ``(r, g, b)`` color
            (3-channel result).
        bg_blur_radius: If given (and no ``bg_rgb_tup``), Gaussian-blur the
            background by this radius (3-channel result).
        bg_greyscale: If True (and neither above), greyscale the background
            (3-channel result).
        mask_gblur_radius: If ``> 0``, Gaussian-blur the mask edge by this radius
            for a soft cutout.

    Returns:
        The composited image — a ``PIL.Image.Image`` when ``get_pil_im`` else a
        numpy array. Shape is ``(H, W, 3)`` for any ``bg_*`` option, or
        ``(H, W, 4)`` (RGBA, transparent background) when none is set.

    Raises:
        ValueError: If the image and mask height/width do not match.

    Refs: https://note.nkmk.me/en/python-pillow-paste/

    >>> import numpy as np
    >>> img = np.zeros((2, 2, 3), dtype=np.uint8)
    >>> img[:, :] = (200, 100, 50)
    >>> mask = np.array([[True, False], [False, False]])
    >>> out = im_apply_mask(img, mask, bg_rgb_tup=(0, 0, 0))
    >>> out.shape
    (2, 2, 3)
    >>> tuple(int(v) for v in out[0, 0])  # foreground kept
    (200, 100, 50)
    >>> tuple(int(v) for v in out[1, 1])  # background -> black
    (0, 0, 0)
    """
    h, w, c = im_rgb_array.shape
    m_h, m_w = mask_array.shape

    if not all([h == m_h, w == m_w]):
        raise ValueError(
            f"im_apply_mask: mask_array size {(m_h, m_w)} must match im_rgb_array {(h, w)}"
        )

    im = Image.fromarray(im_rgb_array)

    # convert bitwise mask from np to pillow
    # ref: https://note.nkmk.me/en/python-pillow-paste/
    pil_mask = Image.fromarray(np.uint8(255 * mask_array))
    pil_mask = (
        pil_mask.filter(ImageFilter.GaussianBlur(radius=mask_gblur_radius))
        if mask_gblur_radius > 0
        else pil_mask
    )

    if bg_rgb_tup:
        bg_im = np.zeros([h, w, 3], dtype=np.uint8)  # black
        bg_im[:, :] = bg_rgb_tup  # apply color

        # old method using just np but doesn't support blurred mask
        # idx = (mask_array != 0)
        # bg_im[idx] = im_rgb_array[idx]

        bg_im = Image.fromarray(bg_im)
        bg_im.paste(im, mask=pil_mask)
        im = bg_im
    elif bg_blur_radius:
        bg_im = im.copy().filter(ImageFilter.GaussianBlur(radius=bg_blur_radius))
        bg_im.paste(im, mask=pil_mask)
        im = bg_im
    elif bg_greyscale:
        bg_im = ImageOps.grayscale(Image.fromarray(im_rgb_array))
        bg_im = np.array(bg_im)
        bg_im = np.stack((bg_im,) * 3, axis=-1)  # greyscale 1-channel to 3-channel

        bg_im = Image.fromarray(bg_im)
        bg_im.paste(im, mask=pil_mask)
        im = bg_im
    else:
        im.putalpha(pil_mask)

    return im if get_pil_im else np.array(im)


def _mask_from_b64(b64_str: str) -> np.ndarray:
    """Decode a base64-encoded PNG mask into a boolean ``(H, W)`` numpy array."""
    raw = base64.b64decode(b64_str)
    im = Image.open(io.BytesIO(raw))
    return np.asarray(im.convert("1"), dtype=bool)


def apply_mask(image: Image.Image, b64_mask: str, bg_rgb_tup=(0, 0, 0)) -> Image.Image:
    """Mask out ``image``'s background using a base64 PNG mask, on a new image.

    A PIL-friendly wrapper around :func:`im_apply_mask`: it decodes the mask,
    keeps the masked foreground, replaces the background with ``bg_rgb_tup``, and
    returns a new RGB ``PIL.Image.Image`` without mutating the input.

    Args:
        image: Source ``PIL.Image.Image`` (never mutated; a new image is returned).
        b64_mask: Base64-encoded PNG mask the same size as ``image``; non-zero
            pixels mark the foreground to keep.
        bg_rgb_tup: Background fill color as an ``(r, g, b)`` tuple (default black).

    Returns:
        A new ``PIL.Image.Image`` with the background filled by ``bg_rgb_tup``.

    Raises:
        ValueError: If ``b64_mask`` cannot be decoded, or the mask size does not
            match the image.

    >>> im = Image.new("RGB", (4, 4), (200, 100, 50))
    >>> import base64, io
    >>> import numpy as np
    >>> m = np.zeros((4, 4), dtype=np.uint8); m[1:3, 1:3] = 255
    >>> buf = io.BytesIO(); Image.fromarray(m).convert("1").save(buf, format="PNG")
    >>> out = apply_mask(im, base64.b64encode(buf.getvalue()).decode())
    >>> out.size
    (4, 4)
    >>> out.getpixel((1, 1)), out.getpixel((0, 0))
    ((200, 100, 50), (0, 0, 0))
    """
    try:
        mask = _mask_from_b64(b64_mask)
    except Exception as e:
        raise ValueError(f"could not decode base64 PNG mask: {e}")
    arr = np.asarray(image.convert("RGB"))
    return im_apply_mask(arr, mask, bg_rgb_tup=tuple(bg_rgb_tup), get_pil_im=True)


def annotate(
    image: Image.Image,
    objects,
    *,
    label_key: str = "object_id",
    color_key: str = "object_id",
    bbox_key: str = "boundingBox",
    mask_key: str = "b64_mask",
    mask_alpha: float = 0.5,
    width: int = 3,
    font=None,
    color_map: dict = None,
) -> Image.Image:
    """Draw every object's pascal_voc box (and optional mask) on a copy of ``image``.

    When ``mask_key`` is truthy, each object carrying a base64-encoded PNG mask
    under that key gets a translucent overlay drawn *beneath* the boxes, colored
    to match its own bounding box (both derive from the same ``color_key`` value,
    so an object's box and mask always share one color).

    Args:
        image: Source ``PIL.Image.Image`` (never mutated; a copy is returned).
        objects: Iterable of dicts. Each must hold a pascal_voc box under
            ``bbox_key`` as ``{"x0", "y0", "x1", "y1"}`` and, optionally, values
            under ``label_key``, ``color_key`` and ``mask_key``.
        label_key: Object key whose value is drawn as the box label.
        color_key: Object key whose value groups boxes into colors (each distinct
            value gets its own palette color).
        bbox_key: Object key holding the pascal_voc box dict.
        mask_key: Object key holding a base64-encoded PNG mask. Falsy (e.g. ``""``)
            disables mask drawing entirely; objects lacking the key are skipped.
        mask_alpha: Mask overlay opacity in ``[0, 1]``.
        width: Box outline width in pixels.
        font: Optional ``PIL.ImageFont`` for labels; defaults to Pillow's default.
        color_map: Optional ``color_key value -> palette index`` dict, mutated in
            place. Pass a shared (optionally pre-seeded) map across many
            ``annotate`` calls to keep a key's color stable — e.g. so a track id
            keeps one color across every frame of a video. Defaults to a fresh
            per-call map (each image colored independently).

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
    color_map = {} if color_map is None else color_map
    # Assign palette indices once, in object order, so an object's mask and box
    # resolve to the identical color regardless of which is drawn first.
    for obj in objects:
        color_for(obj.get(color_key), color_map)

    # Pass 1: masks, composited beneath the boxes.
    if mask_key:
        arr = None
        for obj in objects:
            b64 = obj.get(mask_key)
            if not b64:
                continue
            if arr is None:
                arr = np.asarray(out.convert("RGB"))
            rgb = ImageColor.getrgb(color_for(obj.get(color_key), color_map))
            mask = _mask_from_b64(b64)
            arr = im_color_mask(arr, mask, rgb_tup=rgb, alpha=mask_alpha)
        if arr is not None:
            out = Image.fromarray(arr)

    # Pass 2: boxes and labels on top, reusing the shared color map.
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
    mask_key: str = "b64_mask",
    mask_alpha: float = 0.5,
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
        mask_key: Object key holding a base64-encoded PNG mask (falsy disables masks).
        mask_alpha: Mask overlay opacity in ``[0, 1]``.
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
        mask_key=mask_key,
        mask_alpha=mask_alpha,
        width=width,
        font=font,
    )
    out.save(output_path)
    return output_path
