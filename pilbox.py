def im_crop(im_rgb_array, x0, y0, x1, y1):
    return im_rgb_array[y0:y1, x0:x1, :]


def im_center_crop(im_rgb_array, w, h):
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
    """
    Attempts to load a font, with fallbacks.
    Args:
        font_name (str): The name or relative path of the font file (e.g., 'arial.ttf').
        size (int): The font size.
        fallback_to_default (bool): If True, falls back to Pillow's default font.
    Returns:
        PIL.ImageFont.FreeTypeFont: The loaded font object.
    Raises:
        OSError: If no font can be loaded and fallback is not enabled.
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
    return (
        ImageFont.load_default()
    )  # [9](https://www.geeksforgeeks.org/python/python-pil-imagefont-load_default/)


def im_draw_bbox(
    pil_im,
    x0,
    y0,
    x1,
    y1,
    color="black",
    width=3,
    caption=None,
    caption_font=ImageFont.load_default(),
):
    """
    draw bounding box on the input image pil_im in-place
    Args:
            color: color name as read by Pillow.ImageColor
    """

    if any([type(i) == float for i in [x0, y0, x1, y1]]):
        warnings.warn(
            f"im_draw_bbox: at least one of x0,y0,x1,y1 is of the type float and is converted to int."
        )
        x0 = int(x0)
        y0 = int(y0)
        x1 = int(x1)
        y1 = int(y1)

    draw = ImageDraw.Draw(pil_im)
    draw.rectangle([(x0, y0), (x1, y1)], outline=color, width=width)
    if caption:
        draw.text((x0, y0), text=caption, fill=color, font=caption_font)
    # return None


def im_draw_point(
    pil_im: Image.Image,
    x: int,
    y: int,
    caption: str = None,
    size: int = 10,
    width: int = 2,
    color: str = "red",
) -> Image.Image:
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
            # stroke_width=width,
            font_size=width * 5,
        )

    return im_draw
