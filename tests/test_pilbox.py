"""Unit tests for the lite pilbox bounding-box annotation module."""

import base64
import io

import numpy as np
import pytest
from PIL import Image, ImageColor

import pilbox


def _b64_mask(mask_bool):
    """Encode a boolean numpy mask as a base64 PNG string (as stored in the JSON)."""
    im = Image.fromarray(mask_bool.astype(np.uint8) * 255).convert("1")
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _sample_objects():
    return [
        {"object_id": 0, "boundingBox": {"x0": 10, "y0": 10, "x1": 40, "y1": 60}},
        {"object_id": 1, "boundingBox": {"x0": 50, "y0": 20, "x1": 90, "y1": 80}},
    ]


def test_annotate_returns_same_size_copy_without_mutating_input():
    im = Image.new("RGB", (100, 100), "white")
    before = np.array(im).copy()

    out = pilbox.annotate(im, _sample_objects())

    assert isinstance(out, Image.Image)
    assert out.size == im.size
    # input is untouched
    assert np.array_equal(np.array(im), before)
    # output actually drew something (differs from a blank image)
    assert not np.array_equal(np.array(out), before)


def test_color_for_is_stable_and_distinct():
    mapping = {}
    a1 = pilbox.color_for("a", mapping)
    b = pilbox.color_for("b", mapping)
    a2 = pilbox.color_for("a", mapping)

    assert a1 == a2  # stable across calls
    assert a1 != b  # distinct keys -> distinct colors
    assert a1.startswith("#") and len(a1) == 7


def test_palette_color_never_repeats_consecutively():
    colors = [pilbox.palette_color(i) for i in range(50)]
    assert all(c.startswith("#") and len(c) == 7 for c in colors)
    # consecutive golden-angle hues are always far apart
    assert all(colors[i] != colors[i + 1] for i in range(len(colors) - 1))


def test_im_draw_bbox_coerces_float_coords():
    im = Image.new("RGB", (100, 100), "white")
    before = np.array(im).copy()

    # floats must not raise and must draw
    pilbox.im_draw_bbox(im, 10.5, 10.9, 40.2, 60.7, color="red", caption="x")

    assert not np.array_equal(np.array(im), before)


def test_annotate_shared_color_map_stable_across_calls():
    # A shared color_map must give a key the SAME color regardless of the object
    # order it appears in — the property that keeps a track id one color across
    # video frames. object_id 1 appears first in call A and second in call B.
    im = Image.new("RGB", (60, 60), "white")
    shared = {}

    a = [{"object_id": 1, "boundingBox": {"x0": 2, "y0": 2, "x1": 10, "y1": 10}}]
    b = [
        {"object_id": 0, "boundingBox": {"x0": 2, "y0": 2, "x1": 10, "y1": 10}},
        {"object_id": 1, "boundingBox": {"x0": 20, "y0": 20, "x1": 30, "y1": 30}},
    ]
    pilbox.annotate(im, a, color_map=shared)
    pilbox.annotate(im, b, color_map=shared)

    # object_id 1 was seen first -> palette index 0 in BOTH calls; 0 -> index 1.
    assert shared[1] == 0
    assert shared[0] == 1


def test_annotate_default_color_map_is_per_call():
    # Without a shared map each call colors independently (backward compatible).
    im = Image.new("RGB", (60, 60), "white")
    objs = [{"object_id": 7, "boundingBox": {"x0": 2, "y0": 2, "x1": 10, "y1": 10}}]
    out = pilbox.annotate(im, objs)  # no color_map arg
    assert out.size == (60, 60)


def test_annotate_custom_keys():
    im = Image.new("RGB", (100, 100), "white")
    objs = [{"cls": "player", "box": {"x0": 5, "y0": 5, "x1": 30, "y1": 30}}]
    out = pilbox.annotate(im, objs, label_key="cls", color_key="cls", bbox_key="box")
    assert out.size == (100, 100)


def test_letterbox_size_aspect_and_bars():
    # a 100x50 (2:1) image fit into an 80x80 (1:1) canvas -> letterboxed
    im = Image.new("RGB", (100, 50), (255, 255, 255))
    out = pilbox.letterbox(im, 80, 80)

    assert isinstance(out, Image.Image)
    assert out.size == (80, 80)  # exact target size
    # width fills (80), height scales to 40 -> 20px black bars top and bottom
    assert out.getpixel((40, 2)) == (0, 0, 0)  # top bar black
    assert out.getpixel((40, 78)) == (0, 0, 0)  # bottom bar black
    assert out.getpixel((40, 40)) == (255, 255, 255)  # center holds the image


def test_letterbox_pillarbox_and_custom_fill():
    # a 50x100 (tall) image into 80x80 -> pillarbox (side bars); custom fill color
    im = Image.new("RGB", (50, 100), (255, 255, 255))
    out = pilbox.letterbox(im, 80, 80, fill=(0, 0, 255))
    assert out.size == (80, 80)
    assert out.getpixel((2, 40)) == (0, 0, 255)  # left bar is the fill color
    assert out.getpixel((40, 40)) == (255, 255, 255)  # center holds the image


def test_crop_returns_expected_region():
    im = Image.new("RGB", (100, 100), "white")
    # paint a red patch so we can confirm the crop grabs the right pixels
    im.paste((255, 0, 0), (10, 20, 40, 80))

    out = pilbox.crop(im, 10, 20, 40, 80)

    assert isinstance(out, Image.Image)
    assert out.size == (30, 60)  # (x1 - x0, y1 - y0)
    assert np.array_equal(np.array(out), np.full((60, 30, 3), (255, 0, 0), dtype=np.uint8))


def test_crop_does_not_mutate_input():
    im = Image.new("RGB", (100, 100), "white")
    before = np.array(im).copy()

    out = pilbox.crop(im, 10, 20, 40, 80)

    assert out is not im
    assert np.array_equal(np.array(im), before)


def test_crop_rejects_invalid_box():
    im = Image.new("RGB", (100, 100), "white")
    with pytest.raises(ValueError):
        pilbox.crop(im, 40, 20, 10, 80)  # inverted x (x1 <= x0)
    with pytest.raises(ValueError):
        pilbox.crop(im, 10, 20, 40, 20)  # empty y (y1 <= y0)
    with pytest.raises(ValueError):
        pilbox.crop(im, 10, 20, 200, 80)  # x1 exceeds image width


def test_im_color_mask_blends_masked_pixels_only():
    img = np.zeros((4, 4, 3), dtype=np.uint8)  # black
    mask = np.zeros((4, 4), dtype=bool)
    mask[0, 0] = True

    out = pilbox.im_color_mask(img, mask, rgb_tup=(255, 0, 0), alpha=0.5)

    assert out.dtype == np.uint8 and out.shape == (4, 4, 3)
    assert tuple(int(v) for v in out[0, 0]) == (127, 0, 0)  # masked -> halfway to red
    assert tuple(int(v) for v in out[1, 1]) == (0, 0, 0)  # unmasked untouched


def test_im_color_mask_rejects_shape_mismatch():
    img = np.zeros((4, 4, 3), dtype=np.uint8)
    bad_mask = np.zeros((4, 2), dtype=bool)  # same height, wrong width
    with pytest.raises(ValueError):
        pilbox.im_color_mask(img, bad_mask)


def test_annotate_mask_and_box_share_color():
    im = Image.new("RGB", (40, 40), "white")
    # object 1's mask covers a patch clear of every box outline
    mask1 = np.zeros((40, 40), dtype=bool)
    mask1[25:35, 25:35] = True
    objs = [
        {"object_id": 0, "boundingBox": {"x0": 2, "y0": 2, "x1": 10, "y1": 10}},
        {
            "object_id": 1,
            "boundingBox": {"x0": 15, "y0": 15, "x1": 22, "y1": 22},
            "b64_mask": _b64_mask(mask1),
        },
    ]

    out = pilbox.annotate(im, objs, mask_alpha=1.0)
    arr = np.array(out)

    # object 1's palette color (index 1, since it's the 2nd distinct object_id)
    expected = ImageColor.getrgb(pilbox.palette_color(1))
    # a pixel well inside object 1's mask region is filled with that exact color
    assert tuple(int(v) for v in arr[30, 30]) == expected


def test_annotate_empty_mask_key_skips_masks():
    im = Image.new("RGB", (40, 40), "white")
    mask = np.ones((40, 40), dtype=bool)
    objs = [
        {
            "object_id": 0,
            "boundingBox": {"x0": 2, "y0": 2, "x1": 10, "y1": 10},
            "b64_mask": _b64_mask(mask),
        }
    ]
    # a full-image mask would repaint the corners if drawn; with masks disabled
    # the far corner stays white.
    out = pilbox.annotate(im, objs, mask_key="", mask_alpha=1.0)
    assert tuple(int(v) for v in np.array(out)[38, 38]) == (255, 255, 255)


def _fg_bg_image_and_mask():
    """A 4x4 solid image plus a center 2x2 foreground mask."""
    img = np.zeros((4, 4, 3), dtype=np.uint8)
    img[:, :] = (200, 100, 50)
    mask = np.zeros((4, 4), dtype=bool)
    mask[1:3, 1:3] = True
    return img, mask


def test_im_apply_mask_color_background_keeps_foreground():
    img, mask = _fg_bg_image_and_mask()
    out = pilbox.im_apply_mask(img, mask, bg_rgb_tup=(0, 0, 0))
    assert out.shape == (4, 4, 3)
    assert tuple(int(v) for v in out[1, 1]) == (200, 100, 50)  # foreground kept
    assert tuple(int(v) for v in out[0, 0]) == (0, 0, 0)  # background blacked out


def test_im_apply_mask_rejects_shape_mismatch():
    img, mask = _fg_bg_image_and_mask()
    with pytest.raises(ValueError):
        pilbox.im_apply_mask(img, mask[:, :2], bg_rgb_tup=(0, 0, 0))


def test_im_apply_mask_bg_options_do_not_raise():
    # regression guard: these paths use ImageFilter / ImageOps, which must be imported.
    img, mask = _fg_bg_image_and_mask()
    assert pilbox.im_apply_mask(img, mask, bg_blur_radius=2).shape == (4, 4, 3)
    assert pilbox.im_apply_mask(img, mask, bg_greyscale=True).shape == (4, 4, 3)
    assert pilbox.im_apply_mask(
        img, mask, bg_rgb_tup=(0, 0, 0), mask_gblur_radius=2
    ).shape == (4, 4, 3)
    # no bg_* option -> transparent (RGBA) result
    assert pilbox.im_apply_mask(img, mask).shape == (4, 4, 4)


def test_apply_mask_wrapper_masks_background():
    im = Image.new("RGB", (4, 4), (200, 100, 50))
    _, mask = _fg_bg_image_and_mask()
    out = pilbox.apply_mask(im, _b64_mask(mask), bg_rgb_tup=(0, 0, 0))
    assert isinstance(out, Image.Image) and out.mode == "RGB"
    assert out.getpixel((1, 1)) == (200, 100, 50)  # foreground kept
    assert out.getpixel((0, 0)) == (0, 0, 0)  # background blacked out


def test_apply_mask_rejects_bad_base64():
    im = Image.new("RGB", (4, 4), "white")
    with pytest.raises(ValueError):
        pilbox.apply_mask(im, "not-valid-base64-png!!", bg_rgb_tup=(0, 0, 0))
