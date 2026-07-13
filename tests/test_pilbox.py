"""Unit tests for the lite pilbox bounding-box annotation module."""

import numpy as np
import pytest
from PIL import Image

import pilbox


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


def test_annotate_custom_keys():
    im = Image.new("RGB", (100, 100), "white")
    objs = [{"cls": "player", "box": {"x0": 5, "y0": 5, "x1": 30, "y1": 30}}]
    out = pilbox.annotate(im, objs, label_key="cls", color_key="cls", bbox_key="box")
    assert out.size == (100, 100)


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
