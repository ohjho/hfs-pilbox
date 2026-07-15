"""Unit tests for the boxer bounding-box utilities."""

import pytest

import boxer


def test_to_pascal_voc_pascal_voc_is_identity():
    assert boxer.to_pascal_voc((10, 20, 40, 60), "pascal_voc", 100, 100) == {
        "x0": 10, "y0": 20, "x1": 40, "y1": 60,
    }


def test_to_pascal_voc_albumentations_scales_by_dims():
    # normalized corners -> absolute, using distinct w/h to catch axis swaps
    assert boxer.to_pascal_voc((0.1, 0.2, 0.4, 0.6), "albumentations", 100, 200) == {
        "x0": 10, "y0": 40, "x1": 40, "y1": 120,
    }


def test_to_pascal_voc_coco_absolute_topleft_plus_size():
    assert boxer.to_pascal_voc((10, 20, 30, 40), "coco", 100, 100) == {
        "x0": 10, "y0": 20, "x1": 40, "y1": 60,
    }


def test_to_pascal_voc_coco_normalized_matches_reference_detection():
    # the reference SAM2 frame-118 / track-0 detection on the 1080x1920 video
    box = boxer.to_pascal_voc(
        (0.5157407407407407, 0.36041666666666666, 0.11944444444444445, 0.16145833333333334),
        "coco_normalized",
        1080,
        1920,
    )
    assert box == {"x0": 557, "y0": 692, "x1": 686, "y1": 1002}


def test_to_pascal_voc_rejects_unknown_format():
    with pytest.raises(ValueError):
        boxer.to_pascal_voc((0, 0, 1, 1), "yolo", 100, 100)


def test_get_bbox_dict_relative_vs_absolute():
    # absolute top-left + size
    assert boxer.get_bbox_dict(10, 20, 30, 40) == {"x0": 10, "y0": 20, "x1": 40, "y1": 60}
    # normalized via im_wh
    assert boxer.get_bbox_dict(0.1, 0.2, 0.3, 0.4, im_wh=(100, 200)) == {
        "x0": 10, "y0": 40, "x1": 40, "y1": 120,
    }


def test_bbox_intersects_cross_overlap():
    # tall thin box crossing a short wide one: they overlap although no corner of
    # either lies inside the other (the case the old corner-based test missed).
    a = {"x0": 4, "y0": 0, "x1": 6, "y1": 10}
    b = {"x0": 0, "y0": 4, "x1": 10, "y1": 6}
    assert boxer.bbox_intersects(a, b) is True


def test_bbox_intersects_disjoint():
    a = {"x0": 0, "y0": 0, "x1": 10, "y1": 10}
    b = {"x0": 20, "y0": 20, "x1": 30, "y1": 30}
    assert boxer.bbox_intersects(a, b) is False
