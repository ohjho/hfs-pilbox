"""Unit tests for the vidbox video-annotation module.

Pure logic (format validation, coordinate conversion) is tested directly; the
end-to-end ``annotate_video`` needs the ``ffmpeg`` binary and is skipped without
it, and the on-disk example assets are optional.
"""

import base64
import io
import shutil
import subprocess
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

import vidbox

FFMPEG = shutil.which("ffmpeg")
ASSETS = Path(__file__).resolve().parent.parent / "assets"
EXAMPLE_VIDEO = ASSETS / "17078229_3222904.mp4"
EXAMPLE_JSON = ASSETS / "17078229_3222904-SAM2_tiny_ZeroGPU-with_mask.json"
EXAMPLE_CROP_JSON = ASSETS / "17078229_3222904-VideoCrop-example.json"
EXAMPLE_MASK_JSON = ASSETS / "17078229_3222904-player1box.json"


def _b64_mask(w, h, box):
    """A full-frame base64-PNG mask (mode '1') with ``box``=(x0,y0,x1,y1) as foreground."""
    im = Image.new("1", (w, h), 0)
    ImageDraw.Draw(im).rectangle(box, fill=1)
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def test_annotate_video_rejects_unknown_format():
    with pytest.raises(ValueError):
        vidbox.annotate_video("x.mp4", [], "out.mp4", bbox_format="yolo")


@pytest.fixture
def tiny_video(tmp_path):
    """Generate a 1s, 10fps, 64x48 test clip via ffmpeg (skips if no binary)."""
    if FFMPEG is None:
        pytest.skip("ffmpeg binary not available")
    path = tmp_path / "src.mp4"
    subprocess.run(
        [
            FFMPEG, "-y", "-f", "lavfi",
            "-i", "testsrc=duration=1:size=64x48:rate=10",
            "-pix_fmt", "yuv420p", str(path),
        ],
        check=True,
        capture_output=True,
    )
    return str(path)


def _probe(path):
    out = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=width,height,nb_frames",
            "-of", "default=noprint_wrappers=1", str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return dict(line.split("=") for line in out.strip().splitlines())


def test_annotate_video_synthetic_coco_normalized(tiny_video, tmp_path):
    # two frames get a normalized-coco box; output must match source dims/frames.
    detections = [
        {"frame": 0, "track_id": 0, "x": 0.1, "y": 0.1, "w": 0.3, "h": 0.3, "conf": 1},
        {"frame": 3, "track_id": 1, "x": 0.5, "y": 0.5, "w": 0.2, "h": 0.2, "conf": 1},
    ]
    out = tmp_path / "annotated.mp4"
    result = vidbox.annotate_video(
        tiny_video, detections, str(out), bbox_format="coco_normalized",
        mask_key="",  # synthetic dets carry no mask
    )
    assert Path(result).is_file()
    meta = _probe(result)
    assert meta["width"] == "64" and meta["height"] == "48"
    # 1s @ 10fps native -> ~10 frames
    assert 8 <= int(meta["nb_frames"]) <= 12


def test_crop_video_rejects_conflicting_boxes():
    # same frame, two DIFFERENT boxes -> conflict error (needs no ffmpeg: raises
    # during grouping, before extraction).
    dets = [
        {"frame": 0, "x": 0.1, "y": 0.1, "w": 0.2, "h": 0.2},
        {"frame": 0, "x": 0.5, "y": 0.5, "w": 0.2, "h": 0.2},
    ]
    with pytest.raises(ValueError, match="conflicting"):
        vidbox.crop_video("x.mp4", dets, "out.mp4")


def test_crop_video_rejects_unknown_mode():
    with pytest.raises(ValueError):
        vidbox.crop_video("x.mp4", [], "out.mp4", mode="nope")


@pytest.mark.parametrize("mode", ["window", "box_fit"])
def test_crop_video_dedupes_and_produces_even_constant_size(tiny_video, tmp_path, mode):
    # frame 0 has an EXACT-duplicate row (must dedupe, not error); frames 3 & 6
    # add boxes so the max-box sizing has something to work with.
    dets = [
        {"frame": 0, "track_id": 0, "x": 0.1, "y": 0.1, "w": 0.30, "h": 0.30},
        {"frame": 0, "track_id": 0, "x": 0.1, "y": 0.1, "w": 0.30, "h": 0.30},  # dup
        {"frame": 3, "track_id": 0, "x": 0.2, "y": 0.2, "w": 0.25, "h": 0.40},
        {"frame": 6, "track_id": 0, "x": 0.5, "y": 0.5, "w": 0.20, "h": 0.20},
    ]
    out = tmp_path / f"crop_{mode}.mp4"
    result = vidbox.crop_video(tiny_video, dets, str(out), mode=mode, gap_behavior="jump")
    assert Path(result).is_file()
    meta = _probe(result)
    w, h = int(meta["width"]), int(meta["height"])
    assert w % 2 == 0 and h % 2 == 0  # even dims for yuv420p
    # per-axis max padded box on 64x48: max w=0.30*64=19 -> even 18; max h=0.40*48=19 -> even 18
    assert (w, h) == (18, 18)
    assert 8 <= int(meta["nb_frames"]) <= 12  # 1s @ 10fps native


@pytest.mark.skipif(
    not (EXAMPLE_VIDEO.exists() and EXAMPLE_JSON.exists()),
    reason="example assets not present (git-ignored)",
)
def test_annotate_video_end_to_end_on_assets(tmp_path):
    if FFMPEG is None:
        pytest.skip("ffmpeg binary not available")
    import json

    detections = json.loads(EXAMPLE_JSON.read_text())
    out = tmp_path / "assets_annotated.mp4"
    result = vidbox.annotate_video(
        str(EXAMPLE_VIDEO), detections, str(out), bbox_format="coco_normalized"
    )
    assert Path(result).is_file()
    meta = _probe(result)
    assert meta["width"] == "1080" and meta["height"] == "1920"
    assert int(meta["nb_frames"]) == 296  # every native frame preserved


def test_mask_video_rejects_conflicting_masks():
    # same frame, two DIFFERENT masks -> conflict error (raised before ffmpeg runs)
    dets = [
        {"frame": 0, "mask_b64": _b64_mask(64, 48, (0, 0, 10, 10))},
        {"frame": 0, "mask_b64": _b64_mask(64, 48, (20, 20, 40, 40))},
    ]
    with pytest.raises(ValueError, match="conflicting"):
        vidbox.mask_video("x.mp4", dets, "out.mp4")


def test_mask_video_rejects_unknown_gap_behavior():
    with pytest.raises(ValueError):
        vidbox.mask_video("x.mp4", [], "out.mp4", gap_behavior="nope")


def test_mask_video_skip_vs_fill_frame_counts(tiny_video, tmp_path):
    # masks on 3 frames; frame 0 has an EXACT-duplicate mask row (must dedupe, not error).
    m = _b64_mask(64, 48, (16, 12, 48, 36))
    dets = [
        {"frame": 0, "mask_b64": m},
        {"frame": 0, "mask_b64": m},  # exact duplicate -> collapses
        {"frame": 3, "mask_b64": m},
        {"frame": 6, "mask_b64": m},
    ]
    skip_out = vidbox.mask_video(tiny_video, dets, str(tmp_path / "skip.mp4"), gap_behavior="skip")
    fill_out = vidbox.mask_video(tiny_video, dets, str(tmp_path / "fill.mp4"), gap_behavior="fill")

    skip_meta, fill_meta = _probe(skip_out), _probe(fill_out)
    # both keep source dims
    assert skip_meta["width"] == "64" and skip_meta["height"] == "48"
    assert fill_meta["width"] == "64" and fill_meta["height"] == "48"
    # skip keeps only the 3 masked frames; fill keeps every native frame (~10 for 1s@10fps)
    assert int(skip_meta["nb_frames"]) == 3
    assert int(fill_meta["nb_frames"]) >= 8
    assert int(fill_meta["nb_frames"]) > int(skip_meta["nb_frames"])


@pytest.mark.skipif(
    not (EXAMPLE_VIDEO.exists() and EXAMPLE_MASK_JSON.exists()),
    reason="mask example assets not present (git-ignored)",
)
def test_mask_video_end_to_end_on_assets(tmp_path):
    if FFMPEG is None:
        pytest.skip("ffmpeg binary not available")
    import json

    # 218 of 296 frames carry a mask; frame 118 is an exact-duplicate row (dedupes)
    detections = json.loads(EXAMPLE_MASK_JSON.read_text())
    out = vidbox.mask_video(
        str(EXAMPLE_VIDEO), detections, str(tmp_path / "masked.mp4"), gap_behavior="skip"
    )
    assert Path(out).is_file()
    meta = _probe(out)
    assert meta["width"] == "1080" and meta["height"] == "1920"
    assert int(meta["nb_frames"]) == 218  # gaps skipped


@pytest.mark.skipif(
    not (EXAMPLE_VIDEO.exists() and EXAMPLE_CROP_JSON.exists()),
    reason="crop example assets not present (git-ignored)",
)
def test_crop_video_end_to_end_on_assets(tmp_path):
    if FFMPEG is None:
        pytest.skip("ffmpeg binary not available")
    import json

    # this reference has an exact-duplicate row at frame 118 that must dedupe cleanly
    detections = json.loads(EXAMPLE_CROP_JSON.read_text())
    out = tmp_path / "assets_cropped.mp4"
    result = vidbox.crop_video(
        str(EXAMPLE_VIDEO), detections, str(out), bbox_format="coco_normalized", mode="window"
    )
    assert Path(result).is_file()
    meta = _probe(result)
    w, h = int(meta["width"]), int(meta["height"])
    assert w % 2 == 0 and h % 2 == 0 and w <= 1080 and h <= 1920  # even, within frame
    assert int(meta["nb_frames"]) == 296  # every native frame preserved
