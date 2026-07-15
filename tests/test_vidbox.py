"""Unit tests for the vidbox video-annotation module.

Pure logic (format validation, coordinate conversion) is tested directly; the
end-to-end ``annotate_video`` needs the ``ffmpeg`` binary and is skipped without
it, and the on-disk example assets are optional.
"""

import shutil
import subprocess
from pathlib import Path

import pytest

import vidbox

FFMPEG = shutil.which("ffmpeg")
ASSETS = Path(__file__).resolve().parent.parent / "assets"
EXAMPLE_VIDEO = ASSETS / "17078229_3222904.mp4"
EXAMPLE_JSON = ASSETS / "17078229_3222904-SAM2_tiny_ZeroGPU-with_mask.json"


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
