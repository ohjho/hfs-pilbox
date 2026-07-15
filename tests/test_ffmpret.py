"""Unit tests for the ffmpret video-extraction module.

The pure-Python logic (frame-name parsing, metadata parsing / duration fallback)
is tested with a mocked ``ffmpeg.probe``. The end-to-end extraction test needs the
``ffmpeg`` binary and is skipped when it is unavailable.
"""

import shutil
import subprocess

import pytest

import ffmpret

FFMPEG = shutil.which("ffmpeg")


def test_parse_frame_name_simple():
    assert ffmpret.parse_frame_name("clip_12.jpg") == ("clip", 12)


def test_parse_frame_name_with_underscores():
    # frame types containing underscores must round-trip (rsplit on last "_")
    assert ffmpret.parse_frame_name("my_clip_12.jpg") == ("my_clip", 12)


def test_get_video_metadata_duration_falls_back_to_format(monkeypatch):
    # the video stream carries no "duration" (as with many MKV/WebM files);
    # it must fall back to the container/format duration rather than 0.
    fake_probe = {
        "streams": [
            {
                "codec_type": "video",
                "width": 320,
                "height": 240,
                "r_frame_rate": "30/1",
                "codec_name": "h264",
            }
        ],
        "format": {"duration": "12.5", "format_name": "matroska", "size": "1000"},
    }
    monkeypatch.setattr(ffmpret.ffmpeg, "probe", lambda path: fake_probe)

    meta = ffmpret.get_video_metadata("dummy.mkv", bverbose=False)

    assert meta["duration"] == 12.5
    assert meta["fps"] == 30.0
    assert meta["width"] == 320 and meta["height"] == 240
    assert meta["audio_codec"] == "none"


def test_get_video_metadata_raises_without_video_stream(monkeypatch):
    # a probe failure / no video stream should surface a legible error,
    # not silently return None for callers to subscript.
    monkeypatch.setattr(
        ffmpret.ffmpeg, "probe", lambda path: {"streams": [], "format": {}}
    )
    with pytest.raises(ValueError):
        ffmpret.get_video_metadata("dummy.mp4", bverbose=False)


@pytest.fixture
def tiny_video(tmp_path):
    """Generate a 1s, 10fps, 64x48 test clip via ffmpeg (skips if no binary)."""
    if FFMPEG is None:
        pytest.skip("ffmpeg binary not available")
    path = tmp_path / "testsrc.mp4"
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


def test_extract_frames_returns_expected_frames(tiny_video, tmp_path):
    out_dir = tmp_path / "frames"
    out_dir.mkdir()

    # write_timestamp=False avoids the drawtext/font dependency; source short
    # edge (48) < default max_short_edge so no scaling -> frames stay 64x48.
    frames = ffmpret.extract_frames(
        tiny_video,
        fps=5,
        write_timestamp=False,
        output_dir=str(out_dir),
    )

    # 1 second sampled at 5 fps -> ~5 frames (allow rounding at the boundary)
    assert 4 <= len(frames) <= 6
    assert all(f.size == (64, 48) for f in frames)
    # every returned frame was also written to disk
    assert len(list(out_dir.glob("*.jpg"))) == len(frames)


def test_get_video_metadata_on_real_clip(tiny_video):
    meta = ffmpret.get_video_metadata(tiny_video, bverbose=False)
    assert meta["width"] == 64 and meta["height"] == 48
    assert meta["duration"] == pytest.approx(1.0, abs=0.3)
    assert meta["fps"] == pytest.approx(10.0, abs=0.1)
