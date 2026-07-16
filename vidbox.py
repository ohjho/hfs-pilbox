"""vidbox: annotate a video with per-frame bounding boxes and masks.

The video counterpart of ``pilbox.annotate``. It ties together the three lite
modules — frame I/O (:mod:`ffmpret`), bbox format conversion (:mod:`boxer`), and
drawing (:mod:`pilbox`) — to turn a video plus a flat list of per-frame
detections into an annotated video.

Detections are a flat list of dicts, each carrying a frame index, the box (in one
of the :data:`boxer.BBOX_FORMATS` conventions, read from configurable coordinate
keys) and, optionally, a label/color value and a base64-PNG mask. Every detection
for a frame is drawn onto that frame; a shared color map keeps each ``color_key``
value (e.g. a track id) one stable color across the whole video.

CLI (typer + loguru), kept separate from the lite modules it drives:

```bash
uv run python vidbox.py annotate-video-file in.mp4 dets.json out.mp4 \\
    --bbox-format coco_normalized
```
"""

import json
import sys
from collections import defaultdict

import typer
from loguru import logger
from PIL import Image
from typing import Literal

import boxer
import ffmpret
import pilbox

CROP_MODES = ("window", "box_fit")
GAP_BEHAVIORS = ("jump", "carry_forward")

logger.remove()
logger.add(
    sys.stderr,
    format="<d>{time:YYYY-MM-DD ddd HH:mm:ss}</d> | <lvl>{level}</lvl> | <lvl>{message}</lvl>",
)
app = typer.Typer(pretty_exceptions_show_locals=False)


@app.callback()
def _main():
    """Annotate videos with per-frame bounding boxes and masks."""


def annotate_video(
    video_path: str,
    detections,
    out_path: str,
    *,
    bbox_format: str = "coco_normalized",
    coord_keys=("x", "y", "w", "h"),
    frame_key: str = "frame",
    label_key: str = "track_id",
    color_key: str = "track_id",
    mask_key: str = "mask_b64",
    mask_alpha: float = 0.5,
    width: int = 3,
    font_size: int = 20,
) -> str:
    """Draw per-frame boxes and masks over a video and write the annotated result.

    Frames are extracted at native fps/resolution (so frame index ``i`` matches a
    detection's ``frame_key`` value, and full-frame masks line up), annotated with
    :func:`pilbox.annotate`, then re-encoded to a silent video at the source fps.

    Args:
        video_path: Path to the input video.
        detections: List of detection dicts. Each holds a frame index under
            ``frame_key``, box coordinates under ``coord_keys`` (interpreted per
            ``bbox_format``) and optionally ``label_key`` / ``color_key`` /
            ``mask_key`` values.
        out_path: Destination path for the annotated video.
        bbox_format: One of :data:`boxer.BBOX_FORMATS` describing the box values.
        coord_keys: The four object keys holding the box values, in order.
        frame_key: Object key holding the (0-based) frame index.
        label_key: Object key whose value is drawn as each box's label.
        color_key: Object key used to color-group boxes/masks (stable across frames).
        mask_key: Object key holding a base64-PNG mask (same size as the frame);
            pass ``""`` to disable masks.
        mask_alpha: Mask overlay opacity in ``[0, 1]``.
        width: Box outline width in pixels.
        font_size: Label font size in points.

    Returns:
        ``out_path``.

    Raises:
        ValueError: If ``bbox_format`` is unknown or no frames are decoded.
    """
    if bbox_format not in boxer.BBOX_FORMATS:
        raise ValueError(
            f"unknown bbox_format {bbox_format!r}; expected one of {list(boxer.BBOX_FORMATS)}"
        )

    vmeta = ffmpret.get_video_metadata(video_path, bverbose=False)
    org_w, org_h, fps = vmeta["width"], vmeta["height"], vmeta["fps"]

    # Extract every native frame so frame indices align with the detections.
    frames = ffmpret.extract_frames(video_path, fps=None, write_timestamp=False)
    if not frames:
        raise ValueError(f"no frames decoded from {video_path}")

    # Group detections by frame index.
    by_frame = defaultdict(list)
    for det in detections:
        by_frame[int(det[frame_key])].append(det)

    # Pre-seed a shared color map over all color_key values (sorted for
    # determinism) so each value keeps one color across every frame.
    color_map: dict = {}
    for cid in sorted({det.get(color_key) for det in detections}, key=repr):
        pilbox.color_for(cid, color_map)

    font = pilbox.load_pil_font(size=font_size)

    n_drawn = 0
    annotated = []
    for i, frame in enumerate(frames):
        objects = []
        for det in by_frame.get(i, []):
            coords = [det[k] for k in coord_keys]
            objects.append(
                {
                    **det,
                    "boundingBox": boxer.to_pascal_voc(
                        coords, bbox_format, org_w, org_h
                    ),
                }
            )
        n_drawn += len(objects)
        annotated.append(
            pilbox.annotate(
                frame,
                objects,
                label_key=label_key,
                color_key=color_key,
                bbox_key="boundingBox",
                mask_key=mask_key,
                mask_alpha=mask_alpha,
                width=width,
                font=font,
                color_map=color_map,
            )
        )

    # Warn if detections reference frames beyond what was decoded.
    max_frame = max(by_frame) if by_frame else -1
    if max_frame >= len(frames):
        logger.warning(
            f"{sum(len(v) for k, v in by_frame.items() if k >= len(frames))} detection(s) "
            f"reference frames >= {len(frames)} (decoded {len(frames)}); they were skipped"
        )

    logger.info(
        f"annotated {n_drawn} detection(s) across {len(frames)} frames "
        f"({len(color_map)} distinct {color_key!r})"
    )
    return ffmpret.frames_to_video(annotated, out_path, fps=fps)


def _even(n) -> int:
    """Round ``n`` down to the nearest even int >= 2 (yuv420p needs even dims)."""
    n = int(n)
    return max(2, n - (n % 2))


def _pad_clamp_box(box: dict, padding: float, im_w: int, im_h: int) -> dict:
    """Expand a pascal_voc box about its center by ``padding``, clamped to the frame."""
    cx = (box["x0"] + box["x1"]) / 2
    cy = (box["y0"] + box["y1"]) / 2
    bw = (box["x1"] - box["x0"]) * padding
    bh = (box["y1"] - box["y0"]) * padding
    x0 = max(0, int(round(cx - bw / 2)))
    y0 = max(0, int(round(cy - bh / 2)))
    x1 = min(im_w, int(round(cx + bw / 2)))
    y1 = min(im_h, int(round(cy + bh / 2)))
    # guarantee a non-empty box
    x1 = max(x1, x0 + 1)
    y1 = max(y1, y0 + 1)
    return {"x0": x0, "y0": y0, "x1": x1, "y1": y1}


def _crop_window(frame: Image.Image, box: dict, w: int, h: int) -> Image.Image:
    """Crop a fixed ``w`` x ``h`` window centered on ``box``, clamped inside ``frame``."""
    fw, fh = frame.size
    cx = (box["x0"] + box["x1"]) // 2
    cy = (box["y0"] + box["y1"]) // 2
    x0 = min(max(0, cx - w // 2), fw - w)
    y0 = min(max(0, cy - h // 2), fh - h)
    return pilbox.crop(frame, x0, y0, x0 + w, y0 + h)


def _gap_frame(frame: Image.Image, w: int, h: int, mode: str) -> Image.Image:
    """Frame for a gap (no detection) under ``jump``: centered window / black canvas."""
    if mode == "window":
        fw, fh = frame.size
        x0 = max(0, (fw - w) // 2)
        y0 = max(0, (fh - h) // 2)
        return pilbox.crop(frame, x0, y0, x0 + w, y0 + h)
    return Image.new("RGB", (w, h), (0, 0, 0))


def crop_video(
    video_path: str,
    detections,
    out_path: str,
    *,
    bbox_format: str = "coco_normalized",
    coord_keys=("x", "y", "w", "h"),
    frame_key: str = "frame",
    mode: Literal["window", "box_fit"] = "window",
    padding: float = 1.0,
    gap_behavior: Literal["jump", "carry_forward"] = "jump",
) -> str:
    """Crop a video to a subject that moves frame-to-frame, and write the result.

    The crop is a fixed-size **tracking window** whose size is the per-axis max of all
    per-frame boxes (times ``padding``, rounded even, clamped to the frame) so every box
    fits. Frames are extracted at native fps/resolution, cropped, and re-encoded to a silent
    video at the source fps. Masks (if any) in the detections are ignored — only boxes matter.

    Each frame must carry **at most one** box: exact-duplicate detection rows collapse to one,
    but a frame with two *different* boxes raises ``ValueError``. Frames with no detection are
    gaps, handled per ``gap_behavior``.

    Args:
        video_path: Path to the input video.
        detections: List of detection dicts, each with a frame index under ``frame_key`` and
            box values under ``coord_keys`` (interpreted per ``bbox_format``).
        out_path: Destination path for the cropped video.
        bbox_format: One of :data:`boxer.BBOX_FORMATS` describing the box values.
        coord_keys: The four object keys holding the box values, in order.
        frame_key: Object key holding the (0-based) frame index.
        mode: ``"window"`` (crop a fixed window from the frame, keeping surrounding scene, the
            window pans with the subject) or ``"box_fit"`` (crop exactly to the box, black-pad
            to the output aspect ratio, resize to fill — subject only, no scene).
        padding: Multiplier expanding each box about its center before sizing/cropping.
        gap_behavior: For frames with no detection — ``"jump"`` (window: center on the frame;
            box_fit: a black frame) or ``"carry_forward"`` (reuse the previous output frame).

    Returns:
        ``out_path``.

    Raises:
        ValueError: On an unknown ``mode`` / ``gap_behavior`` / ``bbox_format``, a frame with
            conflicting boxes, no usable detections, or no decoded frames.
    """
    if mode not in CROP_MODES:
        raise ValueError(f"unknown mode {mode!r}; expected one of {list(CROP_MODES)}")
    if gap_behavior not in GAP_BEHAVIORS:
        raise ValueError(
            f"unknown gap_behavior {gap_behavior!r}; expected one of {list(GAP_BEHAVIORS)}"
        )
    if bbox_format not in boxer.BBOX_FORMATS:
        raise ValueError(
            f"unknown bbox_format {bbox_format!r}; expected one of {list(boxer.BBOX_FORMATS)}"
        )

    # Group by frame; collapse exact-duplicate boxes; error on conflicting boxes.
    # Validate the JSON before touching ffmpeg so bad input fails fast.
    grouped = defaultdict(list)
    for det in detections:
        grouped[int(det[frame_key])].append(det)

    coords_by_frame = {}  # frame index -> the frame's single (deduped) coord tuple
    conflicts = []
    for f, dets in grouped.items():
        distinct = {tuple(det[k] for k in coord_keys) for det in dets}
        if len(distinct) > 1:
            conflicts.append(f)
            continue
        coords_by_frame[f] = next(iter(distinct))
    if conflicts:
        raise ValueError(
            "expected at most one box per frame; frames with conflicting boxes: "
            f"{sorted(conflicts)[:20]}"
        )
    if not coords_by_frame:
        raise ValueError("no detections to crop from")

    vmeta = ffmpret.get_video_metadata(video_path, bverbose=False)
    org_w, org_h, fps = vmeta["width"], vmeta["height"], vmeta["fps"]

    by_frame = {  # frame index -> padded, clamped pascal_voc box
        f: _pad_clamp_box(
            boxer.to_pascal_voc(coords, bbox_format, org_w, org_h),
            padding,
            org_w,
            org_h,
        )
        for f, coords in coords_by_frame.items()
    }

    # Fixed output size = per-axis max box, even, clamped to the frame.
    out_w = _even(min(org_w, max(b["x1"] - b["x0"] for b in by_frame.values())))
    out_h = _even(min(org_h, max(b["y1"] - b["y0"] for b in by_frame.values())))

    frames = ffmpret.extract_frames(video_path, fps=None, write_timestamp=False)
    if not frames:
        raise ValueError(f"no frames decoded from {video_path}")

    cropped = []
    last = None
    for i, frame in enumerate(frames):
        box = by_frame.get(i)
        if box is not None:
            if mode == "window":
                out = _crop_window(frame, box, out_w, out_h)
            else:  # box_fit
                sub = pilbox.crop(frame, box["x0"], box["y0"], box["x1"], box["y1"])
                out = pilbox.letterbox(sub, out_w, out_h)
            last = out
        elif gap_behavior == "carry_forward" and last is not None:
            out = last
        else:  # jump, or carry_forward before any boxed frame
            out = _gap_frame(frame, out_w, out_h, mode)
        cropped.append(out)

    logger.info(
        f"cropped {len(by_frame)} boxed frame(s) of {len(frames)} to {out_w}x{out_h} "
        f"(mode={mode}, gap={gap_behavior})"
    )
    return ffmpret.frames_to_video(cropped, out_path, fps=fps)


@app.command()
def annotate_video_file(
    video_path: str,
    json_path: str,
    out_path: str,
    bbox_format: str = "coco_normalized",
    coord_keys: str = "x,y,w,h",
    frame_key: str = "frame",
    label_key: str = "track_id",
    color_key: str = "track_id",
    mask_key: str = "mask_b64",
    mask_alpha: float = 0.5,
    width: int = 3,
    font_size: int = 20,
) -> str:
    """Annotate ``video_path`` using detections from a JSON file; save to ``out_path``.

    ``json_path`` holds a JSON list of detection dicts (see :func:`annotate_video`).
    ``coord_keys`` is a comma-separated list of the four coordinate keys.
    """
    with open(json_path) as f:
        detections = json.load(f)
    return annotate_video(
        video_path,
        detections,
        out_path,
        bbox_format=bbox_format,
        coord_keys=tuple(k.strip() for k in coord_keys.split(",")),
        frame_key=frame_key,
        label_key=label_key,
        color_key=color_key,
        mask_key=mask_key,
        mask_alpha=mask_alpha,
        width=width,
        font_size=font_size,
    )


@app.command()
def crop_video_file(
    video_path: str,
    json_path: str,
    out_path: str,
    bbox_format: str = "coco_normalized",
    coord_keys: str = "x,y,w,h",
    frame_key: str = "frame",
    mode: str = "window",
    padding: float = 1.0,
    gap_behavior: str = "jump",
) -> str:
    """Crop ``video_path`` to a moving subject using boxes from a JSON file; save to ``out_path``.

    ``json_path`` holds a JSON list of detection dicts — one box per frame (see
    :func:`crop_video`; any masks in the JSON are ignored). The output is a constant-size,
    silent video sized to the largest box, so the crop follows the subject across frames.

    Args:
        video_path: Path to the input video.
        json_path: Path to the detections JSON (a flat list of per-frame box dicts).
        out_path: Destination path for the cropped video.
        bbox_format: Box convention in the JSON — one of pascal_voc / albumentations / coco /
            coco_normalized.
        coord_keys: Comma-separated names of the four coordinate keys, in order (e.g. "x,y,w,h").
        frame_key: Object key holding the 0-based frame index.
        mode: How each frame is cropped. "window" keeps the surrounding scene — a fixed-size
            window is cropped from the frame and re-centered on the box, panning to follow the
            subject. "box_fit" shows the subject only — crop exactly to the box, black-pad to the
            output aspect ratio (no stretching), then resize to fill.
        padding: Multiplier that expands each box about its center before sizing/cropping. 1.0 =
            box as-is; e.g. 1.2 adds ~20% margin around the subject; larger values zoom out more.
        gap_behavior: What to show on frames with no detection. "jump" centers the window on the
            frame ("window") or emits a black frame ("box_fit"); "carry_forward" repeats the
            previous output frame.
    """
    with open(json_path) as f:
        detections = json.load(f)
    return crop_video(
        video_path,
        detections,
        out_path,
        bbox_format=bbox_format,
        coord_keys=tuple(k.strip() for k in coord_keys.split(",")),
        frame_key=frame_key,
        mode=mode,
        padding=padding,
        gap_behavior=gap_behavior,
    )


if __name__ == "__main__":
    app()
