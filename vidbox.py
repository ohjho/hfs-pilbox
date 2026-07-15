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

import boxer
import ffmpret
import pilbox

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
                {**det, "boundingBox": boxer.to_pascal_voc(coords, bbox_format, org_w, org_h)}
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


if __name__ == "__main__":
    app()
