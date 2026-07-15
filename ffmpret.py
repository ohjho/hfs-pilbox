import json
import os
import sys
from typing import Optional

import ffmpeg
import typer
from loguru import logger
from PIL import Image
from tqdm import tqdm

logger.remove()
logger.add(
    sys.stderr,
    format="<d>{time:YYYY-MM-DD ddd HH:mm:ss}</d> | <lvl>{level}</lvl> | <lvl>{message}</lvl>",
)
app = typer.Typer(pretty_exceptions_show_locals=False)


def parse_frame_name(fname: str):
    """return a tuple of frame_type and frame_index

    Splits on the last underscore so frame types that themselves contain
    underscores (e.g. ``my_clip``) still round-trip.

    >>> parse_frame_name("clip_12.jpg")
    ('clip', 12)
    >>> parse_frame_name("my_clip_12.jpg")
    ('my_clip', 12)
    """
    fn, fext = os.path.splitext(os.path.basename(fname))
    frame_type, frame_index = fn.rsplit("_", 1)
    return frame_type, int(frame_index)


@app.command()
def get_video_metadata(video_path: str, bverbose: bool = True):
    """
    Extract comprehensive metadata from a video file.

    Args:
        video_path (str): Path to the video file

    Returns:
        dict: Dictionary containing video metadata including:
            - width, height: Video dimensions
            - duration: Video duration in seconds
            - fps: Frames per second
            - codec: Video codec name
            - bitrate: Video bitrate
            - format_name: Container format
            - file_size: File size in bytes
    """
    probe = ffmpeg.probe(video_path)

    # Find the first video stream
    video_stream = next(
        (stream for stream in probe["streams"] if stream["codec_type"] == "video"),
        None,
    )

    if video_stream is None:
        raise ValueError("No video stream found")

    # Get format information (also the fallback source for duration)
    format_info = probe.get("format", {})

    # Extract basic video properties
    width = int(video_stream.get("width", 0))
    height = int(video_stream.get("height", 0))
    # Duration is not always present on the video stream (e.g. MKV/WebM);
    # fall back to the container/format duration so callers don't silently get 0.
    duration = float(video_stream.get("duration") or format_info.get("duration", 0))

    # Calculate FPS
    r_frame_rate = video_stream.get("r_frame_rate", "0/1")
    num, denom = map(int, r_frame_rate.split("/"))
    fps = num / denom if denom != 0 else 0

    # Get codec and bitrate
    codec = video_stream.get("codec_name", "unknown")
    bitrate = (
        int(video_stream.get("bit_rate", 0)) if video_stream.get("bit_rate") else 0
    )

    format_name = format_info.get("format_name", "unknown")
    file_size = int(format_info.get("size", 0))

    # Get audio stream info if available
    audio_stream = next(
        (stream for stream in probe["streams"] if stream["codec_type"] == "audio"),
        None,
    )

    audio_codec = audio_stream.get("codec_name", "none") if audio_stream else "none"
    audio_bitrate = (
        int(audio_stream.get("bit_rate", 0))
        if audio_stream and audio_stream.get("bit_rate")
        else 0
    )

    metadata = {
        "width": width,
        "height": height,
        "duration": duration,
        "fps": fps,
        "video_codec": codec,
        "video_bitrate": bitrate,
        "audio_codec": audio_codec,
        "audio_bitrate": audio_bitrate,
        "format_name": format_name,
        "file_size": file_size,
        "total_streams": len(probe["streams"]),
    }
    if bverbose:
        logger.info(f"Video metadata extracted: {json.dumps(metadata, indent=4)}")
    return metadata


@app.command()
def extract_frames(
    input_path: str,
    fps: int = 8,
    max_short_edge: int = 1080,
    write_timestamp: bool = True,
    write_frame_num: bool = True,
    output_dir: Optional[str] = None,
    out_vid_path: Optional[str] = None,
    text_font_size: int = 20,
    text_y_position: str = "bottom",
):
    """
    Extract frames from a video file using FFmpeg.

    Args:
        input_path (str): Path to the input video file.
        fps (int): Frames per second to extract.
        max_short_edge (int): Maximum length of the shorter edge of the extracted frames.
        write_timestamp (bool): Whether to write the timestamp of each frame.
        write_frame_num (bool): Whether to write the frame number of each frame.
        output_dir (str): Directory to save the extracted frames.
        out_vid_path (str): Path to save the extracted frames as a video.
        text_font_size (int): Font size of the timestamp/frame-number overlay.
        text_y_position (str): Vertical placement of the overlay. One of
            "top", "middle", or "bottom" (default).

    Returns:
        List of PIL Images
    """
    y_position_map = {
        "top": "2*lh",
        "middle": "(h-lh)/2",
        "bottom": "h-(2*lh)",
    }
    assert (
        text_y_position in y_position_map
    ), f"text_y_position must be one of {list(y_position_map)}, got {text_y_position!r}"
    text_y_expr = y_position_map[text_y_position]

    if output_dir:
        assert os.path.isdir(
            output_dir
        ), f"Output directory {output_dir} does not exist"

    # Probe video to get width, height, and duration
    vmeta = get_video_metadata(input_path, bverbose=False)
    org_w, org_h = vmeta["width"], vmeta["height"]
    max_short_edge = int(max_short_edge) if max_short_edge else min(org_w, org_h)
    long_edge = int((max(org_h, org_w) / min(org_h, org_w)) * max_short_edge)
    long_edge += 0 if long_edge % 2 == 0 else 1
    duration = vmeta["duration"]
    org_fps = vmeta["fps"]
    if fps > org_fps:
        logger.debug(
            f"requested fps({fps}) exceeded source fps({org_fps}): fps will be capped to source fps({org_fps})"
        )
        fps = org_fps

    # Calculate total frames to extract based on fps and duration
    total_frames = int(duration * fps)

    # add scale filter only if needed
    add_scale_filter = max_short_edge < min(org_w, org_h)
    w = max_short_edge if org_w < org_h else long_edge
    h = max_short_edge if org_w > org_h else long_edge
    logger.debug(f"Video dimensions: {org_w}x{org_h}")
    if add_scale_filter:
        logger.debug(f"\tscaling video to {w}x{h}")

    # Set drawtext filter text
    drawtext_filter_text = (
        r"text='Timestamp\:%{pts\:hms} \|Frame Number\: %{frame_num}'"
        if write_frame_num
        else r"text='Timestamp\:%{pts\:hms}'"
    )

    # Setup the ffmpeg filter chain
    drawtext_filter = (
        f",drawtext={drawtext_filter_text}: x=(w-tw)/2: y={text_y_expr}: fontcolor=white: fontsize={text_font_size}: box=1: boxcolor=0x00000099: boxborderw=5"
        if write_timestamp
        else ""
    )
    scale_filter = (
        # f",scale='if(lt(iw, ih), {max_short_edge}, -2)':'if(lt(ih, iw), {max_short_edge}, -2)'"
        f",scale='{w}:{h}'"
        if add_scale_filter
        else ""
    )
    filter_chain = f"fps={fps}{drawtext_filter}{scale_filter}"

    # Run ffmpeg process with output as rawvideo piped to stdout.
    # NOTE: stderr is intentionally left to inherit (not piped). If it were
    # piped without being drained, ffmpeg would block once the OS pipe buffer
    # fills, deadlocking against our stdout-only read loop on longer clips.
    process = (
        ffmpeg.input(input_path)
        .output("pipe:", vf=filter_chain, format="rawvideo", pix_fmt="rgb24")
        .run_async(pipe_stdout=True)
    )
    logger.info(f"running ffmpeg with filter:\n{filter_chain}")

    frame_size = (
        long_edge * max_short_edge * 3 if add_scale_filter else org_w * org_h * 3
    )  # 3 bytes per pixel (RGB)
    frames = []

    # total_frames (duration * fps) is only an estimate for the progress bar;
    # read until the pipe is exhausted so we don't drop the tail frame or stop
    # before EOF when the estimate is slightly off.
    with tqdm(total=total_frames, desc="Extracting frames with FFMPEG") as pbar:
        while True:
            in_bytes = process.stdout.read(frame_size)
            if not in_bytes or len(in_bytes) < frame_size:
                break
            frame = Image.frombytes(
                "RGB", (w, h) if add_scale_filter else (org_w, org_h), in_bytes
            )
            frames.append(frame)
            pbar.update(1)

    process.stdout.close()
    process.wait()

    if output_dir:
        vname, _ = os.path.splitext(os.path.basename(input_path))
        for i, im in enumerate(tqdm(frames, desc=f"Saving frames to {output_dir}")):
            output_path = os.path.join(output_dir, f"{vname}_{i}.jpg")
            im.save(output_path)

    if out_vid_path:
        (
            ffmpeg.input(input_path)
            .output(
                out_vid_path,
                vf=filter_chain,
                vcodec="libx264",
                pix_fmt="yuv420p",
                r=fps,
                # add other options as needed
            )
            .run(overwrite_output=True)
        )
        logger.success(f"Video created at {out_vid_path}")

    return frames


@app.command()
def extract_specific_frames(
    input_path: str,
    timestamps_or_frames: list[str] = typer.Option(),
    max_short_edge: int = 1080,
    as_timestamps: bool = True,
    output_dir: Optional[str] = None,
):
    """
    Extract specific frames from a video file using FFmpeg at given timestamps or frame numbers.

    Args:
        input_path (str): Path to the input video file.
        timestamps_or_frames (list): List of timestamps (in seconds) or frame numbers to extract.
        max_short_edge (int): Maximum length of the shorter edge of the extracted frames.
        as_timestamps (bool): If True, treat input list as timestamps. If False, treat as frame numbers.
        output_dir (str): Directory to save the extracted frames as ``{vname}_{target}.jpg``.
            If None, frames are only returned in memory.

    Returns:
        List of PIL Images corresponding to the specified timestamps/frames
    """
    if output_dir:
        assert os.path.isdir(
            output_dir
        ), f"Output directory {output_dir} does not exist"
    vname, _ = os.path.splitext(os.path.basename(input_path))
    # Probe video to get width, height, and duration
    vmeta = get_video_metadata(input_path, bverbose=False)
    org_w, org_h = vmeta["width"], vmeta["height"]
    max_short_edge = int(max_short_edge) if max_short_edge else min(org_w, org_h)
    long_edge = int((max(org_h, org_w) / min(org_h, org_w)) * max_short_edge)
    long_edge += 0 if long_edge % 2 == 0 else 1
    duration = vmeta["duration"]
    org_fps = vmeta["fps"]

    # add scale filter only if needed
    add_scale_filter = max_short_edge < min(org_w, org_h)
    w = max_short_edge if org_w < org_h else long_edge
    h = max_short_edge if org_w > org_h else long_edge
    logger.debug(f"Video dimensions: {org_w}x{org_h}")
    if add_scale_filter:
        logger.debug(f"\tscaling video to {w}x{h}")
    scale_filter = f",scale='{w}:{h}'" if add_scale_filter else ""

    frames = []

    for target in tqdm(timestamps_or_frames, desc="Extracting specific frames"):
        try:
            # Convert frame number to timestamp if needed
            if as_timestamps:
                seek_time = float(target)
                if seek_time > duration:
                    logger.warning(
                        f"Timestamp {seek_time}s exceeds video duration {duration}s, skipping"
                    )
                    frames.append(None)  # keep list index-aligned with input
                    continue
            else:
                # Convert frame number to timestamp
                seek_time = float(target) / org_fps
                if seek_time > duration:
                    logger.warning(f"Frame {target} exceeds video duration, skipping")
                    frames.append(None)  # keep list index-aligned with input
                    continue

            filter_chain = f"fps={org_fps}{scale_filter}"

            # Extract single frame at specific timestamp.
            # NOTE: -ss is passed as an input option (before -i), which is a fast
            # seek that snaps to the nearest keyframe rather than the exact frame.
            # In frame-number mode this means the returned frame may be off by a
            # few frames. Move ss to an output option (after .input()) for
            # exact-but-slower seeking if precise frames are ever required.
            # stderr left to inherit (not piped) to avoid a pipe-buffer
            # deadlock while we only read stdout — see extract_frames.
            process = (
                ffmpeg.input(input_path, ss=seek_time)
                .output(
                    "pipe:",
                    vf=filter_chain,
                    format="rawvideo",
                    pix_fmt="rgb24",
                    frames=1,
                )
                .run_async(pipe_stdout=True)
            )

            frame_size = (
                w * h * 3 if add_scale_filter else org_w * org_h * 3
            )  # 3 bytes per pixel (RGB)

            in_bytes = process.stdout.read(frame_size)
            if in_bytes and len(in_bytes) >= frame_size:
                frame = Image.frombytes(
                    "RGB", (w, h) if add_scale_filter else (org_w, org_h), in_bytes
                )
                frames.append(frame)
                if output_dir:
                    frame.save(os.path.join(output_dir, f"{vname}_{target}.jpg"))
            else:
                logger.warning(
                    f"Failed to extract frame at {'timestamp' if as_timestamps else 'frame'} {target}"
                )
                frames.append(
                    None
                )  # Add None for failed extractions to maintain list alignment

            process.stdout.close()
            process.wait()

        except Exception as e:
            logger.error(
                f"Error extracting frame at {'timestamp' if as_timestamps else 'frame'} {target}: {e}"
            )
            frames.append(None)  # Add None for failed extractions

    # Filter out None values if desired (or keep them for alignment)
    logger.info(
        f"Successfully extracted {len([f for f in frames if f is not None])} out of {len(timestamps_or_frames)} requested frames"
    )

    return frames


@app.command()
def extract_audio(
    video_path: str,
    output_dir: Optional[str] = None,
    overwrite: bool = False,
    lossless: bool = False,
):
    """Extract the audio track of a video file.

    By default the audio is re-encoded to mp3. Set ``lossless=True`` to copy the
    original audio stream without re-encoding into an m4a container.

    Args:
        video_path (str): Path to the input video file.
        output_dir (str): Directory to save the audio under. Defaults to the
            video's own directory when None. Output is written to
            ``{output_dir}/{vname}/{vname}.{ext}``.
        overwrite (bool): Overwrite the output if it already exists.
        lossless (bool): If True, copy the audio stream (acodec="copy") into an
            m4a file instead of re-encoding to mp3.

    Returns:
        str | None: Path to the extracted audio file, or None on failure / no audio.
    """
    # only return audio if its available
    vmeta = get_video_metadata(video_path, bverbose=False)
    if vmeta.get("audio_codec") == "none":
        logger.error(f"No audio found in {video_path}")
        return None

    # Create output directory if it doesn't exist
    output_dir = output_dir if output_dir else os.path.dirname(video_path)
    vname, vext = os.path.splitext(os.path.basename(video_path))
    output_dir = os.path.join(output_dir, vname)
    out_ext = "m4a" if lossless else "mp3"
    output_fname = os.path.join(output_dir, f"{vname}.{out_ext}")
    if os.path.isfile(output_fname):
        if overwrite:
            os.remove(output_fname)
            logger.warning(f"removed existing data: {output_fname}")
        else:
            logger.error(f"overwrite is false and data already exists: {output_fname}")
            return None
    os.makedirs(output_dir, exist_ok=True)

    # Construct the ffmpeg-python pipeline
    stream = ffmpeg.input(video_path)
    config_dict = {"map": "0:a", "acodec": "copy" if lossless else "mp3"}
    stream = ffmpeg.output(stream, output_fname, **config_dict)

    # Execute the ffmpeg command
    try:
        ffmpeg.run(stream, capture_stdout=True, capture_stderr=True)
        logger.success(f"audio extracted to {output_fname}")
        return output_fname
    except ffmpeg.Error as e:
        logger.error(f"Error executing FFmpeg command: {e.stderr.decode()}")
    return None


if __name__ == "__main__":
    app()
