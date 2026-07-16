# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A HuggingFace Space (`sdk: gradio`). `app.py` is a Gradio app that annotates images and
videos with bounding boxes / masks, built on the `pilbox.py` annotation module (the web-UI
counterpart of `annotate_cli.py`). `boxer.py` is a bounding-box utility module (format
conversion, IoU). `ffmpret.py` is a video-extraction module/CLI (frames / metadata / audio)
built on `ffmpeg-python`. `vidbox.py` ties them together to annotate video and backs the
app's **Annotate Video** tab.
No models, no GPU — deps are gradio + numpy + Pillow + loguru + typer + ffmpeg-python + tqdm.
The video features (`ffmpret.py`, `vidbox.py`, and hence the running app) also need a system
`ffmpeg` binary on PATH — the app now imports `vidbox` → `ffmpeg-python`, so it is no longer
pure-`pilbox`.

## Environment & commands

Dependencies are managed with `uv` and pinned in `uv.lock`. Requires Python >=3.10 (the
Space runs 3.12; see `.python-version` = 3.10 for local dev).

```bash
uv sync                 # install deps into .venv from uv.lock
uv run python app.py    # launch the Gradio app locally (also starts an MCP server + /docs)
uv add <pkg>            # add a dependency (updates pyproject.toml + uv.lock)
```

Tests live in `tests/` (pytest). Run them as `uv run python -m pytest tests/` — the
`python -m` form puts the repo root on `sys.path` so the top-level modules (`pilbox`,
`ffmpret`) import; plain `pytest tests/` fails to resolve them. There is no linter config
or build step in this repo.

## Deployment (important)

Pushing to `main` triggers `.github/workflows/deploy_to_hf_space.yaml`, which:
1. Generates `requirements.txt` via `uv export --no-hashes --format requirements-txt`
   **only if one doesn't already exist** (HF Spaces reads `requirements.txt`, not
   `pyproject.toml`). When generated, it is committed (`chore: update requirements.txt`)
   so the push actually carries it; a hand-maintained `requirements.txt` is left untouched.
2. Pushes the repo to the HuggingFace Space. The push is a **force-push only when the
   `FORCE_PUSH` GitHub secret is set** (otherwise a plain `git push`). The template assumes
   you are the sole contributor to the Space.

Before this workflow does anything real you must set the `HF_TOKEN` GitHub secret and
replace the `HF_USERNAME`/`SPACE_NAME` placeholders on the last line of the workflow. The
Space metadata (title, emoji, sdk_version) lives in the YAML front-matter of `README.md`.

**Example assets are not committed.** HF Spaces rejects raw binaries in git (they must go
through LFS/Xet), so image assets (`*.jpg`, `*.png`, …) plus the example data (`assets/*.json`)
are `.gitignore`d — kept on disk for local use but never pushed. Consequence: the demo
example (`assets/example_0_in.jpg` + `assets/example_0.json`) isn't on the Space, so
`app.py`'s example won't render there (`_load_example()` treats both as optional, so this
degrades gracefully). To ship an example on the Space, switch the image to Git LFS and
commit the JSON.

## app.py architecture

- A `gr.TabbedInterface` combining four `gr.Interface`s — no image/video logic lives in
  `app.py`; each callback delegates to `pilbox` / `vidbox`.
  - **Annotate** tab: `annotate_image(...)` → `pilbox.annotate` (`api_name="annotate"`).
    Inputs mirror the CLI options: image, pascal_voc JSON (paste-in `gr.Code`), `label_key`,
    `color_key`, `mask_key`, `mask_alpha`, `width`, `font_size`; output is the annotated image.
    When an object carries a base64-PNG mask under `mask_key` (default `"b64_mask"`), it is
    overlaid as a translucent mask **beneath** the box, colored to match that object's box
    (same `color_key` value → same color). Clearing the `Mask key` field disables masks.
  - **Annotate Video** tab: `annotate_video(...)` → `vidbox.annotate_video`
    (`api_name="annotate_video"`). Inputs are a `gr.Video`, a detections JSON **file upload**
    (`gr.File` — the mask JSON is large, so not a paste-in `gr.Code`), a `gr.Dropdown` of the
    four `boxer.BBOX_FORMATS` (default `coco_normalized`), a comma-separated `coord_keys`
    textbox (default `"x,y,w,h"`), then `label_key` / `color_key` / `mask_key` (default
    `"mask_b64"`, matching the SAM2 example data) / `mask_alpha` / `width` / `font_size`
    mirroring the image Annotate tab. Output is a silent annotated video at source
    resolution/fps. Each distinct `color_key` value keeps one stable color across all frames.
  - **Crop** tab: `crop_image(...)` → `pilbox.crop` (`api_name="crop"`). Inputs are an image
    plus manual `gr.Number` fields `x0/y0/x1/y1` (pascal_voc box); output is the cropped
    region.
  - **Crop Video** tab: `crop_video(...)` → `vidbox.crop_video` (`api_name="crop_video"`).
    Inputs are a `gr.Video`, a detections JSON `gr.File`, the `boxer.BBOX_FORMATS` dropdown +
    `coord_keys`, a `gr.Dropdown` mode (`vidbox.CROP_MODES` = `window`/`box_fit`, default
    `window`), a padding slider (1.0–2.0), and a gap-behavior dropdown (`vidbox.GAP_BEHAVIORS`
    = `jump`/`carry_forward`, default `jump`). Output is a cropped video that follows the
    subject; **masks in the JSON are ignored** — only boxes are used, one per frame.
  - **Mask** tab: `mask_image(...)` → `pilbox.apply_mask` (`api_name="mask"`). Inputs are an
    image, a base64-encoded PNG mask (paste-in `gr.Textbox`, same size as the image), and a
    `gr.ColorPicker` background color (default black `#000000`); output is the foreground cut
    out over that solid background. The color picker is a `string` in the API/MCP schema
    (a CSS color); `_rgb_from_css` converts `#rrggbb`/`rgba(...)` to an RGB tuple.
- Demo examples are loaded from `assets/` at import, each guarded so a missing asset doesn't
  crash startup: `_load_example()` (image + JSON), `_example_mask()` (first object's
  `b64_mask` for the Mask tab), `_load_video_example()` (the SAM2 video + `*-with_mask.json`
  for Annotate Video), and `_load_crop_video_example()` (the video + `*-VideoCrop-example.json`
  for Crop Video). All assets are git-ignored, so on the Space the examples are simply absent.
- Bad JSON, invalid crop boxes, undecodable/mismatched masks, unknown bbox formats, and
  conflicting per-frame crop boxes are surfaced as a `gr.Error` (`pilbox.crop` /
  `pilbox.apply_mask` / `vidbox.annotate_video` / `vidbox.crop_video` raise `ValueError`,
  re-raised as `gr.Error`).
- Launched under `if __name__ == "__main__"` with `mcp_server=True` and `docs_url="/docs"`
  (MCP server + FastAPI Swagger docs); each tab's endpoint (`annotate`, `annotate_video`,
  `crop`, `crop_video`, `mask`) is exposed as its own MCP tool. MCP tool name = the Python
  function name (`annotate_image` / `annotate_video` / `crop_image` / `crop_video` /
  `mask_image`), not `api_name`.

## Bounding-box conventions (boxer.py)

- Standard bbox dict is `{x0, y0, x1, y1}` (top-left / bottom-right absolute pixels).
  Note `boxer()` returns `{x1, x2, y1, y2}` instead — a different key convention.
- `to_pascal_voc(coords, fmt, im_w, im_h) -> {x0,y0,x1,y1}` converts any of the four
  supported formats in `BBOX_FORMATS` to pascal_voc absolute pixels: `pascal_voc`
  `[x0,y0,x1,y1]` abs, `albumentations` `[x0,y0,x1,y1]` norm, `coco` `[x0,y0,w,h]` abs,
  `coco_normalized` `[x0,y0,w,h]` norm. It reuses `get_bbox_dict` for the coco variants and is
  format-explicit — it does **not** use `bbox_convert` (whose `all(coord<=1)` auto-detect is
  fragile and would mis-convert an absolute box). This is what `vidbox` calls per detection.
- `bbox_convert` auto-detects relative vs absolute by testing whether all coords are `<= 1`.
- `bbox_intersects` uses the standard axis-aligned overlap test (fixed: the old corner-in-rect
  checks missed "cross" overlaps); `get_bbox_iou` builds on it.
- `bboxes_to_im_mask` and image crops use numpy `[y, x]` (row, col) indexing throughout.

## pilbox.py — lite bbox annotation module

`pilbox.py` is a standalone, importable annotation module depending only on **numpy +
Pillow + loguru** (no matplotlib). It uses the **pascal_voc** convention: each box is a
dict `{x0, y0, x1, y1}` of absolute top-left / bottom-right pixels (matching the
`boundingBox` in `assets/example_0.json`).

- `annotate(image, objects, *, label_key="object_id", color_key="object_id",
  bbox_key="boundingBox", mask_key="b64_mask", mask_alpha=0.5, width=3, font=None,
  color_map=None) -> Image.Image` is the main entry point. It returns an annotated **copy**
  (never mutates the input). Each distinct `color_key` value gets its own color; `label_key`'s
  value is drawn as a filled label tab above the box's top-left corner. Colors are pre-assigned
  in object order so an object's mask and box always share one color; masks are composited in a
  first pass (beneath) and boxes/labels in a second pass (on top). `mask_key=""` disables masks.
  Pass a shared (optionally pre-seeded) `color_map` dict to keep a `color_key` value's color
  stable across **multiple** `annotate` calls — this is how `vidbox` keeps a track id one color
  across every video frame; default `None` = a fresh per-call map.
- Colors come from `palette_color(index)` — golden-angle HSV hues, so the palette is
  **unbounded** (never runs out). `color_for(key, mapping)` assigns a stable palette index
  per unique key. (`PIL.ImageColor.colormap` is the named-color alternative.)
- `im_color_mask(im_rgb_array, mask_array, rgb_tup=..., alpha=0.5, get_pil_im=False)` blends
  a solid color into the image wherever the boolean mask is set; `_mask_from_b64` decodes a
  base64-PNG mask to a boolean array. Both back `annotate`'s mask overlay.
- `im_apply_mask(im_rgb_array, mask_array, *, bg_rgb_tup=None, bg_blur_radius=None,
  bg_greyscale=False, mask_gblur_radius=0, get_pil_im=False)` keeps the masked **foreground**
  and transforms the **background**: solid `bg_rgb_tup` color, `bg_blur_radius` Gaussian blur,
  or `bg_greyscale` — checked in that order; with none set the background is made transparent
  (RGBA). `mask_gblur_radius` softens the cutout edge. (The blur/greyscale paths need
  `ImageFilter`/`ImageOps`, both imported at the top of `pilbox.py`.)
- `apply_mask(image, b64_mask, bg_rgb_tup=(0,0,0)) -> Image.Image` is the PIL-in/PIL-out
  wrapper around `im_apply_mask` used by `app.py`'s Mask tab: decodes the base64 PNG mask
  (raising `ValueError` on bad base64 or a size mismatch) and returns a new RGB image with the
  background filled by `bg_rgb_tup`.
- `annotate_file(image_path, annotations_path, output_path, ...)` wraps load → annotate →
  save for the CLI.
- `crop(image, x0, y0, x1, y1) -> Image.Image` is a PIL-in/PIL-out wrapper around `im_crop`:
  it validates the pascal_voc box (raises `ValueError` if empty/inverted or out of bounds),
  converts to RGB, and returns a new image (never mutates the input). Used by `app.py`'s Crop
  tab.
- `letterbox(image, w, h, fill=(0,0,0)) -> Image.Image` scales `image` to fit inside `w`×`h`
  preserving aspect and pastes it centered on a solid `fill` canvas (black bars, no stretch).
  Backs `vidbox.crop_video`'s `box_fit` mode.
- `im_crop`, `im_center_crop`, `im_draw_point`, `load_pil_font` are retained helpers.

CLI: `annotate_cli.py` (typer + loguru, run separately so `import pilbox` stays lite):

```bash
uv run python annotate_cli.py assets/example_0_in.jpg assets/example_0.json -o out.jpg
# options: --label-key --color-key --mask-key --mask-alpha --width --font-size
```


## ffmpret.py — video I/O (frames / metadata / audio)

A typer CLI + importable module (typer + loguru) for pulling stills/metadata/audio out of a
video and encoding frames back into one, via `ffmpeg-python`. Depends on `ffmpeg-python` +
`tqdm` + a system `ffmpeg` binary. Imported by `vidbox.py` (and hence transitively by
`app.py`). All commands probe the video once via `get_video_metadata`.

- `get_video_metadata(video_path, bverbose=True) -> dict` probes width/height/duration/fps/
  codecs/bitrates/size. `duration` falls back to the container (`format`) duration when the
  video stream omits it (common for MKV/WebM) so downstream frame counts aren't silently 0.
  Raises (does not return `None`) when there's no video stream, so failures are legible to
  the callers that subscript the result.
- `extract_frames(input_path, fps=8, max_short_edge=1080, write_timestamp=True,
  write_frame_num=True, output_dir=None, out_vid_path=None, text_font_size=20,
  text_y_position="bottom") -> list[PIL.Image]` decodes frames by piping ffmpeg `rawvideo`
  to stdout. Requested `fps` is capped to source fps; **`fps=None` skips resampling and
  extracts every native frame** (so output index i == source frame i — what `vidbox` needs
  for frame-accurate annotation). Frames may be scaled down so the short edge ≤
  `max_short_edge`. An optional `drawtext` overlay stamps timestamp/frame-number
  (`text_y_position` ∈ {top, middle, bottom}); it uses ffmpeg's default font. Saves to
  `{output_dir}/{vname}_{i}.jpg` and/or re-encodes to `out_vid_path` when given.
  **stderr is intentionally left to inherit (not piped)** while only stdout is read — piping
  an undrained stderr deadlocks ffmpeg on longer clips. The read loop runs until the pipe is
  exhausted (`total_frames` is only the tqdm estimate).
- `frames_to_video(frames, out_path, fps, *, vcodec="libx264", pix_fmt="yuv420p") -> out_path`
  is the inverse: pipes a list of same-size PIL frames to ffmpeg's stdin as `rawvideo` and
  encodes a (silent) video (scaling to even dimensions for yuv420p). Used by `vidbox`. Only
  stdin is piped, so there's no stderr deadlock.
- `extract_specific_frames(input_path, timestamps_or_frames, max_short_edge=1080,
  as_timestamps=True, output_dir=None) -> list` grabs one frame per timestamp/frame number
  (one ffmpeg process each), keeping the list index-aligned with the input (failed grabs →
  `None`). `-ss` is an input option (fast keyframe seek); see the in-code note for exact-seek.
- `extract_audio(video_path, output_dir=None, overwrite=False, lossless=False) -> str|None`
  extracts the audio track to `{output_dir}/{vname}/{vname}.{ext}` (`output_dir` defaults to
  the video's own directory). Default re-encodes to **mp3**; `lossless=True` copies the stream
  (`acodec=copy`) into **m4a**. Returns `None` (logs an error) when the video has no audio or
  the output exists and `overwrite=False`.
- `parse_frame_name("my_clip_12.jpg") -> ("my_clip", 12)` splits on the **last** `_` so frame
  types containing underscores round-trip.

```bash
uv run python ffmpret.py get-video-metadata clip.mp4
uv run python ffmpret.py extract-frames clip.mp4 --fps 8 --output-dir frames/
uv run python ffmpret.py extract-audio clip.mp4 --output-dir audio/           # mp3
uv run python ffmpret.py extract-audio clip.mp4 --output-dir audio/ --lossless # m4a copy
```


## vidbox.py — video annotation + crop (ties ffmpret + boxer + pilbox)

The video counterparts of `pilbox.annotate` / `pilbox.crop`; back the app's **Annotate Video**
and **Crop Video** tabs and add no new deps (it imports the three existing modules). Keeps
`pilbox` lite by living in its own module.

- `annotate_video(video_path, detections, out_path, *, bbox_format="coco_normalized",
  coord_keys=("x","y","w","h"), frame_key="frame", label_key="track_id",
  color_key="track_id", mask_key="mask_b64", mask_alpha=0.5, width=3, font_size=20) ->
  out_path`. `detections` is a **flat** list of per-frame dicts (frame index under
  `frame_key`, box values under `coord_keys` read positionally per `bbox_format`, optional
  `label_key`/`color_key`/`mask_key`). It: probes metadata → extracts **every native frame**
  (`ffmpret.extract_frames(fps=None)`) → groups detections by frame → converts each box via
  `boxer.to_pascal_voc` → `pilbox.annotate`s each frame with a **shared pre-seeded
  `color_map`** (stable per-`color_key` color across frames) → re-encodes with
  `ffmpret.frames_to_video` at source fps. Masks (`mask_key`) are full-frame base64 PNGs, so
  annotation is at **native resolution** (no downscale) to keep them aligned. Detections whose
  frame index exceeds the decoded frame count are warned + skipped. Raises `ValueError` on an
  unknown `bbox_format` or zero decoded frames.
- `crop_video(video_path, detections, out_path, *, bbox_format="coco_normalized",
  coord_keys=("x","y","w","h"), frame_key="frame", mode="window", padding=1.0,
  gap_behavior="jump") -> out_path` crops a video to a subject that moves frame-to-frame.
  **Masks are ignored** (boxes only), and each frame must have **at most one** box: exact-
  duplicate rows dedupe, but two *different* boxes on a frame raise `ValueError` (validated
  before ffmpeg runs). Output size = per-axis max box (× `padding`, `_even`, clamped to the
  frame); constant across frames so it encodes. `mode` ∈ `CROP_MODES` (`window` = a fixed
  window cropped from the frame, re-centered on each box, keeps scene, pans; `box_fit` = crop
  to the box then `pilbox.letterbox` to fill, subject only). `gap_behavior` ∈ `GAP_BEHAVIORS`
  for frames with no box (`jump` = centered window / black frame; `carry_forward` = repeat the
  previous output). Reuses `ffmpret.extract_frames(fps=None)` + `frames_to_video`, `pilbox.crop`
  + `letterbox`; module helpers `_even` / `_pad_clamp_box` / `_crop_window` / `_gap_frame`.
- CLIs `annotate_video_file` / `crop_video_file` (comma-separated `--coord-keys`); a
  `@app.callback()` keeps the subcommand names (Typer otherwise collapses a lone command).

```bash
uv run python vidbox.py annotate-video-file in.mp4 dets.json out.mp4 --bbox-format coco_normalized
uv run python vidbox.py crop-video-file in.mp4 dets.json out.mp4 --mode window   # or --mode box_fit
```


## Guidelines

- Keep requirements to a minimum and lite
- Create tests in `tests/` and update CLAUDE.md for each new feature
- use Google-style docstring for new functions and add a doctest compatible unit test if possible
- keep code modular to ensure ease in future refactoring
- use `typer` for CLI and `loguru` for logging
- Prefer native gradio features over custom CSS
- Keep custom CSS minimal
