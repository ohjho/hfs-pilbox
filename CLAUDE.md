# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A HuggingFace Space (`sdk: gradio`). `app.py` is a Gradio app that draws pascal_voc
bounding boxes on an image (the web-UI counterpart of `annotate_cli.py`), built on the
`pilbox.py` annotation module. `boxer.py` is a standalone bounding-box utility module.
`ffmpret.py` is a standalone video-extraction CLI (frames / metadata / audio) built on
`ffmpeg-python`; it is not yet wired into the Gradio app.
No models, no GPU — deps are gradio + numpy + Pillow + loguru + typer, plus ffmpeg-python +
tqdm for `ffmpret.py` (which also needs a system `ffmpeg` binary on PATH).

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

- A `gr.TabbedInterface` combining three `gr.Interface`s — no image logic lives in `app.py`;
  each callback delegates to `pilbox`.
  - **Annotate** tab: `annotate_image(...)` → `pilbox.annotate` (`api_name="annotate"`).
    Inputs mirror the CLI options: image, pascal_voc JSON (paste-in `gr.Code`), `label_key`,
    `color_key`, `mask_key`, `mask_alpha`, `width`, `font_size`; output is the annotated image.
    When an object carries a base64-PNG mask under `mask_key` (default `"b64_mask"`), it is
    overlaid as a translucent mask **beneath** the box, colored to match that object's box
    (same `color_key` value → same color). Clearing the `Mask key` field disables masks.
  - **Crop** tab: `crop_image(...)` → `pilbox.crop` (`api_name="crop"`). Inputs are an image
    plus manual `gr.Number` fields `x0/y0/x1/y1` (pascal_voc box); output is the cropped
    region.
  - **Mask** tab: `mask_image(...)` → `pilbox.apply_mask` (`api_name="mask"`). Inputs are an
    image, a base64-encoded PNG mask (paste-in `gr.Textbox`, same size as the image), and a
    `gr.ColorPicker` background color (default black `#000000`); output is the foreground cut
    out over that solid background. The color picker is a `string` in the API/MCP schema
    (a CSS color); `_rgb_from_css` converts `#rrggbb`/`rgba(...)` to an RGB tuple.
- The demo example (image + JSON) is loaded from `assets/` at import via `_load_example()`,
  guarded so a missing asset doesn't crash startup; `_example_mask()` pulls the first object's
  `b64_mask` out of that JSON to preload the Mask tab (empty when assets are absent).
- Bad JSON (annotate), invalid crop boxes, and undecodable/mismatched masks are surfaced to the
  user as a `gr.Error` (`pilbox.crop` / `pilbox.apply_mask` raise `ValueError`, re-raised as
  `gr.Error`).
- Launched under `if __name__ == "__main__"` with `mcp_server=True` and `docs_url="/docs"`
  (MCP server + FastAPI Swagger docs); each tab's endpoint (`annotate`, `crop`, `mask`) is
  exposed as its own MCP tool. MCP tool name = the Python function name (`annotate_image` /
  `crop_image` / `mask_image`), not `api_name`.

## Bounding-box conventions (boxer.py)

- Standard bbox dict is `{x0, y0, x1, y1}` (top-left / bottom-right absolute pixels).
  Note `boxer()` returns `{x1, x2, y1, y2}` instead — a different key convention.
- `bbox_convert` auto-detects relative vs absolute by testing whether all coords are `<= 1`.
- `bboxes_to_im_mask` and image crops use numpy `[y, x]` (row, col) indexing throughout.

## pilbox.py — lite bbox annotation module

`pilbox.py` is a standalone, importable annotation module depending only on **numpy +
Pillow + loguru** (no matplotlib). It uses the **pascal_voc** convention: each box is a
dict `{x0, y0, x1, y1}` of absolute top-left / bottom-right pixels (matching the
`boundingBox` in `assets/example_0.json`).

- `annotate(image, objects, *, label_key="object_id", color_key="object_id",
  bbox_key="boundingBox", mask_key="b64_mask", mask_alpha=0.5, width=3, font=None) ->
  Image.Image` is the main entry point. It returns an annotated **copy** (never mutates the
  input). Each distinct `color_key` value gets its own color; `label_key`'s value is drawn as
  a filled label tab above the box's top-left corner. Colors are pre-assigned in object order
  so an object's mask and box always share one color; masks are composited in a first pass
  (beneath) and boxes/labels in a second pass (on top). `mask_key=""` disables masks.
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
- `im_crop`, `im_center_crop`, `im_draw_point`, `load_pil_font` are retained helpers.

CLI: `annotate_cli.py` (typer + loguru, run separately so `import pilbox` stays lite):

```bash
uv run python annotate_cli.py assets/example_0_in.jpg assets/example_0.json -o out.jpg
# options: --label-key --color-key --mask-key --mask-alpha --width --font-size
```


## ffmpret.py — video extraction CLI (standalone)

A standalone typer CLI (typer + loguru) for pulling stills/metadata/audio out of a video via
`ffmpeg-python`. It depends on `ffmpeg-python` + `tqdm` + a system `ffmpeg` binary, so it is
kept separate from the lite `pilbox`/app path and is **not** imported by `app.py` (yet — the
intended next step is an "annotate video" feature reusing `pilbox` across frames). All
commands probe the video once via `get_video_metadata`.

- `get_video_metadata(video_path, bverbose=True) -> dict` probes width/height/duration/fps/
  codecs/bitrates/size. `duration` falls back to the container (`format`) duration when the
  video stream omits it (common for MKV/WebM) so downstream frame counts aren't silently 0.
  Raises (does not return `None`) when there's no video stream, so failures are legible to
  the callers that subscript the result.
- `extract_frames(input_path, fps=8, max_short_edge=1080, write_timestamp=True,
  write_frame_num=True, output_dir=None, out_vid_path=None, text_font_size=20,
  text_y_position="bottom") -> list[PIL.Image]` decodes frames by piping ffmpeg `rawvideo`
  to stdout. Requested `fps` is capped to source fps; frames may be scaled down so the short
  edge ≤ `max_short_edge`. An optional `drawtext` overlay stamps timestamp/frame-number
  (`text_y_position` ∈ {top, middle, bottom}); it uses ffmpeg's default font. Saves to
  `{output_dir}/{vname}_{i}.jpg` and/or re-encodes to `out_vid_path` when given.
  **stderr is intentionally left to inherit (not piped)** while only stdout is read — piping
  an undrained stderr deadlocks ffmpeg on longer clips. The read loop runs until the pipe is
  exhausted (`total_frames = duration*fps` is only the tqdm estimate).
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


## Guidelines

- Keep requirements to a minimum and lite
- Create tests in `tests/` and update CLAUDE.md for each new feature
- use Google-style docstring for new functions and add a doctest compatible unit test if possible
- keep code modular to ensure ease in future refactoring
- use `typer` for CLI and `loguru` for logging
- Prefer native gradio features over custom CSS
- Keep custom CSS minimal
