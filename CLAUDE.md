# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A HuggingFace Space (`sdk: gradio`). `app.py` is a Gradio app that draws pascal_voc
bounding boxes on an image (the web-UI counterpart of `annotate_cli.py`), built on the
`pilbox.py` annotation module. `boxer.py` is a standalone bounding-box utility module.
No models, no GPU — deps are just gradio + numpy + Pillow + loguru + typer.

## Environment & commands

Dependencies are managed with `uv` and pinned in `uv.lock`. Requires Python >=3.10 (the
Space runs 3.12; see `.python-version` = 3.10 for local dev).

```bash
uv sync                 # install deps into .venv from uv.lock
uv run python app.py    # launch the Gradio app locally (also starts an MCP server + /docs)
uv add <pkg>            # add a dependency (updates pyproject.toml + uv.lock)
```

There is no test suite, linter config, or build step in this repo.

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

- A `gr.TabbedInterface` combining two `gr.Interface`s — no image logic lives in `app.py`;
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
- The demo example (image + JSON) is loaded from `assets/` at import via `_load_example()`,
  guarded so a missing asset doesn't crash startup.
- Bad JSON (annotate) and invalid crop boxes are surfaced to the user as a `gr.Error`
  (`pilbox.crop` raises `ValueError`, re-raised as `gr.Error`).
- Launched under `if __name__ == "__main__"` with `mcp_server=True` and `docs_url="/docs"`
  (MCP server + FastAPI Swagger docs); each tab's endpoint (`annotate`, `crop`) is exposed as
  its own MCP tool. MCP tool name = the Python function name (`annotate_image` / `crop_image`),
  not `api_name`.

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


## Guidelines

- Keep requirements to a minimum and lite
- Create tests in `tests/` and update CLAUDE.md for each new feature
- use Google-style docstring for new functions and add a doctest compatible unit test if possible
- keep code modular to ensure ease in future refactoring
- use `typer` for CLI and `loguru` for logging
- Prefer native gradio features over custom CSS
- Keep custom CSS minimal
