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
1. Runs `uv export --no-hashes --format requirements-txt > requirements.txt` (HF Spaces
   reads `requirements.txt`, not `pyproject.toml` — this is generated on the fly, never committed).
2. **Force-pushes** the repo to the HuggingFace Space. Because of the generated
   `requirements.txt`, the template assumes you are the sole contributor to the Space.

Before this workflow does anything real you must set the `HF_TOKEN` GitHub secret and
replace the `HF_USERNAME`/`SPACE_NAME` placeholders on the last line of the workflow. The
Space metadata (title, emoji, sdk_version) lives in the YAML front-matter of `README.md`.

## app.py architecture

- A single `gr.Interface` whose callback `annotate_image(...)` delegates to
  `pilbox.annotate` — no drawing logic lives in `app.py`. Inputs mirror the CLI options:
  image, pascal_voc JSON (paste-in `gr.Code`), `label_key`, `color_key`, `width`,
  `font_size`; output is the annotated image.
- The demo example (image + JSON) is loaded from `assets/` at import via `_load_example()`,
  guarded so a missing asset doesn't crash startup.
- Bad JSON is surfaced to the user as a `gr.Error`.
- Launched under `if __name__ == "__main__"` with `mcp_server=True` and `docs_url="/docs"`
  (MCP server + FastAPI Swagger docs); the annotate endpoint is exposed as `api_name="annotate"`.

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
  bbox_key="boundingBox", width=3, font=None) -> Image.Image` is the main entry point.
  It returns an annotated **copy** (never mutates the input). Each distinct `color_key`
  value gets its own color; `label_key`'s value is drawn as a filled label tab above the
  box's top-left corner.
- Colors come from `palette_color(index)` — golden-angle HSV hues, so the palette is
  **unbounded** (never runs out). `color_for(key, mapping)` assigns a stable palette index
  per unique key. (`PIL.ImageColor.colormap` is the named-color alternative.)
- `annotate_file(image_path, annotations_path, output_path, ...)` wraps load → annotate →
  save for the CLI.
- `im_crop`, `im_center_crop`, `im_draw_point`, `load_pil_font` are retained helpers.

CLI: `annotate_cli.py` (typer + loguru, run separately so `import pilbox` stays lite):

```bash
uv run python annotate_cli.py assets/example_0_in.jpg assets/example_0.json -o out.jpg
# options: --label-key --color-key --width --font-size
```


## Guidelines

- Keep requirements to a minimum and lite
- Create tests in `tests/` and update CLAUDE.md for each new feature
- use Google-style docstring for new functions and add a doctest compatible unit test if possible
- keep code modular to ensure ease in future refactoring
- use `typer` for CLI and `loguru` for logging
- Prefer native gradio features over custom CSS
- Keep custom CSS minimal
