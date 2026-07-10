# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A HuggingFace Space template (`sdk: gradio`) that runs on ZeroGPU. `app.py` is a Gradio
app comparing "small" vision-language models (Qwen2.5-VL, InternVL3) on video captioning.
`boxer.py` and `pilbox.py` are standalone bounding-box / image utility modules.

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

- **Model loading happens at import time.** `MODEL_ZOO` and `PROCESSORS` dicts are built
  eagerly at module top-level, loading every model in the zoo before the app starts. Adding
  a model means adding matching entries to both dicts.
- `load_model` branches on `model_family` (derived from the model name) in a `match`
  statement; only `InternVL3` is wired up, others raise `ValueError`. `video_inference`
  has a second `match` for building inputs — both must be extended for a new family.
- `@spaces.GPU(duration=120)` on `video_inference` is the ZeroGPU decorator; GPU is only
  allocated during that call. `DEVICE = "auto"` and `DTYPE` (bf16/fp16) are resolved at import.
- Flash Attention is pip-installed at runtime via a `subprocess.run` at the top of the file
  (ZeroGPU quirk), but the zoo currently loads models with `use_flash_attention=False`.

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
