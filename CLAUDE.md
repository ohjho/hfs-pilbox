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

## Bounding-box conventions (boxer.py / pilbox.py)

- Standard bbox dict is `{x0, y0, x1, y1}` (top-left / bottom-right absolute pixels).
  Note `boxer()` returns `{x1, x2, y1, y2}` instead — a different key convention.
- `bbox_convert` auto-detects relative vs absolute by testing whether all coords are `<= 1`.
- `bboxes_to_im_mask` and image crops use numpy `[y, x]` (row, col) indexing throughout.
- `pilbox.py` references `os`, `logger`, `warnings`, and PIL (`Image`, `ImageDraw`,
  `ImageFont`) but has no import block — it is meant to be spliced into a host module, not
  run standalone.


## Guidelines

- Keep requirements to a minimum and lite
- Create tests in `tests/` and update CLAUDE.md for each new feature
- use Google-style docstring for new functions and add a doctest compatible unit test if possible
- keep code modular to ensure ease in future refactoring
- use `typer` for CLI and `loguru` for logging
- Prefer native gradio features over custom CSS
- Keep custom CSS minimal
