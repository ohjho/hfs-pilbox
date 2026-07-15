"""Typer CLI to annotate an image with bounding boxes from a JSON file.

Run with uv, e.g.::

    uv run python annotate_cli.py assets/example_0_in.jpg assets/example_0.json

The JSON must be a list of object dicts, each with a pascal_voc ``boundingBox``
(``{x0, y0, x1, y1}``); see ``assets/example_0.json``.
"""

from pathlib import Path
from typing import Optional

import typer
from loguru import logger

import pilbox

app = typer.Typer(add_completion=False, help=__doc__)


@app.command()
def main(
    image: Path = typer.Argument(
        ..., exists=True, dir_okay=False, readable=True, help="Source image path."
    ),
    annotations: Path = typer.Argument(
        ..., exists=True, dir_okay=False, readable=True, help="JSON annotations path."
    ),
    output: Optional[Path] = typer.Option(
        None, "--output", "-o", help="Output path (default: <image>_out.jpg)."
    ),
    label_key: str = typer.Option(
        "object_id", "--label-key", help="Object key drawn as each box's label."
    ),
    color_key: str = typer.Option(
        "object_id", "--color-key", help="Object key used to color-group boxes."
    ),
    mask_key: str = typer.Option(
        "b64_mask",
        "--mask-key",
        help="Object key holding a base64 PNG mask (empty disables masks).",
    ),
    mask_alpha: float = typer.Option(
        0.5, "--mask-alpha", help="Mask overlay opacity in [0, 1]."
    ),
    width: int = typer.Option(3, "--width", help="Box outline width in pixels."),
    font_size: int = typer.Option(20, "--font-size", help="Label font size."),
):
    """Draw bounding boxes from ANNOTATIONS onto IMAGE and save the result."""
    out_path = output or image.with_name(f"{image.stem}_out.jpg")
    logger.info(f"annotating {image} with {annotations} -> {out_path}")
    pilbox.annotate_file(
        str(image),
        str(annotations),
        str(out_path),
        label_key=label_key,
        color_key=color_key,
        mask_key=mask_key,
        mask_alpha=mask_alpha,
        width=width,
        font_size=font_size,
    )
    logger.info(f"saved {out_path}")


if __name__ == "__main__":
    app()
