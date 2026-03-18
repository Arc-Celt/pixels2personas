"""
PC image example grids.

Input:
- Base directory containing per-PC folders with `top10/` and `bottom10/` image examples

Output:
- all_pcs_examples.pdf
- top7_pcs_examples.pdf

Example:

python scripts/plotting/pc_image_examples.py \
  --base-image-dir /path/to/avatar_images \
  --output-dir /path/to/out
"""

from __future__ import annotations
import argparse
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def load_and_resize_image(image_path: Path, target_size: Tuple[int, int]) -> np.ndarray:
    try:
        img = Image.open(image_path)
        img = img.convert("RGB")
        img = img.resize(target_size, Image.Resampling.LANCZOS)
        return np.array(img)
    except Exception:
        return np.ones((target_size[1], target_size[0], 3), dtype=np.uint8) * 200


def list_images(dir_path: Path) -> List[Path]:
    if not dir_path.exists():
        return []
    exts = {".jpg", ".jpeg", ".png"}
    return sorted(
        [p for p in dir_path.iterdir() if p.is_file() and p.suffix.lower() in exts]
    )


def create_combined_plot(
    base_image_dir: Path,
    pc_nums: Sequence[int],
    output_path: Path,
    dpi: int = 300,
    cell_size: Tuple[int, int] = (100, 120),
    figsize: Tuple[int, int] = (18, 40),
    font_size: int = 22,
    label_font_size: int = 26,
) -> None:
    n_pcs = len(pc_nums)
    fig, axes = plt.subplots(n_pcs * 2, 10, figsize=figsize, dpi=dpi)
    if n_pcs * 2 == 1:
        axes = np.array([axes])

    for idx, pc_num in enumerate(pc_nums):
        pc_dir = base_image_dir / f"pc_{pc_num}"
        top10_images = list_images(pc_dir / "top10")
        bottom10_images = list_images(pc_dir / "bottom10")

        top_row = idx * 2
        bottom_row = top_row + 1

        for i in range(10):
            ax = axes[top_row, i]
            if i < len(top10_images):
                ax.imshow(load_and_resize_image(top10_images[i], target_size=cell_size))
            else:
                ax.text(
                    0.5,
                    0.5,
                    "No Image",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                    fontsize=font_size,
                )
            ax.axis("off")

        for i in range(10):
            ax = axes[bottom_row, i]
            if i < len(bottom10_images):
                ax.imshow(
                    load_and_resize_image(bottom10_images[i], target_size=cell_size)
                )
            else:
                ax.text(
                    0.5,
                    0.5,
                    "No Image",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                    fontsize=font_size,
                )
            ax.axis("off")

    plt.tight_layout(w_pad=0.01, h_pad=0.5)

    for idx, pc_num in enumerate(pc_nums):
        top_row = idx * 2
        bottom_row = top_row + 1

        top_bbox = axes[top_row, 0].get_position()
        bottom_bbox = axes[bottom_row, 0].get_position()
        group_center_y = (
            top_bbox.y0 + top_bbox.height / 2 + bottom_bbox.y0 + bottom_bbox.height / 2
        ) / 2
        pc_label_x = top_bbox.x0 - 0.02
        top_bottom_label_x = top_bbox.x0 - 0.01

        fig.text(
            pc_label_x,
            group_center_y,
            f"PC{pc_num}",
            fontsize=label_font_size,
            fontweight="bold",
            rotation=90,
            ha="center",
            va="center",
        )
        fig.text(
            top_bottom_label_x,
            top_bbox.y0 + top_bbox.height / 2,
            "Top",
            rotation=90,
            fontsize=font_size,
            fontweight="bold",
            ha="center",
            va="center",
        )
        fig.text(
            top_bottom_label_x,
            bottom_bbox.y0 + bottom_bbox.height / 2,
            "Bottom",
            rotation=90,
            fontsize=font_size,
            fontweight="bold",
            ha="center",
            va="center",
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Create paper appendix PC image grids.")
    p.add_argument("--base-image-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--all-pcs-output-name", type=str, default="all_pcs_examples.pdf")
    p.add_argument("--top7-output-name", type=str, default="top7_pcs_examples.pdf")
    p.add_argument(
        "--num-pcs",
        type=int,
        default=10,
        help="Number of PCs for all_pcs_examples (default: 10).",
    )
    p.add_argument("--dpi", type=int, default=300)
    return p


def main(argv: Iterable[str] | None = None) -> None:
    args = build_arg_parser().parse_args(list(argv) if argv is not None else None)
    out_dir: Path = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    all_out = out_dir / args.all_pcs_output_name
    top7_out = out_dir / args.top7_output_name

    create_combined_plot(
        base_image_dir=args.base_image_dir,
        pc_nums=list(range(1, int(args.num_pcs) + 1)),
        output_path=all_out,
        figsize=(18, 40),
        dpi=int(args.dpi),
    )
    print(f"Saved: {all_out}")

    create_combined_plot(
        base_image_dir=args.base_image_dir,
        pc_nums=list(range(1, 8)),
        output_path=top7_out,
        figsize=(18, 28),
        dpi=int(args.dpi),
    )
    print(f"Saved: {top7_out}")


if __name__ == "__main__":
    main()
