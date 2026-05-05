"""
Compare attention maps across experiments for the same samples.

For each benchmark × sampled frame, produces one PNG: rows = experiments, columns = map types.
Edit the CONFIG section below, then run:  python src/compare_attention_maps.py
"""

import os
import re
import random
from pathlib import Path

# ---------------------------------------------------------------------------
# CONFIG — edit this section
# ---------------------------------------------------------------------------

# Template for experiment output folders.
# Use {benchmark} and {experiment} as placeholders.
FOLDER_TEMPLATE = "/leonardo_scratch/large/userexternal/mcavicch/SAM3_OUTPUT_DATA/{benchmark}/{experiment}"

BENCHMARKS = [
    "PASCAL-5i",
    "COCO-20i",
    "VSPW",
    "YOUTUBE_FSVOS",
    "LVIS-92i",
]

EXPERIMENTS = [
    "1_SHOT_CROSS_ATTN_5_POINTS_TOP_K_ALL_LAYERS",
    "1_SHOT_CROSS_ATTN_5_POINTS_TOP_K_LAST_LAYER",
    "1_SHOT_SELF_ON_SUPPORT_5_POINTS_TOP_K_ALL_LAYERS",
    "1_SHOT_SELF_ON_SUPPORT_5_POINTS_TOP_K_LAST_LAYER",
    "1_SHOT_5_POINTS_QUERY_AS_SUPPORT",
    # add more experiment folder names here
]

EXPERIMENT_NAMES = [
    "CROSS_ATTN_ALL_LAYERS",
    "CROSS_ATTN_LAST_LAYER",
    "SELF_ON_SUPPORT_ALL_LAYERS",
    "SELF_ON_SUPPORT_LAST_LAYER",
    "QUERY_AS_SUPPORT",
]  # list of labels matching EXPERIMENTS order, or None to use folder names

# Extra columns prepended to the attention-map columns.
# Resolved relative to each experiment's sample directory, shown in every row.
EXTRA_COLUMNS = [
    ("support overlay",  "frames/support_0_overlay.png"),
    ("GT overlay",       "ground_truth/frame_{frame_tag}_overlay.png"),
    ("prediction",       "output/frame_{frame_tag}_overlay.png"),
]

ATTN_MAP_TYPES = [
    "cross_total",
    "cross_text",
    "cross_points",
    "self",
    "cross_points_sampled",  # comment out to exclude sampled-points overlay
]

N_SAMPLES = 10      # number of frames to compare per benchmark; None = all common frames
RANDOM_SEED = 42    # for reproducible sampling
OUTPUT_DIR = "/leonardo_work/IscrC_MARSv2/SAM3_FSVOS/attention_maps_comparison"  # one sub-folder per benchmark will be created here

# ---------------------------------------------------------------------------

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(it, **kwargs):
        return it

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

# _sampled is an optional suffix after _{attn_layers}: frame_{tag}_cross_points_{last|all}_sampled.png
# stale files (dense_cross_attn, attn_points, attn_prior) are ignored by not matching
_MAP_PATTERN = re.compile(
    r"^frame_(.+?)_(cross_total|cross_text|cross_points|self)_(last|all)(_sampled)?\.png$"
)


def discover_frames(folder: str) -> dict:
    """Return {sample_key: {map_type: Path}} by walking {folder}/**/attention_maps/.

    Also resolves EXTRA_COLUMNS paths for each sample.
    """
    frames: dict = {}
    folder_path = Path(folder)
    if not folder_path.is_dir():
        print(f"  WARNING: folder not found: {folder}")
        return frames
    for attn_dir, _, fnames in os.walk(folder_path):
        if Path(attn_dir).name != "attention_maps":
            continue
        rel = Path(attn_dir).relative_to(folder_path).parent  # e.g. fold_2/sample_id
        sample_abs = folder_path / rel
        for fname in fnames:
            m = _MAP_PATTERN.match(fname)
            if not m:
                continue
            frame_tag, map_type, _, sampled = m.group(1), m.group(2), m.group(3), m.group(4)
            if sampled:
                map_type = map_type + "_sampled"
            key = str(rel / frame_tag)
            frames.setdefault(key, {})[map_type] = Path(attn_dir) / fname

            # Resolve extra-column paths (once per key)
            if "__extra_resolved" not in frames[key]:
                for col_label, rel_tmpl in EXTRA_COLUMNS:
                    col_path = sample_abs / rel_tmpl.format(frame_tag=frame_tag)
                    frames[key][col_label] = col_path if col_path.is_file() else None
                frames[key]["__extra_resolved"] = True
    return frames


def _render_cell(ax, filepath, placeholder="not found"):
    """Load and display an image in the given axes, or show a placeholder."""
    if filepath is not None and Path(filepath).is_file():
        img = np.array(Image.open(filepath).convert("RGB"))
        ax.imshow(img, aspect="auto")
    else:
        ax.set_facecolor("#cccccc")
        ax.text(0.5, 0.5, placeholder, ha="center", va="center",
                transform=ax.transAxes, fontsize=8, color="#555555")


def make_comparison(tag: str, per_exp_frames: list, exp_names: list, output_path: Path):
    extra_labels   = [label for label, _ in EXTRA_COLUMNS]
    all_col_labels = extra_labels + ATTN_MAP_TYPES

    num_rows = len(per_exp_frames)
    num_cols = len(all_col_labels)
    fig, axes = plt.subplots(
        num_rows, num_cols,
        figsize=(4 * num_cols, 3.5 * num_rows),
        dpi=100,
        squeeze=False,
    )

    for row_idx, (frames, exp_name) in tqdm(enumerate(zip(per_exp_frames, exp_names)), desc=f"map_type"):
        for col_idx, col_label in enumerate(all_col_labels):
            ax = axes[row_idx][col_idx]
            ax.axis("off")

            # All columns shown in every row
            _render_cell(ax, frames.get(col_label))
            if row_idx == 0:
                ax.set_title(col_label.replace("_", " "), fontsize=10, pad=6)

    # Row labels: placed in figure coordinates after layout is known
    plt.tight_layout(rect=[0.12, 0, 1, 0.99])
    for row_idx, exp_name in enumerate(exp_names):
        ax = axes[row_idx][0]
        pos = ax.get_position()
        y_center = (pos.y0 + pos.y1) / 2
        fig.text(0.01, y_center, exp_name, va="center", ha="left",
                 rotation=90, fontsize=11, fontweight="bold")

    fig.suptitle(tag, fontsize=11, y=0.995)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def run_benchmark(benchmark: str, exp_names: list):
    folders = [FOLDER_TEMPLATE.format(benchmark=benchmark, experiment=exp) for exp in EXPERIMENTS]

    print(f"\n[{benchmark}] Scanning experiment folders...")
    all_frames = []
    for folder, name in zip(folders, exp_names):
        frames = discover_frames(folder)
        print(f"  {name}: {len(frames)} frame tags found  ({folder})")
        all_frames.append(frames)

    common_tags = set(all_frames[0].keys())
    for frames in all_frames[1:]:
        common_tags &= set(frames.keys())

    if not common_tags:
        print(f"  WARNING: no common frame tags found for benchmark '{benchmark}', skipping.")
        return

    print(f"  {len(common_tags)} common frame tags found.")

    rng = random.Random(RANDOM_SEED)
    tags = sorted(common_tags)
    if N_SAMPLES is not None and N_SAMPLES < len(tags):
        tags = rng.sample(tags, N_SAMPLES)
        print(f"  Sampling {N_SAMPLES} frames (seed={RANDOM_SEED}).")

    output_dir = Path(OUTPUT_DIR) / benchmark
    output_dir.mkdir(parents=True, exist_ok=True)

    for tag in tqdm(tags, desc=f"{benchmark}"):
        per_exp = [frames[tag] for frames in all_frames]
        out_path = output_dir / f"{tag}_comparison.png"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        make_comparison(tag, per_exp, exp_names, out_path)

    print(f"  {len(tags)} files saved to: {output_dir.resolve()}")


def main():
    exp_names = EXPERIMENT_NAMES if EXPERIMENT_NAMES is not None else EXPERIMENTS
    if len(exp_names) != len(EXPERIMENTS):
        raise ValueError("EXPERIMENT_NAMES length must match EXPERIMENTS length")

    for benchmark in BENCHMARKS:
        run_benchmark(benchmark, exp_names)

    print("\nAll benchmarks done.")


if __name__ == "__main__":
    main()
