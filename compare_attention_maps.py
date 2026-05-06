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
    # "1_SHOT_CROSS_ATTN_5_POINTS_TOP_K_LAST_LAYER",
    "1_SHOT_SELF_ATTN_BOTTOMK_5_POINTS_TEXT_ONLY_ALL_LAYERS",
    # "1_SHOT_SELF_ATTN_BOTTOMK_5_POINTS_TEXT_ONLY_LAST_LAYER",
    "1_SHOT_SELF_ATTN_BOTTOMK_5_POINTS_TEXT_ONLY_ALL_LAYERS_AGGR_MAX",
    # "1_SHOT_SELF_ATTN_BOTTOMK_5_POINTS_TEXT_ONLY_LAST_LAYER_AGGR_MAX",
    "1_SHOT_SELF_ATTN_BOTTOMK_5_POINTS_TEXT_SUPPORT_ALL_LAYERS",
    # "1_SHOT_SELF_ATTN_BOTTOMK_5_POINTS_TEXT_SUPPORT_LAST_LAYER",
    "1_SHOT_SELF_ATTN_BOTTOMK_5_POINTS_TEXT_SUPPORT_ALL_LAYERS_AGGR_MAX",
    # "1_SHOT_SELF_ATTN_BOTTOMK_5_POINTS_TEXT_SUPPORT_LAST_LAYER_AGGR_MAX",
    "1_SHOT_MATCHER_5_POINTS_FUSED_K_MEDOIDS_POINTS",
    "1_SHOT_5_POINTS_QUERY_AS_SUPPORT",
    # add more experiment folder names here
]

EXPERIMENT_NAMES = [
    "CROSS_ATTN_ALL_LAYERS",
    # "CROSS_ATTN_LAST_LAYER",
    "SELF_ATTN_TEXT_ONLY_ALL_LAYERS",
    # "SELF_ATTN_TEXT_ONLY_LAST_LAYER",
    "SELF_ATTN_TEXT_ONLY_ALL_LAYERS_AGGR_MAX",
    # "SELF_ATTN_TEXT_ONLY_LAST_LAYER_AGGR_MAX",
    "SELF_ATTN_TEXT_SUPPORT_ALL_LAYERS",
    # "SELF_ATTN_TEXT_SUPPORT_LAST_LAYER",
    "SELF_ATTN_TEXT_SUPPORT_ALL_LAYERS_AGGR_MAX",
    # "SELF_ATTN_TEXT_SUPPORT_LAST_LAYER_AGGR_MAX",
    "MATCHER_5_POINTS_FUSED_K_MEDOIDS_POINTS",
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

N_SAMPLES = 10      # number of frames to compare per benchmark; None = all frames
RANDOM_SEED = 42    # for reproducible sampling
# Fast mode: discover tags from the first experiment only, then do targeted lookups for the rest.
# Set to False to revert to full scan + intersection across all experiments (slower but safer).
SKIP_INTERSECTION_CHECK = True
OUTPUT_DIR = "/leonardo_work/IscrC_MARSv2/SAM3_FSVOS/attention_maps_comparison"  # one sub-folder per benchmark will be created here
CELL_SIZE_INCHES = 3.5    # size of each cell (square); reduce if figures are too wide/tall

# Per-layer map layout
N_LAYERS = 6
LAYER_TYPES_ORDER = ["self", "cross_total", "cross_text", "cross_points"]

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
_LAYER_PATTERN = re.compile(
    r"^frame_(.+?)_layer_(\d+)_(self|cross|cross_points|cross_text)\.png$"
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
            if m:
                frame_tag, map_type, _, sampled = m.group(1), m.group(2), m.group(3), m.group(4)
                if sampled:
                    map_type = map_type + "_sampled"
                key = str(rel / frame_tag)
                frames.setdefault(key, {})[map_type] = Path(attn_dir) / fname
            else:
                m2 = _LAYER_PATTERN.match(fname)
                if not m2:
                    continue
                frame_tag, layer_num, ltype = m2.group(1), m2.group(2), m2.group(3)
                if ltype == "cross":
                    ltype = "cross_total"
                key = str(rel / frame_tag)
                frames.setdefault(key, {})[f"layer_{layer_num}_{ltype}"] = Path(attn_dir) / fname

            # Resolve extra-column paths and matcher_points (once per key)
            # if "__extra_resolved" not in frames[key]:
            #     for col_label, rel_tmpl in EXTRA_COLUMNS:
            #         col_path = sample_abs / rel_tmpl.format(frame_tag=frame_tag)
            #         frames[key][col_label] = col_path if col_path.is_file() else None
            #     matcher_path = sample_abs / "bounding_box" / f"frame_{frame_tag}_matcher_points.png"
            #     frames[key]["MATCHER_POINTS"] = matcher_path if matcher_path.is_file() else None
            #     frames[key]["__extra_resolved"] = True
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


def _layer_label(col_key: str) -> str:
    """'layer_3_cross_total' -> 'cross_total L3'"""
    parts = col_key.split("_", 2)  # ['layer', '3', 'cross_total']
    return f"{parts[2]} L{parts[1]}"


def make_comparison(tag: str, per_exp_frames: list, exp_names: list, output_path: Path):
    extra_labels = [label for label, _ in EXTRA_COLUMNS]

    per_layer_cols = [
        f"layer_{n}_{t}"
        for t in LAYER_TYPES_ORDER
        for n in range(N_LAYERS)
    ]

    has_layers = any(
        any(k.startswith("layer_") for k in f) for f in per_exp_frames
    )
    # has_matcher = any(f.get("MATCHER_POINTS") is not None for f in per_exp_frames)

    all_col_keys = list(extra_labels)
    if has_layers:
        all_col_keys += per_layer_cols
    all_col_keys += list(ATTN_MAP_TYPES)
    # if has_matcher:
    #     all_col_keys += ["MATCHER_POINTS"]

    all_col_labels = list(extra_labels)
    if has_layers:
        all_col_labels += [_layer_label(c) for c in per_layer_cols]
    all_col_labels += [t.replace("_", " ") for t in ATTN_MAP_TYPES]
    # if has_matcher:
    #     all_col_labels += ["MATCHER_POINTS"]

    num_rows = len(per_exp_frames)
    num_cols = len(all_col_keys)
    fig, axes = plt.subplots(
        num_rows, num_cols,
        figsize=(CELL_SIZE_INCHES * num_cols, CELL_SIZE_INCHES * num_rows),
        dpi=100,
        squeeze=False,
    )

    for row_idx, (frames, exp_name) in tqdm(enumerate(zip(per_exp_frames, exp_names)), desc=f"map_type"):
        for col_idx, (col_key, col_label) in enumerate(zip(all_col_keys, all_col_labels)):
            ax = axes[row_idx][col_idx]
            ax.axis("off")

            _render_cell(ax, frames.get(col_key))
            if row_idx == 0:
                ax.set_title(col_label, fontsize=10, pad=6)

    # Fixed-inch left margin so cells stay square regardless of column count
    label_margin_in = 1.8
    left_frac = label_margin_in / (CELL_SIZE_INCHES * num_cols)
    plt.tight_layout(rect=[left_frac, 0, 1, 0.99])
    for row_idx, exp_name in enumerate(exp_names):
        ax = axes[row_idx][0]
        pos = ax.get_position()
        y_center = (pos.y0 + pos.y1) / 2
        fig.text(left_frac - 0.005, y_center, exp_name, va="center", ha="right",
                 rotation=0, fontsize=9, fontweight="bold")

    fig.suptitle(tag, fontsize=11, y=0.995)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _load_frames_for_tags(folder: str, tags: list) -> dict:
    """Targeted lookup: scan only the attention_maps dirs for the given tags."""
    frames: dict = {}
    folder_path = Path(folder)
    if not folder_path.is_dir():
        print(f"  WARNING: folder not found: {folder}")
        return {tag: {} for tag in tags}
    for key in tags:
        key_path = Path(key)
        frame_tag = key_path.name
        sample_rel = key_path.parent
        attn_dir = folder_path / sample_rel / "attention_maps"
        sample_abs = folder_path / sample_rel
        frames[key] = {}
        if not attn_dir.is_dir():
            continue
        for fname in os.listdir(attn_dir):
            m = _MAP_PATTERN.match(fname)
            if m and m.group(1) == frame_tag:
                map_type = m.group(2) + ("_sampled" if m.group(4) else "")
                frames[key][map_type] = attn_dir / fname
                continue
            m2 = _LAYER_PATTERN.match(fname)
            if m2 and m2.group(1) == frame_tag:
                ltype = m2.group(3) if m2.group(3) != "cross" else "cross_total"
                frames[key][f"layer_{m2.group(2)}_{ltype}"] = attn_dir / fname
        for col_label, rel_tmpl in EXTRA_COLUMNS:
            col_path = sample_abs / rel_tmpl.format(frame_tag=frame_tag)
            frames[key][col_label] = col_path if col_path.is_file() else None
        # matcher_path = sample_abs / "bounding_box" / f"frame_{frame_tag}_matcher_points.png"
        # frames[key]["MATCHER_POINTS"] = matcher_path if matcher_path.is_file() else None
    return frames


def run_benchmark(benchmark: str, exp_names: list):
    folders = [FOLDER_TEMPLATE.format(benchmark=benchmark, experiment=exp) for exp in EXPERIMENTS]

    rng = random.Random(RANDOM_SEED)

    if SKIP_INTERSECTION_CHECK:
        print(f"\n[{benchmark}] Scanning {exp_names[0]} to discover tags...")
        first_frames = discover_frames(folders[0])
        if not first_frames:
            print(f"  WARNING: no frames found in first experiment, skipping.")
            return
        print(f"  {len(first_frames)} frame tags found.")
        tags = sorted(first_frames.keys())
        if N_SAMPLES is not None and N_SAMPLES < len(tags):
            tags = rng.sample(tags, N_SAMPLES)
            print(f"  Sampling {N_SAMPLES} frames (seed={RANDOM_SEED}).")
        all_frames = [first_frames]
        for folder, name in zip(folders[1:], exp_names[1:]):
            frames = _load_frames_for_tags(folder, tags)
            print(f"  {name}: loaded {len(frames)} tags (targeted)")
            all_frames.append(frames)
    else:
        print(f"\n[{benchmark}] Scanning all experiment folders...")
        all_frames = []
        for folder, name in zip(folders, exp_names):
            frames = discover_frames(folder)
            print(f"  {name}: {len(frames)} frame tags found")
            all_frames.append(frames)
        common_tags = set(all_frames[0].keys())
        for frames in all_frames[1:]:
            common_tags &= set(frames.keys())
        if not common_tags:
            print(f"  WARNING: no common frame tags found for benchmark '{benchmark}', skipping.")
            return
        print(f"  {len(common_tags)} common frame tags found.")
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
