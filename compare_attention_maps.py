"""
Compare attention maps across experiments for the same samples.

For each benchmark × sampled frame, produces one PNG: rows = experiments, columns = map types.
Edit the CONFIG section below, then run:  python src/compare_attention_maps.py

Sampling strategy
-----------------
The script enumerates candidate sample directories from the FIRST experiment only
(cheap directory listing, no deep walk).  For each randomly drawn candidate it
verifies that the corresponding attention_maps directory exists in EVERY other
experiment before accepting it.  Candidates that are missing from one or more
experiments are silently skipped and a fresh one is drawn.  This continues until
the requested N_SAMPLES have been collected (or the pool is exhausted).

ZIP export
----------
After plots are produced, the script creates one ZIP per benchmark at:
  {OUTPUT_DIR}/{benchmark}_samples.zip

The archive mirrors the original folder structure:
  {benchmark}/{experiment_name}/{relative_sample_path}/...

where {relative_sample_path} is the path relative to the experiment root
(e.g. fold_1/val2014/COCO_val2014_000000001234.jpg_40_641).
The ENTIRE contents of each sample directory are included.
"""

import os
import re
import random
import zipfile
import shutil
from pathlib import Path

# ---------------------------------------------------------------------------
# CONFIG — edit this section
# ---------------------------------------------------------------------------

# Template for experiment output folders.
# Use {benchmark} and {experiment} as placeholders.
FOLDER_TEMPLATE = "/megaverse/storage/samele/FSS-SAM3/output/{benchmark}/{experiment}"

BENCHMARKS = [
    # "PASCAL-5i",
    "COCO-20i",
    # "VSPW",
    # "YOUTUBE_FSVOS",
    # "LVIS-92i",
]

EXPERIMENTS = [
    "QUERY_AS_SUPPORT_ALL_LAYERS_FOR_MAPS",
    "EXP7_TEXT_ONLY_ALL_LAYERS_FOR_MAPS",
    # "1_SHOT_SELF_ATTN_BOTTOMK_5_POINTS_TEXT_ONLY_ALL_LAYERS",
    # "1_SHOT_SELF_ATTN_BOTTOMK_5_POINTS_TEXT_BOX_ALL_LAYERS_NO_POS_BIAS",
    # "EXP7_BOX_ALL_SAMP_QIMG_SPROMPT_NO_POS_BIAS",
    # "EXP5_ATTN_PRIOR_TOPK_ALL_LAYERS_NO_POS_BIAS",
    # "EXP5_SAMPL_TEXT_TOP_K_ALL_LAYERS_AGGR_MAX",
    # "EXP5_SAMPL_SUPPORT_TOP_K_ALL_LAYERS_NO_POS_BIAS_AGGR_MAX",
    # "1_SHOT_CROSS_ATTN_5_POINTS_TOP_K_ALL_LAYERS",
    # "1_SHOT_CROSS_ATTN_5_POINTS_TOP_K_LAST_LAYER",
    # "1_SHOT_SELF_ATTN_BOTTOMK_5_POINTS_TEXT_ONLY_ALL_LAYERS",
    # "1_SHOT_SELF_ATTN_BOTTOMK_5_POINTS_TEXT_ONLY_LAST_LAYER",
    # "1_SHOT_SELF_ATTN_BOTTOMK_5_POINTS_TEXT_SUPPORT_ALL_LAYERS",
    # "1_SHOT_SELF_ATTN_BOTTOMK_5_POINTS_TEXT_SUPPORT_LAST_LAYER",
    # "1_SHOT_SELF_ATTN_TOPK_5_POINTS_TEXT_ONLY_ALL_LAYERS_AGGR_MAX",
    # "1_SHOT_SELF_ATTN_TOPK_5_POINTS_TEXT_ONLY_LAST_LAYER_AGGR_MAX",
    # "1_SHOT_SELF_ATTN_TOPK_5_POINTS_TEXT_SUPPORT_ALL_LAYERS_AGGR_MAX",
    # "1_SHOT_SELF_ATTN_TOPK_5_POINTS_TEXT_SUPPORT_LAST_LAYER_AGGR_MAX",
    # "1_SHOT_MATCHER_5_POINTS_FUSED_K_MEDOIDS_POINTS",
    # "1_SHOT_5_POINTS_QUERY_AS_SUPPORT",
    # add more experiment folder names here
]

EXPERIMENT_NAMES = [
    "QUERY AS SUPPORT",
    "ZERO SHOT",
    # "SELF BOTTOM-K TEXT ALL_L",
    # "SELF BOTTOM-K TEXT+BOX NO_POS_BIAS ALL_L",
    # "SELF BOTTOM-K TEXT+BOX ALL_L TEXT_INJECTION SAMPLING QUERY+SUPPORT",
    # "CROSS TOP-K TEXT+POINTS NO_POS_BIAS ALL_L",
    # "CROSS TOP-K TEXT ALL_L AGGR_MAX",
    # "CROSS TOP-K POINTS NO_POS_BIAS ALL_L AGGR_MAX",
    # "CROSS-ATTN TOP-K TEXT+SUPPORT ALL LAYERS",
    # "CROSS-ATTN TOP-K TEXT+SUPPORT LAST LAYERS",
    # "SELF-ATTN BOTTOM-K TEXT ONLY ALL LAYERS",
    # "SELF-ATTN BOTTOM-K TEXT ONLY LAST LAYERS",
    # "SELF-ATTN BOTTOM-K TEXT+SUPPORT ALL LAYERS",
    # "SELF-ATTN BOTTOM-K TEXT+SUPPORT LAST LAYERS",
    # "SELF-ATTN TOP-K TEXT ONLY ALL LAYERS AGGR MAX",
    # "SELF-ATTN TOP-K TEXT ONLY LAST LAYERS AGGR MAX",
    # "SELF-ATTN TOP-K TEXT+SUPPORT ALL LAYERS AGGR MAX",
    # "SELF-ATTN TOP-K TEXT+SUPPORT LAST LAYERS AGGR MAX",
    # "MATCHER FUSED K-MEDOIDS",
    # "QUERY AS SUPPORT",
]  # list of labels matching EXPERIMENTS order, or None to use folder names

# Subfolder inside 'attention_maps' to read from for each experiment ("inference", "sampling", or "" for root).
# Must match EXPERIMENTS length.
EXPERIMENT_SUBFOLDERS = [
    # "",  # Example: read from attention_maps/sampling
    # "",          # Read from root of attention_maps (old behavior)
    # "",
    # "",
    # "sampling",
    # "sampling",
    "inference_maps",
    "sampling_maps",
    # add corresponding subfolder here for more experiments
]

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
CACHE_SAMPLED_TAGS = False   # if True, saves chosen frame tags to a .txt file and reloads them on next run

OUTPUT_DIR = "/megaverse/storage/samele/FSS-SAM3/attention_maps_comparison_2"  # one sub-folder per benchmark will be created here
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


# ---------------------------------------------------------------------------
# Sampling helpers
# ---------------------------------------------------------------------------

def _list_candidate_tags(folder: str, map_source: str = "") -> list:
    """
    Enumerate all *sample-level* directories under *folder* that actually
    contain an attention_maps directory (optionally with map_source subfolder).

    Returns a list of tag strings in the SAME format as ``discover_frames``:
        ``<rel_path_to_sample_dir>/<frame_tag_from_filename>``
    where ``frame_tag_from_filename`` is extracted from an actual file inside
    attention_maps/ (e.g. ``0`` from ``frame_0_cross_total_all.png``), NOT
    the directory name (which may differ).

    This is a fast, shallow enumeration — it peeks at one file per sample.
    """
    folder_path = Path(folder)
    print(f"  Scanning for candidate sample dirs in {folder_path}…")
    if not folder_path.is_dir():
        return []

    tags = []

    def _peek_frame_tag(attn_dir: Path) -> str | None:
        """Return the first frame_tag found by scanning filenames in attn_dir."""
        try:
            for fname in os.listdir(attn_dir):
                m = _MAP_PATTERN.match(fname) or _LAYER_PATTERN.match(fname)
                if m:
                    return m.group(1)
        except OSError:
            pass
        return None

    def _process_sample_dir(sample_dir: Path):
        """Check attention_maps exists, peek frame_tag, append tag."""
        print("      Processing", sample_dir)
        attn_dir = sample_dir / "attention_maps"
        if map_source:
            attn_dir = attn_dir / map_source
        print("        Checking", attn_dir)
        if not attn_dir.is_dir():
            return
        frame_tag = _peek_frame_tag(attn_dir)
        if frame_tag is None:
            return  # empty or unrecognised dir
        rel = sample_dir.relative_to(folder_path)  # e.g. fold_1/val2014/COCO_...
        tags.append(str(rel / frame_tag))

    # Walk up to 3 levels: fold_X / [optional_subdir] / sample_id
    for depth1 in folder_path.iterdir():          # e.g. fold_1
        print("  Scanning", depth1)
        if not depth1.is_dir():
            continue
        for depth2 in depth1.iterdir():           # e.g. val2014  (or sample directly)
            print("    Scanning", depth2)
            if not depth2.is_dir():
                continue
            # Check if depth2 itself is a sample (contains attention_maps)
            if (depth2 / "attention_maps").is_dir():
                _process_sample_dir(depth2)
                continue
            # Otherwise go one level deeper
            for depth3 in depth2.iterdir():       # e.g. sample_id
                print("      Scanning", depth3)
                if not depth3.is_dir():
                    continue
                _process_sample_dir(depth3)

    return tags


def _tag_exists_in_experiment(tag: str, folder: str, map_source: str = "") -> bool:
    """
    Return True if the attention_maps directory for *tag* exists in *folder*.

    tag format (same as discover_frames): ``<rel_to_sample_dir>/<frame_tag>``
    e.g. ``fold_1/val2014/COCO_val2014_000000000164.jpg_40_641/0``

    The sample directory is therefore ``folder / tag_parent``, i.e. everything
    except the last component (the frame_tag).
    """
    folder_path = Path(folder)
    key_path = Path(tag)
    sample_dir = folder_path / key_path.parent  # strip frame_tag → sample dir
    attn_dir = sample_dir / "attention_maps"
    if map_source:
        attn_dir = attn_dir / map_source
    return attn_dir.is_dir()


def _sample_tags(
    folder: str,
    exp_source: str,
    n: int,
    rng: random.Random,
) -> list:
    """
    Randomly draw *n* tags from the first experiment only — no cross-experiment
    checks.  Fast: just a shallow directory listing + shuffle.

    Returns the accepted tag list (may be shorter than *n* if the pool is
    smaller than requested).
    """
    print(f"  Building candidate pool from first experiment (lightweight scan)…")
    candidates = _list_candidate_tags(folder, exp_source)
    if not candidates:
        print(f"  WARNING: no candidate sample dirs found in first experiment.")
        return []

    print(f"  {len(candidates)} candidate samples found.")

    if n is None or n >= len(candidates):
        return list(candidates)

    return rng.sample(candidates, n)


# ---------------------------------------------------------------------------
# Frame / map loading (unchanged from original)
# ---------------------------------------------------------------------------

def discover_frames(folder: str, map_source: str = "") -> dict:
    """Return {sample_key: {map_type: Path}} by walking {folder}/**/attention_maps/[map_source].

    Also resolves EXTRA_COLUMNS paths for each sample.
    """
    frames: dict = {}
    folder_path = Path(folder)
    if not folder_path.is_dir():
        print(f"  WARNING: folder not found: {folder}")
        return frames
    for attn_dir, _, fnames in os.walk(folder_path):
        p_attn_dir = Path(attn_dir)
        if map_source:
            if p_attn_dir.name != map_source or p_attn_dir.parent.name != "attention_maps":
                continue
            rel = p_attn_dir.relative_to(folder_path).parent.parent  # e.g. fold_2/sample_id
        else:
            if p_attn_dir.name != "attention_maps":
                continue
            rel = p_attn_dir.relative_to(folder_path).parent  # e.g. fold_2/sample_id
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

            # Resolve extra-column paths once per (key, frame_tag) pair
            if "__extra_resolved" not in frames[key]:
                for col_label, rel_tmpl in EXTRA_COLUMNS:
                    col_path = sample_abs / rel_tmpl.format(frame_tag=frame_tag)
                    frames[key][col_label] = col_path if col_path.is_file() else None
                frames[key]["__extra_resolved"] = True
    return frames


def _render_cell(ax, filepath):
    """Load and display an image in the given axes, or show a neutral grey placeholder."""
    if filepath is not None and Path(filepath).is_file():
        img = np.array(Image.open(filepath).convert("RGB"))
        ax.imshow(img, aspect="auto")
    else:
        ax.set_facecolor("#dddddd")


def _layer_label(col_key: str) -> str:
    """'layer_3_cross_total' -> 'cross_total L3'"""
    parts = col_key.split("_", 2)  # ['layer', '3', 'cross_total']
    return f"{parts[2]} L{parts[1]}"


def _load_frames_for_tags(folder: str, tags: list, map_source: str = "") -> dict:
    """Targeted lookup: scan only the attention_maps dirs for the given tags.

    tag format: ``<rel_path_to_sample_dir>/<frame_tag>``
    e.g. ``fold_1/val2014/COCO_val2014_000000000164.jpg_40_641/0``

    - ``key_path.name``   → actual frame_tag used in filenames (e.g. ``"0"``)
    - ``key_path.parent`` → path to sample dir relative to experiment folder
    """
    frames: dict = {}
    folder_path = Path(folder)
    if not folder_path.is_dir():
        print(f"  WARNING: folder not found: {folder}")
        return {tag: {} for tag in tags}
    for key in tags:
        key_path = Path(key)
        frame_tag = key_path.name           # e.g. "0"
        sample_dir_rel = key_path.parent    # e.g. fold_1/val2014/COCO_val2014...
        sample_abs = folder_path / sample_dir_rel
        attn_dir = sample_abs / "attention_maps"
        if map_source:
            attn_dir = attn_dir / map_source
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
    return frames


# ---------------------------------------------------------------------------
# Figure generation (unchanged from original)
# ---------------------------------------------------------------------------

def make_comparison(tag: str, per_exp_frames: list, exp_names: list, output_path: Path):
    """Always saves a flat comparison (aggregated maps).
    Additionally saves a layered comparison (per-layer maps) when layer data is present."""
    extra_labels = [label for label, _ in EXTRA_COLUMNS]
    has_layers = any(any(k.startswith("layer_") for k in f) for f in per_exp_frames)

    # Always produce the flat (aggregated) figure
    _make_comparison_flat(tag, per_exp_frames, exp_names, extra_labels, output_path)

    # Produce the per-layer figure alongside (different filename),
    # keeping only experiments that actually have layer data.
    if has_layers:
        layer_mask   = [any(k.startswith("layer_") for k in f) for f in per_exp_frames]
        layer_frames = [f for f, m in zip(per_exp_frames, layer_mask) if m]
        layer_names  = [n for n, m in zip(exp_names,      layer_mask) if m]
        if layer_frames:
            layers_path = output_path.with_name(output_path.stem + "_layers" + output_path.suffix)
            _make_comparison_layered(tag, layer_frames, layer_names, layers_path)


def _make_comparison_flat(tag, per_exp_frames, exp_names, extra_labels, output_path):
    """One row per experiment: overlay images + aggregated attention maps."""
    all_col_keys   = list(extra_labels) + list(ATTN_MAP_TYPES)
    all_col_labels = list(extra_labels) + [t.replace("_", " ") for t in ATTN_MAP_TYPES]

    num_rows = len(per_exp_frames)
    num_cols = len(all_col_keys)
    label_margin_in = 1.8
    figwidth = label_margin_in + CELL_SIZE_INCHES * num_cols
    fig, axes = plt.subplots(
        num_rows, num_cols,
        figsize=(figwidth, CELL_SIZE_INCHES * num_rows),
        dpi=100, squeeze=False,
    )

    for row_idx, (frames, _) in enumerate(zip(per_exp_frames, exp_names)):
        for col_idx, (col_key, col_label) in enumerate(zip(all_col_keys, all_col_labels)):
            ax = axes[row_idx][col_idx]
            ax.axis("off")
            _render_cell(ax, frames.get(col_key))
            if row_idx == 0:
                ax.set_title(col_label, fontsize=10, pad=6)

    left_frac = label_margin_in / figwidth
    plt.tight_layout(rect=(left_frac, 0, 1, 0.99))
    for row_idx, exp_name in enumerate(exp_names):
        pos = axes[row_idx][0].get_position()
        y_center = (pos.y0 + pos.y1) / 2
        fig.text(left_frac - 0.005, y_center, exp_name,
                 va="center", ha="right", fontsize=9, fontweight="bold")

    fig.suptitle(tag, fontsize=11, y=0.995)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _make_comparison_layered(tag, per_exp_frames, exp_names, output_path):
    """Per-layer-only figure (Option-A layout).

    Each experiment block = one sub-row per attention type in LAYER_TYPES_ORDER.
    Columns = L0 … L{N_LAYERS-1}.  No summary row (see the flat figure for that).

    Left margin:
      • Outer band (rotated 90°): experiment name, large, centred over the block
      • Inner band (horizontal):  attention-type label on each sub-row
    Thick separator line + blank gap row between experiment blocks.
    """
    n_subrows  = len(LAYER_TYPES_ORDER)   # one sub-row per type (e.g. 4)
    num_exp    = len(per_exp_frames)
    # Insert one blank "gap" row between experiment blocks for breathing room
    GAP_ROWS   = 1
    rows_per_block = n_subrows + GAP_ROWS   # last block has no trailing gap
    num_rows   = num_exp * n_subrows + (num_exp - 1) * GAP_ROWS
    num_cols   = N_LAYERS

    MARGIN_EXP_IN  = 1.8   # wider band → bigger font for exp name
    MARGIN_TYPE_IN = 1.3
    label_margin_in = MARGIN_EXP_IN + MARGIN_TYPE_IN

    # Height ratios: data rows = 1, gap rows = 0.15
    height_ratios = []
    for exp_idx in range(num_exp):
        height_ratios.extend([1.0] * n_subrows)
        if exp_idx < num_exp - 1:
            height_ratios.extend([0.15] * GAP_ROWS)

    figheight = CELL_SIZE_INCHES * sum(height_ratios)
    figwidth  = label_margin_in + CELL_SIZE_INCHES * num_cols
    fig, axes = plt.subplots(
        num_rows, num_cols,
        figsize=(figwidth, figheight),
        dpi=100, squeeze=False,
        gridspec_kw={"height_ratios": height_ratios},
    )

    # Hide gap rows
    for exp_idx in range(num_exp - 1):
        gap_row = exp_idx * rows_per_block + n_subrows
        for col in range(num_cols):
            axes[gap_row][col].axis("off")
            axes[gap_row][col].set_facecolor("white")

    # ── Fill layer cells ──────────────────────────────────────────────────────
    for exp_idx, (frames, _) in enumerate(zip(per_exp_frames, exp_names)):
        base = exp_idx * rows_per_block
        for ti, ltype in enumerate(LAYER_TYPES_ORDER):
            row = base + ti
            for col in range(num_cols):
                ax = axes[row][col]
                ax.axis("off")
                _render_cell(ax, frames.get(f"layer_{col}_{ltype}"))

    # ── Layout ────────────────────────────────────────────────────────────────
    # Use subplots_adjust for precise control (avoids tight_layout+gridspec warning).
    left_frac = label_margin_in / figwidth
    fig.subplots_adjust(left=left_frac, right=0.99, top=0.975, bottom=0.015, hspace=0.08)

    # After layout, query actual axes positions for pixel-accurate label placement.
    axes_x0 = axes[0][0].get_position().x0   # true left edge of the axes area
    x_type  = axes_x0 - 0.008                # type label: just left of axes
    x_exp   = axes_x0 * 0.38                 # exp name:  in the outer margin band

    # ── Column headers: L0 … L{N_LAYERS-1} above very first data row ─────────
    for col in range(num_cols):
        pos = axes[0][col].get_position()
        fig.text((pos.x0 + pos.x1) / 2, pos.y1 + 0.004,
                 f"L{col}", ha="center", va="bottom", fontsize=11, fontweight="bold")

    # ── Left-margin labels ────────────────────────────────────────────────────
    sub_row_labels = [lt.replace("_", " ") for lt in LAYER_TYPES_ORDER]

    for exp_idx, exp_name in enumerate(exp_names):
        base = exp_idx * rows_per_block

        # Experiment name: large, rotated, centred over all sub-rows of this block
        pos_top = axes[base][0].get_position()
        pos_bot = axes[base + n_subrows - 1][0].get_position()
        y_center = (pos_top.y1 + pos_bot.y0) / 2
        fig.text(x_exp, y_center, exp_name,
                 va="center", ha="center", rotation=90,
                 fontsize=11, fontweight="bold")

        # Sub-row type labels: placed using actual axes x0, so they never overlap
        for sr_idx, sr_label in enumerate(sub_row_labels):
            pos = axes[base + sr_idx][0].get_position()
            y_c = (pos.y0 + pos.y1) / 2
            fig.text(x_type, y_c, sr_label,
                     va="center", ha="right", fontsize=9, color="#222222")

    # ── Thick separator lines in the gap between experiment blocks ────────────
    for exp_idx in range(num_exp - 1):
        gap_row = exp_idx * rows_per_block + n_subrows
        pos_gap = axes[gap_row][0].get_position()
        y_mid = (pos_gap.y0 + pos_gap.y1) / 2
        line = plt.Line2D(
            [left_frac * 0.05, 1.0], [y_mid, y_mid],
            transform=fig.transFigure,
            color="#333333", linewidth=2.5, linestyle="-", clip_on=False,
        )
        fig.add_artist(line)

    fig.suptitle(tag, fontsize=11, y=0.999)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# ZIP export
# ---------------------------------------------------------------------------

def _create_samples_zip(
    benchmark: str,
    tags: list,
    folders: list,
    experiments: list,
    exp_names: list,
):
    """
    Create a ZIP file at {OUTPUT_DIR}/{benchmark}_samples_comparison.zip.

    Archive structure:
      {benchmark}/{experiment_name}/{tag_relative_path}/...

    where {tag_relative_path} is the tag itself (e.g. fold_1/val2014/sample_id),
    and the entire sample directory is included.
    """
    zip_path = Path(OUTPUT_DIR) / f"{benchmark}_samples.zip"
    print(f"\n  Building ZIP: {zip_path}")

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, allowZip64=True) as zf:
        for folder, exp_name in zip(folders, exp_names):
            folder_path = Path(folder)
            if not folder_path.is_dir():
                print(f"    WARN: experiment folder missing, skipping: {folder}")
                continue

            for tag in tags:
                # tag = sample_dir_rel / frame_tag (e.g. fold_1/val2014/COCO_.../0)
                # The actual sample directory is the parent of the frame_tag component.
                tag_path = Path(tag)
                sample_dir_rel = tag_path.parent  # e.g. fold_1/val2014/COCO_val2014_...
                sample_dir = folder_path / sample_dir_rel
                if not sample_dir.is_dir():
                    print(f"    WARN: sample dir missing for tag '{tag}' in '{exp_name}', skipping.")
                    continue

                # Archive prefix: benchmark / exp_name / sample_dir_rel
                # e.g. COCO-20i/EXP_NAME/fold_1/val2014/COCO_val2014_...
                archive_prefix = str(Path(benchmark) / exp_name / sample_dir_rel)

                # Walk the entire sample directory
                for root, dirs, files in os.walk(sample_dir):
                    root_path = Path(root)
                    rel_root = root_path.relative_to(sample_dir)
                    for fname in files:
                        src_file = root_path / fname
                        arcname = str(Path(archive_prefix) / rel_root / fname)
                        zf.write(src_file, arcname)

    print(f"  ZIP saved: {zip_path}  ({zip_path.stat().st_size / 1024 / 1024:.1f} MB)")


# ---------------------------------------------------------------------------
# Main benchmark runner
# ---------------------------------------------------------------------------

def run_benchmark(benchmark: str, exp_names: list, exp_sources: list):
    folders = [FOLDER_TEMPLATE.format(benchmark=benchmark, experiment=exp) for exp in EXPERIMENTS]

    rng = random.Random(RANDOM_SEED)
    output_dir = Path(OUTPUT_DIR) / benchmark
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_file = Path(OUTPUT_DIR) / f"cached_tags_{benchmark}.txt"

    # ── Tag selection ─────────────────────────────────────────────────────────
    tags = None
    if CACHE_SAMPLED_TAGS and cache_file.is_file():
        print(f"\n[{benchmark}] Loading cached tags from {cache_file}…")
        with open(cache_file, "r") as f:
            tags = [line.strip() for line in f if line.strip()]
        print(f"  {len(tags)} tags loaded from cache.")
        all_frames = []
        for folder, name, source in zip(folders, exp_names, exp_sources):
            frames = _load_frames_for_tags(folder, tags, source)
            print(f"  {name}: loaded {len(frames)} tags (targeted)")
            all_frames.append(frames)
    else:
        print(f"\n[{benchmark}] Random sampling from first experiment (no cross-experiment check)…")
        tags = _sample_tags(folders[0], exp_sources[0], N_SAMPLES, rng)

        if not tags:
            print(f"  WARNING: no samples found for benchmark '{benchmark}', skipping.")
            return

        print(f"  {len(tags)} tags sampled.")

        if CACHE_SAMPLED_TAGS:
            with open(cache_file, "w") as f:
                for t in tags:
                    f.write(t + "\n")
            print(f"  Saved {len(tags)} tags to {cache_file}.")

        all_frames = []
        for folder, name, source in zip(folders, exp_names, exp_sources):
            frames = _load_frames_for_tags(folder, tags, source)
            print(f"  {name}: loaded {len(frames)} tags (targeted)")
            all_frames.append(frames)

    # ── Plot generation ───────────────────────────────────────────────────────
#    for tag in tqdm(tags, desc=f"{benchmark}"):
 #       per_exp = [frames.get(tag, {}) for frames in all_frames]
  #      out_path = output_dir / f"{tag}_comparison.png"
   #     out_path.parent.mkdir(parents=True, exist_ok=True)
    #    make_comparison(tag, per_exp, exp_names, out_path)

    #print(f"  {len(tags)} comparison plot(s) saved to: {output_dir.resolve()}")

    # ── ZIP export ────────────────────────────────────────────────────────────
    _create_samples_zip(benchmark, tags, folders, EXPERIMENTS, exp_names)


def main():
    exp_names = EXPERIMENT_NAMES if EXPERIMENT_NAMES is not None else EXPERIMENTS
    if len(exp_names) != len(EXPERIMENTS):
        raise ValueError("EXPERIMENT_NAMES length must match EXPERIMENTS length")

    exp_sources = EXPERIMENT_SUBFOLDERS if "EXPERIMENT_SUBFOLDERS" in globals() else [""] * len(EXPERIMENTS)
    if len(exp_sources) != len(EXPERIMENTS):
        raise ValueError("EXPERIMENT_SUBFOLDERS length must match EXPERIMENTS length")

    for benchmark in BENCHMARKS:
        run_benchmark(benchmark, exp_names, exp_sources)

    print("\nAll benchmarks done.")


if __name__ == "__main__":
    main()
