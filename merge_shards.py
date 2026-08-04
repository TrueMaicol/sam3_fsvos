"""
merge_shards.py — Concatenate per-shard CSV results from test_SAM3_CROSS_IMAGE.py
into a single set of files identical to what a non-sharded run would have produced.

Each shard assigns class_id starting from 0 (local virtual index).
This script re-numbers class_id globally so that each shard's ids are offset
by the total number of virtual classes produced by the previous shards.

Usage
-----
    # Auto-discover all shard dirs:
    python merge_shards.py \\
        --log_dir  /megaverse/storage/samele/FSS-SAM3/experiment_results_logs \\
        --benchmark ade20k \\
        --session_name EXP7_TEXT_ONLY_ALL_LAYERS_ALL_LEMMAS/fold_1 \\
        --auto_discover

    # Or specify the number of shards explicitly:
    python merge_shards.py \\
        --log_dir  /megaverse/storage/samele/FSS-SAM3/experiment_results_logs \\
        --benchmark ade20k \\
        --session_name EXP7_TEXT_ONLY_ALL_LAYERS_ALL_LEMMAS/fold_1 \\
        --num_shards 4

Output is written to:
    <log_dir>/<benchmark>/<parent_of_session>/<base_of_session>_merged/
"""

import argparse
import os
import re
import sys

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Directory helpers
# ---------------------------------------------------------------------------

def shard_dir(log_dir, benchmark, session_name, shard_id, num_shards):
    """
    Build the path for one shard's output directory.
    session_name may contain '/', e.g. 'EXP7_FOO/fold_1'.
    The suffix _shard{i}of{N} is appended to the *last* component only.
    """
    parent = os.path.dirname(session_name)   # e.g. 'EXP7_FOO'
    base   = os.path.basename(session_name)  # e.g. 'fold_1'
    return os.path.join(log_dir, benchmark, parent,
                        f"{base}_shard{shard_id}of{num_shards}")


def discover_shards(log_dir, benchmark, session_name):
    """
    Scan the directory that contains the shard folders and return
    a list of (shard_id, num_shards, abs_path) sorted by shard_id.
    """
    parent   = os.path.dirname(session_name)
    base     = os.path.basename(session_name)
    scan_dir = os.path.join(log_dir, benchmark, parent)

    if not os.path.isdir(scan_dir):
        raise FileNotFoundError(f"Directory not found: {scan_dir}")

    pattern = re.compile(r"^" + re.escape(base) + r"_shard(\d+)of(\d+)$")
    found = []
    for entry in os.listdir(scan_dir):
        m = pattern.match(entry)
        if m:
            found.append((int(m.group(1)), int(m.group(2)),
                          os.path.join(scan_dir, entry)))

    found.sort(key=lambda x: x[0])
    return found


# ---------------------------------------------------------------------------
# Core: concatenate + remap class_id
# ---------------------------------------------------------------------------

CLASS_ID_COLS = ("class_id",)   # column name used in every CSV


def concat_and_remap(shard_paths, csv_filename):
    """
    Load <csv_filename> from every shard directory, offset each shard's
    class_id so they form a single contiguous range, then concatenate.

    The offset for shard k is the sum of (max class_id + 1) across all
    previous shards — i.e. we use the number of unique class_ids in each
    shard's class_scores CSV to know the stride.

    Returns (merged_df, offsets_list) where offsets_list[k] is the integer
    offset applied to shard k.
    """
    frames = []
    offsets = []
    next_id = 0

    for p in shard_paths:
        fpath = os.path.join(p, csv_filename)
        if not os.path.isfile(fpath):
            print(f"  [WARNING] Missing: {fpath}", file=sys.stderr)
            offsets.append(None)
            frames.append(None)
            continue

        df = pd.read_csv(fpath, sep=";")

        offset = next_id
        offsets.append(offset)

        if "class_id" in df.columns:
            # Determine the stride for this shard = number of unique class_ids
            shard_n_classes = int(df["class_id"].max()) + 1
            df = df.copy()
            df["class_id"] = df["class_id"] + offset
            next_id += shard_n_classes
        # else: no class_id column (e.g. box_sizes), just append as-is

        frames.append(df)

    valid = [f for f in frames if f is not None]
    if not valid:
        return pd.DataFrame(), offsets

    merged = pd.concat(valid, ignore_index=True)
    return merged, offsets


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def print_summary(class_scores_df):
    if class_scores_df.empty or "iou_score" not in class_scores_df.columns:
        return

    # mIoU = mean over all virtual class rows (same as a single-shard run)
    mean_iou = class_scores_df["iou_score"].mean()
    n = len(class_scores_df)

    print("\n" + "=" * 60)
    print("  MERGED RESULTS SUMMARY")
    print("=" * 60)
    print(f"  Total virtual classes : {n}")
    print(f"  Global mIoU           : {mean_iou:.6f}  ({mean_iou * 100:.2f} %)")
    for col in ["point_accuracy_micro", "point_accuracy_macro",
                "all_point_accuracy_micro", "all_point_accuracy_macro"]:
        if col in class_scores_df.columns:
            print(f"  Mean {col:35s}: {class_scores_df[col].mean():.6f}")
    print("=" * 60 + "\n")


# ---------------------------------------------------------------------------
# Argument parsing & main
# ---------------------------------------------------------------------------

def get_arguments():
    p = argparse.ArgumentParser(
        description="Merge sharded CSV results from test_SAM3_CROSS_IMAGE.py"
    )
    p.add_argument("--log_dir", type=str,
                   default="/megaverse/storage/samele/FSS-SAM3/experiment_results_logs")
    p.add_argument("--benchmark", type=str, default="ade20k")
    p.add_argument("--session_name", type=str, required=True,
                   help="Session name WITHOUT the shard suffix, "
                        "e.g. EXP7_TEXT_ONLY_ALL_LAYERS_ALL_LEMMAS/fold_1")
    p.add_argument("--num_shards", type=int, default=None,
                   help="Total number of shards. Ignored if --auto_discover is set.")
    p.add_argument("--auto_discover", action="store_true",
                   help="Auto-discover shard directories.")
    p.add_argument("--out_suffix", type=str, default="merged",
                   help="Suffix for the output directory name.")
    p.add_argument("--dry_run", action="store_true",
                   help="Show what would be done without writing files.")
    return p.parse_args()


def main():
    args = get_arguments()

    # ── Find shard directories ──────────────────────────────────────────────
    if args.auto_discover:
        shards = discover_shards(args.log_dir, args.benchmark, args.session_name)
        if not shards:
            print(f"[ERROR] No shard directories found for session "
                  f"'{args.session_name}' under "
                  f"{os.path.join(args.log_dir, args.benchmark, os.path.dirname(args.session_name))}")
            sys.exit(1)
        num_shards   = shards[0][1]
        found_ids    = [s[0] for s in shards]
        shard_paths  = [s[2] for s in shards]
        print(f"[Discover] Found {len(shards)} shard(s): ids={found_ids}, declared total={num_shards}")
        missing = set(range(num_shards)) - set(found_ids)
        if missing:
            print(f"  [WARNING] Missing shard ids: {sorted(missing)} — results will be incomplete!")
    else:
        if args.num_shards is None:
            print("[ERROR] Provide --num_shards N or use --auto_discover.")
            sys.exit(1)
        num_shards  = args.num_shards
        shard_paths = []
        for sid in range(num_shards):
            p = shard_dir(args.log_dir, args.benchmark, args.session_name, sid, num_shards)
            if os.path.isdir(p):
                shard_paths.append(p)
                print(f"  [OK]      shard {sid}: {p}")
            else:
                shard_paths.append(p)   # keep placeholder; concat_and_remap will warn
                print(f"  [MISSING] shard {sid}: {p}")

    print(f"\n[Shards] Processing order:")
    for i, p in enumerate(shard_paths):
        print(f"  {i}: {p}")

    # ── Output directory ────────────────────────────────────────────────────
    parent     = os.path.dirname(args.session_name)
    base       = os.path.basename(args.session_name)
    out_dir    = os.path.join(args.log_dir, args.benchmark, parent,
                              f"{base}_{args.out_suffix}")
    print(f"\n[Output] → {out_dir}")
    if not args.dry_run:
        os.makedirs(out_dir, exist_ok=True)

    bm = args.benchmark

    # ── CSV files to merge ──────────────────────────────────────────────────
    csv_files = [
        f"{bm}_class_scores.csv",
        f"{bm}_sample_scores.csv",
        f"{bm}_blob_scores.csv",
        f"{bm}_size_scores.csv",
        f"{bm}_point_features.csv",
        f"{bm}_box_sizes_scores.csv",
    ]

    class_scores_df = None

    for csv_name in csv_files:
        print(f"\n  [{csv_name}]", end=" ", flush=True)
        merged, offsets = concat_and_remap(shard_paths, csv_name)

        if merged.empty:
            print("(empty — skipped)")
            continue

        print(f"{len(merged)} rows  |  class_id offsets per shard: {offsets}")

        if csv_name == f"{bm}_class_scores.csv":
            class_scores_df = merged

        out_path = os.path.join(out_dir, csv_name)
        if not args.dry_run:
            merged.to_csv(out_path, index=False, sep=";")
            print(f"    → {out_path}")

    # ── Summary ─────────────────────────────────────────────────────────────
    if class_scores_df is not None:
        print_summary(class_scores_df)


if __name__ == "__main__":
    main()
