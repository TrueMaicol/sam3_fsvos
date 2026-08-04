"""
Extract the exact sample directories that need to be zipped on the other server.
Uses the same sampling logic as compare_attention_maps.py.

Run with:
    /megaverse/storage/samele/FSS-SAM3/sam3_venv/bin/python3 src/extract_sample_paths.py
"""

import os, re, random
from pathlib import Path

BASE = "/megaverse/storage/samele/FSS-SAM3/output"
BENCHMARKS = ["COCO-20i", "LVIS-92i", "PASCAL-5i"]
REF_EXP = "EXP7_BOX_ALL_SAMP_QIMG_SPROMPT_NO_POS_BIAS"
OTHER_EXPS = [
    "1_SHOT_SELF_ATTN_BOTTOMK_5_POINTS_TEXT_BOX_ALL_LAYERS_NO_POS_BIAS",
    "1_SHOT_SELF_ATTN_BOTTOMK_5_POINTS_TEXT_ONLY_ALL_LAYERS",
]
N_SAMPLES = 10
RANDOM_SEED = 42

MAP_PATTERN = re.compile(
    r"^frame_(.+?)_(cross_total|cross_text|cross_points|self)_(last|all)(_sampled)?\.png$"
)


def discover_tags(folder):
    tags = set()
    folder_path = Path(folder)
    if not folder_path.is_dir():
        print(f"  WARNING: not found: {folder}", flush=True)
        return []
    for attn_dir, _, fnames in os.walk(folder_path):
        if Path(attn_dir).name != "attention_maps":
            continue
        rel = Path(attn_dir).relative_to(folder_path).parent
        for fname in fnames:
            m = MAP_PATTERN.match(fname)
            if m:
                tags.add(str(rel / m.group(1)))
    return sorted(tags)


def main():
    rng = random.Random(RANDOM_SEED)
    all_dirs = []  # (benchmark, sample_rel_str) pairs

    print("=== SAMPLED TAGS (relative to experiment folder) ===\n", flush=True)
    for benchmark in BENCHMARKS:
        folder = f"{BASE}/{benchmark}/{REF_EXP}"
        tags = discover_tags(folder)
        sampled = rng.sample(tags, min(N_SAMPLES, len(tags)))
        sampled.sort()
        print(f"[{benchmark}]  ({len(tags)} total → {len(sampled)} sampled)", flush=True)
        for t in sampled:
            sample_rel = str(Path(t).parent)
            print(f"  {t}  →  sample dir: {sample_rel}", flush=True)
            all_dirs.append((benchmark, sample_rel))
        print()

    print("\n" + "=" * 70, flush=True)
    print("DIRECTORIES TO PACK IN ZIP (paths on the OTHER server)", flush=True)
    print("(pack each as: output/{benchmark}/{experiment}/{sample_rel}/)", flush=True)
    print("=" * 70 + "\n", flush=True)
    for benchmark, sample_rel in all_dirs:
        for exp in OTHER_EXPS:
            print(f"output/{benchmark}/{exp}/{sample_rel}/", flush=True)


if __name__ == "__main__":
    main()
