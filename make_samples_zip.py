#!/usr/bin/env python3
"""
Run this on the other server to create samples.zip.
Packs the selected sample directories for the two experiments.

Usage:
    python3 make_samples_zip.py --base /path/to/base/dir --out samples.zip
"""

import zipfile, sys
from pathlib import Path

# --- CONFIG ---
BASE = Path("/leonardo_work/IscrC_MARSv2/SAM3_FSVOS")  # adjust if needed

EXPERIMENTS = [
    "1_SHOT_SELF_ATTN_BOTTOMK_5_POINTS_TEXT_BOX_ALL_LAYERS_NO_POS_BIAS",
    "1_SHOT_SELF_ATTN_BOTTOMK_5_POINTS_TEXT_ONLY_ALL_LAYERS",
]

SAMPLES = {
    "COCO-20i": [
        "fold_1/val2014/COCO_val2014_000000072582.jpg_0_726",
        "fold_1/val2014/COCO_val2014_000000261363.jpg_36_370",
        "fold_1/val2014/COCO_val2014_000000289444.jpg_32_754",
        "fold_1/val2014/COCO_val2014_000000349896.jpg_20_643",
        "fold_1/val2014/COCO_val2014_000000531896.jpg_28_215",
        "fold_2/val2014/COCO_val2014_000000003837.jpg_37_484",
        "fold_2/val2014/COCO_val2014_000000068771.jpg_77_456",
        "fold_3/val2014/COCO_val2014_000000365456.jpg_38_356",
        "fold_4/val2014/COCO_val2014_000000011099.jpg_67_932",
        "fold_4/val2014/COCO_val2014_000000023446.jpg_63_960",
    ],
    "LVIS-92i": [
        "fold_1/train2017/000000307292.jpg_26_231",
        "fold_1/train2017/000000325149.jpg_55_548",
        "fold_10/train2017/000000187283.jpg_18_1307",
        "fold_10/train2017/000000256940.jpg_56_931",
        "fold_3/train2017/000000085847.jpg_9_1807",
        "fold_3/train2017/000000228943.jpg_38_1051",
        "fold_6/train2017/000000004535.jpg_1_433",
        "fold_7/train2017/000000554040.jpg_50_1603",
        "fold_8/train2017/000000304827.jpg_21_639",
        "fold_9/train2017/000000475465.jpg_25_533",
    ],
    "PASCAL-5i": [
        "fold_1/2007_005705_3_770",
        "fold_1/2010_004556_2_150",
        "fold_2/2010_002868_5_793",
        "fold_3/2007_003194_11_639",
        "fold_3/2007_009562_11_348",
        "fold_3/2008_000811_13_640",
        "fold_3/2008_006722_14_558",
        "fold_3/2009_004738_13_710",
        "fold_3/2010_005992_11_240",
        "fold_3/2011_001862_11_401",
    ],
}

# --- CLI override ---
out_path = Path("samples.zip")
for i, arg in enumerate(sys.argv[1:]):
    if arg == "--base" and i + 1 < len(sys.argv) - 1:
        BASE = Path(sys.argv[i + 2])
    if arg == "--out" and i + 1 < len(sys.argv) - 1:
        out_path = Path(sys.argv[i + 2])

# --- Pack ---
missing = []
total_files = 0

with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as zf:
    for benchmark, sample_rels in SAMPLES.items():
        for exp in EXPERIMENTS:
            for sample_rel in sample_rels:
                src_dir = BASE / "output" / benchmark / exp / sample_rel
                if not src_dir.is_dir():
                    print(f"  MISSING: {src_dir}")
                    missing.append(str(src_dir))
                    continue
                for f in src_dir.rglob("*"):
                    if f.is_file():
                        arcname = Path("output") / benchmark / exp / sample_rel / f.relative_to(src_dir)
                        zf.write(f, arcname)
                        total_files += 1

print(f"\nDone → {out_path}  ({total_files} files packed)")
if missing:
    print(f"WARNING: {len(missing)} directories not found (experiment may not exist on this server):")
    for m in missing:
        print(f"  {m}")
