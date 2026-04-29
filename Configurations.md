# Configurations — `test_SAM3_CROSS_IMAGE.py`

This file documents every flag, every experiment, and the inference flow of the SAM3 cross-image few-shot segmentation pipeline.

---

## Architecture Primer

The pipeline runs on a single query image given N support images with GT masks.

```
Support images + masks
    │
    ▼
encode_support_visual_tokens()          # random points sampled from GT masks → SAM3 point encoder
    │ aggregated_visual_prompt [seq, 1, 256]
    │ + text tokens (if --disable_text is off)
    ▼
Fusion Encoder (TransformerEncoderFusion, 6 layers)
    Q = image patches of query image [5184, 256]
    K/V = prompt tokens (text + visual) [seq_prompt, 256]
    → conditioned query image features [5184, 256]
    ▼
SAM3 Decoder → segmentation mask
```

The key design levers are:
- **What prompt tokens** reach the Fusion Encoder (random support points / matcher points / attention-prior points)
- **What feature volumes** are used for matching (backbone vs. fusion encoder output)
- **How many final points** reach the decoder and how they are selected (`--sampling`)

---

## All Flags

### Infrastructure

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--checkpoint` | str | None | Path to SAM3 model checkpoint |
| `--benchmark` | str | `youtube-fsvos` | Dataset benchmark to evaluate on. Choices: `youtube_fsvos`, `minivspw`, `coco`, `lvis`, `ade20k`, `pascal`, `coco-20i`, `pascal-5i`, `lvis-92i` |
| `--session_name` | str | None | Name for this run (used in output paths and logs) |
| `--dataset_path` | str | None | Root path to the dataset |
| `--data_list_path` | str | None | Path to the data list / split file |
| `--output_dir` | str | `./output` | Directory for output masks and visualizations |
| `--fold` | int | 1 | Cross-validation fold (used by COCO-20i, Pascal-5i, LVIS-92i) |
| `--frame_num` | int | 1 | Number of query frames to evaluate per episode |
| `--nshot` | int | 1 | Number of support images per episode (N-shot) |
| `--run_n` | int | 0 | Run index (for repeated experiments with different seeds) |
| `--seed` | int | 0 | Random seed for reproducibility |
| `--log_dir` | str | `.../JOB_OUTPUT/logs` | Directory for log files |

### Class Label Handling

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--use_synset_names` | bool | False | Use WordNet synset names as text labels instead of raw class names |
| `--synset_mapping_folder_path` | str | `.../synset_mappings` | Path to the synset mapping JSON files |
| `--use_grouping_ade20k` | bool | False | Group ADE20K classes using a JSON mapping (ADE20K only) |
| `--all_lemmas` | bool | False | Iterate over all WordNet lemmas for a class, instead of just the canonical one |
| `--disable_text` | bool | False | Replace the class label with the dummy token `"visual"` — effectively disables text conditioning |

### Point Prompt Controls

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--num_points_from_mask` | int | 20 | Number of points to sample / select as geometric prompts for the decoder |
| `--skip_coords` | bool | False | Skip spatial coordinate embeddings when encoding point prompts. When False, the exemplar encoder receives both appearance and position information; when True, position is dropped |
| `--use_query_as_support` | bool | False | Use the query image itself as the support image (1-shot self-support). Incompatible with `--matcher_points` and `--use_query_self_matching` |

### Matcher Controls

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--matcher_points` | bool | False | Use bipartite patch matching (support→query) to compute prompt points instead of random sampling. Requires `--nshot > 0` |
| `--use_fused_matcher_features` | bool | False | Use fusion encoder output features for matching instead of raw backbone features. Fusion features are conditioned on the text label |
| `--sampling` | str | `random` | Subsampling strategy applied to the matcher's candidate points. Choices below. Only active when `--matcher_points` or `--use_attn_prior rerank_matcher` is used |

**`--sampling` choices:**

| Value | Method |
|-------|--------|
| `random` | Uniform random subsampling to `--num_points_from_mask` |
| `top-k` | Keep top-k candidates by cosine similarity score |
| `patch-core` | Greedy coreset subsampling (maximises coverage in feature space) |
| `k-means-embeddings` | K-Means clustering on patch embeddings, one centroid per cluster |
| `k-means-points` | K-Means clustering on 2D point coordinates |
| `k-medoids-embeddings` | K-Medoids on patch embeddings (returns actual data points) |
| `k-medoids-points` | K-Medoids on 2D point coordinates |

### Query Self-Matching (Experiment 4 — failed)

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--use_query_self_matching` | bool | False | Run bipartite matching between two feature volumes of the query image: one conditioned on text + support visual embeddings, one conditioned on text only. Required flags: `--matcher_points`, `--use_fused_matcher_features`, `--nshot > 0` |

### Attention Prior (Experiment 5)

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--use_attn_prior` | bool | False | Use cross-attention weights from the Fusion Encoder as a localization prior on the query image. Required flag: `--use_fused_matcher_features` |
| `--attn_prior_mode` | str | `topk_sampling` | How to use the attention map. Choices: `topk_sampling` (sample top-k patches directly — no matcher), `rerank_matcher` (run matcher, then reorder candidates by attention weight) |
| `--attn_prior_layers` | str | `last` | Which Fusion Encoder layers to aggregate attention from. `last` = final layer only (most semantic); `all` = mean over all 6 layers |

### Debug / Visualization

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--visualize_embeddings` | bool | False | Save t-SNE plots of patch embeddings and matched points |

---

## Validation Rules

Enforced by `validate_args()` at startup — invalid combinations raise an exception immediately.

| Rule |
|------|
| `--matcher_points` requires `--nshot > 0` |
| `--matcher_points` is incompatible with `--use_query_as_support` |
| `--frame_num` must be > 0 |
| `--use_query_self_matching` requires `--matcher_points` |
| `--use_query_self_matching` requires `--use_fused_matcher_features` |
| `--use_query_self_matching` is incompatible with `--use_query_as_support` |
| `--use_query_self_matching` requires `--nshot > 0` |
| `--use_attn_prior` requires `--use_fused_matcher_features` |
| `--use_attn_prior --attn_prior_mode rerank_matcher` requires `--matcher_points` |
| `--use_attn_prior` is incompatible with `--use_query_self_matching` |

---

## Experiments

### Experiment 1 — Random support points (baseline)

**Flags:**
```
--nshot N
[--disable_text]
[--skip_coords]
```

**Flow:**
```
For each support image:
    sample --num_points_from_mask random points from GT mask
    encode points via SAM3 exemplar encoder
        (--skip_coords: drop spatial info from encoding)
Aggregate all support point embeddings → visual prompt
Combine with text label tokens (unless --disable_text)
    → final prompt fed to Fusion Encoder on query image
Fusion Encoder: query patches cross-attend to final prompt
SAM3 decoder → mask
```

**Notes:** The visual prompt encodes *where* and *what* is in the support. `--skip_coords` tests whether removing spatial information hurts (it typically does, because the exemplar encoder partially encodes object appearance via position).

---

### Experiment 2 — Self-support (query image as its own support)

**Flags:**
```
--nshot 1
--use_query_as_support
[--disable_text]
[--skip_coords]
```

**Flow:** Identical to Experiment 1, but the support image is replaced by the query image itself. The GT mask of the query is used to sample the support points. This tests what the model can do when it "sees" the answer — an upper-bound sanity check.

---

### Experiment 3a — Matcher with backbone features

**Flags:**
```
--nshot N
--matcher_points
[--sampling random|top-k|...]
[--disable_text]
[--skip_coords]
```

**Flow:**
```
Extract backbone (PE) features for support images → ref_features [N*5184, 256]
Extract backbone (PE) features for query image   → tar_features [5184, 256]

Bipartite matching (forward + backward):
    forward:  linear_sum_assignment(ref→tar) → candidate matches
    backward: linear_sum_assignment(tar→ref on matched subset) → filter to consistent pairs
    reduce to top half by similarity score

Apply --sampling to reduce to --num_points_from_mask final points
Encode final points on query image → visual prompt
Combine with text tokens → Fusion Encoder on query → mask
```

**Notes:** Features are L2-normalized backbone outputs (not conditioned on text). Matching uses cosine similarity.

---

### Experiment 3b — Matcher with fusion encoder features

**Flags:**
```
--nshot N
--matcher_points
--use_fused_matcher_features
[--sampling random|top-k|...]
[--disable_text]
[--skip_coords]
```

**Flow:** Same as 3a, but both feature volumes are obtained from the Fusion Encoder output (after cross-attention with text tokens). Text conditioning makes features more semantically aligned with the class label before matching.

---

### Experiment 3c — Matcher with non-random subsampling

**Flags:**
```
--nshot N
--matcher_points
[--use_fused_matcher_features]
--sampling k-medoids-points   # or any non-random choice
```

**Flow:** Same as 3a/3b, but the final point selection from matcher candidates uses a structured method (k-medoids, k-means, patch-core, top-k). These strategies improve spatial diversity or feature coverage compared to random sampling.

---

### Experiment 4 — Query self-matching (failed / deprecated)

**Flags:**
```
--nshot N
--matcher_points
--use_fused_matcher_features
--use_query_self_matching
[--sampling ...]
```

**Flow:**
```
Aggregate support visual tokens (random points from support GT masks)

Reference volume: query image fused with (text + support visual tokens)
Target volume:    query image fused with (text only)

Bipartite matching between reference and target volumes
    → both come from the same image → near-identity matching
    → backward filter discards almost nothing (~2592 candidates remain)
    → k-medoids does all the real selection

Apply --sampling → final points on query → Fusion Encoder → mask
```

**Why it failed:** Since both feature volumes are computed from the identical query image, the bipartite matching degenerates into a near-identity assignment. All 5184 patches match, the backward filter passes ~half, and the sampling strategy has to do all the work with no geometric signal from the matching step.

---

### Experiment 5 — Attention prior

**Flags:**
```
--nshot N
--use_fused_matcher_features
--use_attn_prior
--attn_prior_mode topk_sampling | rerank_matcher
--attn_prior_layers last | all
[--matcher_points]          # required only for rerank_matcher
[--sampling ...]            # applied after rerank_matcher, ignored for topk_sampling
```

**Motivation:** Instead of bipartite matching, use the cross-attention weights of the Fusion Encoder as a localization prior. The Fusion Encoder runs cross-attention with Q = query image patches and K/V = prompt tokens (text + support visual embeddings). Patches with high attention weight are likely to contain the object referred to by the prompts.

**Attention map construction:**
```
For each target layer (last or all 6):
    cross_attn_image weights → [1, 5184, seq_prompt]  (heads averaged by PyTorch)
    sum over seq_prompt dim  → [1, 5184]              (total attention per patch)
Mean over layers             → [5184]                 → reshape [72, 72] heatmap
```

#### Mode: `topk_sampling`

```
Aggregate support visual tokens (random points from GT masks)
Run Fusion Encoder on query with (text + support visual tokens) — capture cross-attn weights
Construct attention map [5184] from target layers
topk(attn_map, k=--num_points_from_mask) → patch indices → pixel centers → normalized coords
Encode selected points on query image → visual prompt
Combine with text tokens → Fusion Encoder on query → mask
```

Does NOT use the bipartite matcher. Does NOT go through `--sampling` (the attention ranking is the final selection).

#### Mode: `rerank_matcher`

```
Aggregate support visual tokens (random points from GT masks)
Run standard bipartite matcher (support→query, fusion encoder features)
    → candidate points on query (possibly hundreds)
Run Fusion Encoder on query with (text + support visual tokens) — capture cross-attn weights
Construct attention map [5184]
Look up attention score for each candidate point's patch
Reorder candidates by descending attention score
Apply --sampling to reduce to --num_points_from_mask final points
Encode final points on query image → visual prompt
Combine with text tokens → Fusion Encoder on query → mask
```

The matcher provides geometrically consistent candidates; attention reranks them by semantic relevance. `--sampling` (e.g. `k-medoids-points`) can then enforce spatial diversity on the reranked set.

---

## Flag Dependency Map

```
--use_query_self_matching
    requires: --matcher_points, --use_fused_matcher_features, --nshot > 0
    incompatible with: --use_query_as_support, --use_attn_prior

--use_attn_prior
    requires: --use_fused_matcher_features
    --attn_prior_mode rerank_matcher also requires: --matcher_points
    incompatible with: --use_query_self_matching

--matcher_points
    requires: --nshot > 0
    incompatible with: --use_query_as_support

--sampling         active only when --matcher_points or (--use_attn_prior --attn_prior_mode rerank_matcher)
--attn_prior_mode  active only when --use_attn_prior
--attn_prior_layers active only when --use_attn_prior
```

---

## Quick Reference: Experiment Flag Combinations

| Experiment | Key flags |
|-----------|-----------|
| 1 — random support | `--nshot N` |
| 2 — self-support | `--nshot 1 --use_query_as_support` |
| 3a — matcher + backbone | `--nshot N --matcher_points` |
| 3b — matcher + fusion features | `--nshot N --matcher_points --use_fused_matcher_features` |
| 3c — matcher + structured sampling | `--nshot N --matcher_points [--use_fused_matcher_features] --sampling k-medoids-points` |
| 4 — query self-matching (failed) | `--nshot N --matcher_points --use_fused_matcher_features --use_query_self_matching` |
| 5a — attention prior topk | `--nshot N --use_fused_matcher_features --use_attn_prior --attn_prior_mode topk_sampling` |
| 5b — attention prior rerank | `--nshot N --matcher_points --use_fused_matcher_features --use_attn_prior --attn_prior_mode rerank_matcher` |
