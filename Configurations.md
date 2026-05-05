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
- **What prompt tokens** reach the Fusion Encoder (random support points / matcher points / attention-prior points / dense cross-attention points / self-attention bottom-k points)
- **What feature volumes** are used for matching (backbone vs. fusion encoder output)
- **How many final points** reach the decoder and how they are selected (`--sampling`)
- **How self-attention is computed** (standard query self-attention vs. dense cross-attention to support foreground patches — Exp 6)

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
| `--use_query_as_support` | bool | False | Use the query image itself as the support image (1-shot self-support). Incompatible with `--experiment_mode matcher` and `--experiment_mode self_matching` |

### Experiment Selector

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--experiment_mode` | str | `random` | Selects which experiment / point-selection strategy to run. See experiment descriptions below. Choices: `random`, `matcher`, `self_matching`, `attn_prior`, `dense_cross_attn`, `self_attn_bottomk` |

**`--experiment_mode` choices:**

| Value | Experiment | Description |
|-------|-----------|-------------|
| `random` | Exp 1 / 2 | Random points sampled from support GT masks (default baseline) |
| `matcher` | Exp 3 | Bipartite patch matching support→query to compute prompt points |
| `self_matching` | Exp 4 | Query self-matching: match query features with/without support embeddings |
| `attn_prior` | Exp 5 | Top-k from Fusion Encoder cross-attention map |
| `dense_cross_attn` | Exp 6 | Dense cross-attention to support foreground patches |
| `self_attn_bottomk` | Exp 7 | Bottom-k from Fusion Encoder self-attention map |

### Matcher Controls

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--use_fused_matcher_features` | bool | False | Use fusion encoder output features for matching instead of raw backbone features. Only relevant for `--experiment_mode matcher`; for all other fusion-encoder experiments the fusion encoder is always used |
| `--sampling` | str | `random` | Subsampling strategy applied to the matcher's candidate points. Only active for `experiment_mode=matcher` or `attn_prior --attn_prior_mode rerank_matcher` |

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

### Attention Map Aggregation (Experiments 5, 6, 7)

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--attn_layers` | str | `last` | Which Fusion Encoder layers to aggregate attention from, for both point selection (Exp 5/6/7) and map saving (all experiments). `last` = final layer only (most semantic); `all` = mean over all 6 layers |

### Attention Prior Sub-options (Experiment 5)

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--attn_prior_mode` | str | `topk_sampling` | How to use the cross-attention map. Choices: `topk_sampling` (sample top-k patches directly — no matcher), `rerank_matcher` (run matcher, then reorder candidates by attention weight) |

### Dense Cross-Attention Sub-options (Experiment 6)

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--dense_cross_attn_mode` | str | `topk_sampling` | How to derive final prompt points from the heatmap. Choices: `topk_sampling` (direct top-k from heatmap — no matcher), `rerank_matcher` (run matcher, then rerank by heatmap score) |
| `--dense_cross_attn_skip_text_injection` | bool | False | Skip the pooled-text→image-patch injection before the Fusion Encoder, so query features are purely visual during the dense cross-attention pass |

### Sampling Pass Input Control (Experiments 5 and 7)

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--sampling_inputs` | str | `both` | Controls what goes into the Fusion Encoder during the **point-selection forward pass** (the dedicated pass that extracts the attention map used for point selection). Only relevant for `--experiment_mode attn_prior` and `self_attn_bottomk`. The final SAM3 inference pass is always run with text + support visual tokens |

**`--sampling_inputs` choices:**

| Value | Description |
|-------|-------------|
| `both` | Text label + support visual prompt tokens (default — same as inference pass) |
| `text_only` | Text label only; no support visual tokens in the sampling pass |
| `support_only` | Support visual prompt tokens only; text tokens are excluded from the cross-attention prompt |

This flag is an ablation tool to isolate which input signal is responsible for the attention map's localization quality. Requires `--nshot > 0` when using `support_only`.

### Attention Map Saves (always produced for every experiment)

Four heatmaps are saved per frame, all using **pre-softmax** weights:

| Filename | Content |
|----------|---------|
| `frame_{tag}_cross_total_{attn_layers}.png` | Cross-attn sum over **all** prompt token columns (text + visual) |
| `frame_{tag}_cross_text_{attn_layers}.png` | Cross-attn sum over **text** token columns only |
| `frame_{tag}_cross_points_{attn_layers}.png` | Cross-attn sum over **visual/point** token columns only — decision signal in Exp 5 |
| `frame_{tag}_self_{attn_layers}.png` | Self-attn row-sum (standard Q=K=V=query patches, or dense support cross-attn in Exp 6) — decision signal in Exp 7 |

A fifth file `frame_{tag}_cross_points_{attn_layers}_sampled.png` shows the sampled points overlaid on the points-prior map.

**What each map means per experiment:**

| Map | Exp 1 / 2 / 3 / 4 | Exp 5 — attention prior | Exp 6 — dense cross-attn | Exp 7 — self-attn bottom-k |
|-----|--------------------|-------------------------|--------------------------|---------------------------|
| `cross_total` | Query patches → all prompt tokens (inference pass) | Query patches → all prompt tokens (decision pass) | Query patches → all prompt tokens (decision pass) | Query patches → all prompt tokens (decision pass) |
| `cross_text` | Query patches → text tokens only (inference pass) | Query patches → text tokens only (decision pass) | Query patches → text tokens only (decision pass) | Query patches → text tokens only (decision pass) |
| `cross_points` | Query patches → visual/point tokens (inference pass) | **Decision signal** — top-k of this map selects prompt points | Diagnostic only — not used for point selection | Diagnostic only — not used for point selection |
| `self` | Standard query self-attention (Q=K=V=query patches, inference pass) | Standard query self-attention (inference pass) | **Decision signal** — dense cross-attn: Q=query patches, K/V=support foreground patches | **Decision signal** — row-sum of self-attn; bottom-k selects prompt points |
| `cross_points_sampled` | Support mask points overlaid on `cross_points` | Points selected from `cross_points` overlaid on the map | Points selected from `self` overlaid on `cross_points` | Points selected from `self` overlaid on the map |

**Capture pass by experiment:**
- Exp 1/2/3/4: flags armed before the SAM3 inference forward pass; no extra encoder run
- Exp 5: captured inside `get_fused_image_features` (the decision pass — no extra forward pass)
- Exp 6: captured inside `get_dense_cross_attn_map` (the decision pass — no extra forward pass)
- Exp 7: captured inside `get_fused_image_features` (the decision pass — no extra forward pass)

### Debug / Visualization

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--visualize_embeddings` | bool | False | Save t-SNE plots of patch embeddings and matched points (only for `experiment_mode=matcher`) |

---

## Validation Rules

Enforced by `validate_args()` at startup — invalid combinations raise an exception immediately.

| Rule |
|------|
| `--frame_num` must be > 0 |
| `--experiment_mode matcher` requires `--nshot > 0` |
| `--experiment_mode matcher` is incompatible with `--use_query_as_support` |
| `--experiment_mode self_matching` requires `--nshot > 0` |
| `--experiment_mode self_matching` is incompatible with `--use_query_as_support` |
| `--experiment_mode dense_cross_attn` requires `--nshot > 0` |
| `--experiment_mode self_attn_bottomk --sampling_inputs support_only` requires `--nshot > 0` |
| `--sampling_inputs text_only\|support_only` only applies to `--experiment_mode attn_prior` or `self_attn_bottomk` |

---

## Experiments

### Experiment 1 — Random support points (baseline)

**Flags:**
```
--experiment_mode random
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
--experiment_mode random
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
--experiment_mode matcher
--nshot N
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
--experiment_mode matcher
--nshot N
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
--experiment_mode matcher
--nshot N
[--use_fused_matcher_features]
--sampling k-medoids-points   # or any non-random choice
```

**Flow:** Same as 3a/3b, but the final point selection from matcher candidates uses a structured method (k-medoids, k-means, patch-core, top-k). These strategies improve spatial diversity or feature coverage compared to random sampling.

---

### Experiment 4 — Query self-matching (failed / deprecated)

**Flags:**
```
--experiment_mode self_matching
--nshot N
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
--experiment_mode attn_prior
--nshot N
--attn_prior_mode topk_sampling | rerank_matcher
--attn_layers last | all
[--sampling_inputs both | text_only | support_only]
[--sampling ...]            # applied after rerank_matcher, ignored for topk_sampling
```

**Motivation:** Instead of bipartite matching, use the cross-attention weights of the Fusion Encoder as a localization prior. The Fusion Encoder runs cross-attention with Q = query image patches and K/V = prompt tokens (text + support visual embeddings). Patches with high attention weight are likely to contain the object referred to by the prompts.

**Attention map construction:**
```
For each target layer (last or all 6) — armed via capture_cross_attn_weights=True:
    cross_attn_image fires return_pre_softmax=True, average_attn_weights=True
    pre-softmax logits → [1, 5184, seq_prompt]
    Columns 0..num_text_tokens-1 = text tokens; columns num_text_tokens.. = visual/point tokens
    Split: text_map = sum over text columns; points_map = sum over visual columns
Mean over target layers → [5184] → reshape [72, 72] heatmap per split

Point selection uses last_cross_attn_points_map (visual token prior only).
```

All weights are **pre-softmax** raw scaled dot-products (`Q·Kᵀ / √d`), consistent with Exp 6.

#### Mode: `topk_sampling`

```
Resolve sampling-pass inputs from --sampling_inputs
    both:         use text label + support visual tokens
    text_only:    use text label only (no visual_prompt)
    support_only: use support visual tokens only (no text in cross-attention)
Run Fusion Encoder on query — capture cross-attn weights
Construct attention map [5184] from target layers
topk(attn_map, k=--num_points_from_mask) → patch indices → pixel centers → normalized coords
Encode selected points on query image → visual prompt
Combine with text tokens → Fusion Encoder on query → mask
```

Does NOT use the bipartite matcher. Does NOT go through `--sampling` (the attention ranking is the final selection).

#### Mode: `rerank_matcher`

```
Resolve sampling-pass inputs from --sampling_inputs (applied to the attention capture pass)
Run standard bipartite matcher (support→query, fusion encoder features)
    → candidate points on query (possibly hundreds)
Run Fusion Encoder on query — capture cross-attn weights
Construct attention map [5184]
Look up attention score for each candidate point's patch
Reorder candidates by descending attention score
Apply --sampling to reduce to --num_points_from_mask final points
Encode final points on query image → visual prompt
Combine with text tokens → Fusion Encoder on query → mask
```

The matcher provides geometrically consistent candidates; attention reranks them by semantic relevance. `--sampling` (e.g. `k-medoids-points`) can then enforce spatial diversity on the reranked set.

---

### Experiment 6 — Dense Dual Cross-Attention

**Flags:**
```
--experiment_mode dense_cross_attn
--nshot N
--dense_cross_attn_mode topk_sampling | rerank_matcher
--attn_layers all | last
[--dense_cross_attn_skip_text_injection]
[--sampling ...]            # applied after rerank_matcher, ignored for topk_sampling
```

**Motivation:** In the standard pipeline, the query image only sees the support object through a compressed visual prompt in the Fusion Encoder's cross-attention. Experiment 6 replaces the Fusion Encoder's self-attention with a dense cross-attention where the query image patches directly attend to the support object's foreground patches at patch resolution, enabling explicit patch-level correspondence.

**Why pre-softmax logits are required:** If only foreground support patches are used as Keys, `softmax(dim=-1)` over those K keys always sums to 1.0 per query position — the heatmap would be identically flat. Raw pre-softmax dot-products (`Q·Kᵀ / √d`) are unbounded, so spatial variation is preserved.

**Two-pass design:** The dense cross-attention run is used *exclusively* to extract the localization heatmap. Once points are derived from the heatmap, the pipeline restarts with the standard Fusion Encoder (query patches self-attend normally; cross-attend to text + support visual tokens) to generate the final mask.

**Heatmap construction:**
```
Extract backbone features for N support images → [N, 256, 72, 72]
Avg-pool GT support masks → [N, 72, 72]; threshold at 0.01 → binary foreground mask
Select foreground patches: [K, 256]  (K = total foreground patches across all N images)

For each target encoder layer (all or last):
    self_attn(Q=query patches [5184, 256], K=V=support_fg_patches [K, 256])
    capture pre-softmax logits: [1, 5184, K]
    sum over support patches → [5184]
Mean over target layers → [5184] → reshape [72, 72] heatmap
```

#### Mode: `topk_sampling`

```
Aggregate support visual tokens (random points from GT masks)
Run modified Fusion Encoder pass with dense_support_feats active
    → capture pre-softmax self-attn logits from target layers
Construct heatmap [72, 72]
topk(heatmap, k=--num_points_from_mask) → patch indices → pixel centers → normalized coords
Encode selected points on query image → visual prompt
Combine with text tokens → standard Fusion Encoder on query → mask
```

Does NOT use the bipartite matcher. The heatmap ranking is the final selection — `--sampling` is ignored.

#### Mode: `rerank_matcher`

```
Aggregate support visual tokens (random points from GT masks)
Run modified Fusion Encoder pass → construct heatmap [72, 72]
Run standard bipartite matcher (support→query, fusion encoder features)
    → oversample: --num_points_from_mask × 3 candidate points
Score each candidate by heatmap value at its patch location
Sort candidates by descending heatmap score → keep top --num_points_from_mask
Encode final points on query image → visual prompt
Combine with text tokens → standard Fusion Encoder on query → mask
```

The matcher provides geometrically consistent candidates; the heatmap reranks them by direct patch-level support relevance.

**Implementation notes:**
- `dense_support_feats` is set as per-layer state on each `TransformerEncoderLayer` before the encoder forward pass; `capture_self_attn_weights=True` is set on target layers to record pre-softmax logits into `last_self_attn_weights`
- `return_pre_softmax=True` in `MultiheadAttentionWrapper.forward()` skips the softmax and returns raw scaled dot-products; `average_attn_weights=True` averages over heads → shape `[batch, tgt_len, src_len]`
- The cross-attention layers of the encoder (query→prompt) are unchanged; only self-attention is replaced
- Feature space note: at layers > 0, query tokens have been transformed by prior cross-attention, while support K/V remain at backbone level — a progressive mismatch that is acknowledged and evaluated empirically

---

### Experiment 7 — Self-attention bottom-k

**Flags:**
```
--experiment_mode self_attn_bottomk
--nshot N
--attn_layers last | all
[--sampling_inputs both | text_only | support_only]
```

**Motivation:** The Fusion Encoder's self-attention map (standard Q=K=V=query patches) empirically shows low row-sum values at the target object's location. Experiment 7 exploits this by selecting the k query patches with the *lowest* self-attention activation as prompt points, bypassing explicit cross-image matching entirely.

**Self-attention map construction:**
```
For each target layer (last or all 6) — armed via capture_self_attn_weights=True:
    self_attn(Q=K=V=query patches) fires return_pre_softmax=True, average_attn_weights=True
    pre-softmax logits → [1, 5184, 5184] (heads averaged)
    Row-sum: for each query patch i, sum_j logits[i, j] → scalar → [5184]
Mean over target layers → [5184] → reshape [72, 72] heatmap
```

All weights are **pre-softmax** raw scaled dot-products averaged over heads. This produces the same heatmap as `matcher_calculator.last_self_attn_map` (which is saved as `frame_{tag}_self_{attn_layers}.png`).

**Flow:**
```
Resolve sampling-pass inputs from --sampling_inputs:
    both:         run Fusion Encoder with text label + support visual tokens
    text_only:    run Fusion Encoder with text label only (no visual_prompt)
    support_only: run Fusion Encoder with support visual tokens only (no text in cross-attn)

Run Fusion Encoder on query — capture self-attn weights
Construct self-attention heatmap [5184] from target layers
bottomk(heatmap, k=--num_points_from_mask, largest=False) → patch indices → pixel centers → normalized coords
Convert patch index → pixel center: px = (idx % 72) * 14 + 7,  py = (idx // 72) * 14 + 7
Normalize: [px / 1008, py / 1008]

Encode selected points on query image → visual prompt
Combine with text tokens → Fusion Encoder on query → mask
```

Does NOT use the bipartite matcher or the cross-attention map. `--sampling` is not applied.

**`--sampling_inputs` ablation purpose:**
- `both` (default): replicates the full prompt context used in inference; tests whether the self-attention map is driven by the combined signal
- `text_only`: isolates the contribution of the text label to the self-attention localization
- `support_only`: isolates the contribution of the support visual tokens

---

## Encoder Layer Capture API

`TransformerEncoderLayer` exposes four per-layer state attributes for extracting attention weights without extra forward passes:

| Attribute | Set before pass | Read after pass | Shape stored |
|-----------|----------------|-----------------|--------------|
| `capture_cross_attn_weights` | `True` | `last_cross_attn_weights` | `[batch, 5184, seq_prompt]` |
| `capture_self_attn_weights` | `True` | `last_self_attn_weights` | `[batch, 5184, src_len]` — `src_len` is 5184 (standard) or K support patches (Exp 6) |

Both use `return_pre_softmax=True, average_attn_weights=True` internally — raw scaled dot-products averaged over heads. Always clear both flag and storage after reading (set flag to `False`, storage to `None`) to avoid stale tensors on subsequent passes.

`capture_self_attn_weights` captures whichever keys are active during the forward pass it was armed for: standard self-attention (Q=K=V=query patches) or dense cross-attention to support foreground patches when `dense_support_feats` is injected (Exp 6). No extra forward pass is required for either case.

---

## Flag Dependency Map

```
--experiment_mode matcher
    --use_fused_matcher_features  optional (backbone features used if False)
    --sampling                    active for all matcher sub-modes
    incompatible with: --use_query_as_support

--experiment_mode self_matching
    requires: --nshot > 0
    incompatible with: --use_query_as_support

--experiment_mode attn_prior
    --attn_prior_mode             topk_sampling (default) or rerank_matcher
    --attn_layers                 last (default) or all
    --sampling_inputs             controls fusion encoder inputs in sampling pass
    --sampling                    active only when --attn_prior_mode rerank_matcher

--experiment_mode dense_cross_attn
    requires: --nshot > 0
    --dense_cross_attn_mode       topk_sampling (default) or rerank_matcher
    --dense_cross_attn_skip_text_injection  optional
    --attn_layers                 last (default) or all
    --sampling                    active only when --dense_cross_attn_mode rerank_matcher

--experiment_mode self_attn_bottomk
    --attn_layers                 last (default) or all
    --sampling_inputs             controls fusion encoder inputs in sampling pass
    --sampling_inputs support_only requires --nshot > 0

--sampling_inputs                 applies only to --experiment_mode attn_prior or self_attn_bottomk
--attn_layers                     affects point selection for Exp 5/6/7 and attention map saving for all experiments
--dense_cross_attn_mode           active only for --experiment_mode dense_cross_attn
--attn_prior_mode                 active only for --experiment_mode attn_prior
```

---

## Quick Reference: Experiment Flag Combinations

| Experiment | Key flags |
|-----------|-----------|
| 1 — random support | `--experiment_mode random --nshot N` |
| 2 — self-support | `--experiment_mode random --nshot 1 --use_query_as_support` |
| 3a — matcher + backbone | `--experiment_mode matcher --nshot N` |
| 3b — matcher + fusion features | `--experiment_mode matcher --nshot N --use_fused_matcher_features` |
| 3c — matcher + structured sampling | `--experiment_mode matcher --nshot N [--use_fused_matcher_features] --sampling k-medoids-points` |
| 4 — query self-matching (failed) | `--experiment_mode self_matching --nshot N` |
| 5a — attention prior topk | `--experiment_mode attn_prior --nshot N --attn_prior_mode topk_sampling --attn_layers last` |
| 5b — attention prior rerank | `--experiment_mode attn_prior --nshot N --attn_prior_mode rerank_matcher --attn_layers last` |
| 5c — attention prior, text only sampling | `--experiment_mode attn_prior --nshot N --sampling_inputs text_only --attn_layers last` |
| 6a — dense cross-attn topk | `--experiment_mode dense_cross_attn --nshot N --dense_cross_attn_mode topk_sampling --attn_layers all` |
| 6b — dense cross-attn rerank | `--experiment_mode dense_cross_attn --nshot N --dense_cross_attn_mode rerank_matcher --attn_layers all` |
| 7a — self-attn bottom-k (both) | `--experiment_mode self_attn_bottomk --nshot N --attn_layers last` |
| 7b — self-attn bottom-k, text only | `--experiment_mode self_attn_bottomk --nshot N --sampling_inputs text_only --attn_layers last` |
| 7c — self-attn bottom-k, support only | `--experiment_mode self_attn_bottomk --nshot N --sampling_inputs support_only --attn_layers last` |
