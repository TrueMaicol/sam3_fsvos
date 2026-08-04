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
    │ + text tokens (real class label, or "visual" sentinel if --disable_text_inference)
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
| `--disable_text_inference` | bool | False | Replace the real class label with the `"visual"` sentinel token **at inference time only**. SAM3 still receives a text token (`"visual"`) but no custom class label. To suppress the class label in the sampling pass too, combine with `--sampling_inputs support_only` (Exp 5 and 7 only) — see [Suppressing the class label in both passes](#suppressing-the-class-label-in-both-passes) |

### Point Prompt Controls

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--num_points_from_mask` | int | 20 | Number of points to sample / select as geometric prompts for the decoder |
| `--skip_coords` | bool | False | Skip spatial coordinate embeddings when encoding point prompts. When False, the exemplar encoder receives both appearance and position information; when True, position is dropped |
| `--use_query_as_support` | bool | False | Use the query image itself as the support image (1-shot self-support). Incompatible with `--experiment_mode matcher` and `--experiment_mode self_matching` |
| `--sample_points_from_image` | bool | False | When `--experiment_mode random` and `--support_prompt_type points`: sample the point prompts **uniformly from the entire image canvas** instead of only from the foreground mask region. Applies to Exp 1 (random points over the full support image) and Exp 2 (random points over the full query image, via `--use_query_as_support`). Has no effect when `--support_prompt_type box`. |

### Support Prompt Type

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--support_prompt_type` | str | `points` | Type of visual prompt built from support images. `points`: random points sampled from the GT mask (default). `box`: bounding box of a connected foreground blob (see `--blob_selection`), encoded via the geometry encoder's ROI-align path. Applies to `--experiment_mode random` (direct support encoding) and `attn_prior` / `self_attn` (sampling pass). |
| `--blob_selection` | str | `largest` | Which connected foreground blob to use when `--support_prompt_type box`. `largest` (default): biggest blob by pixel area. `smallest`: smallest blob with area ≥ 10 px (blobs smaller than 10 px are treated as noise and ignored). |

**`--support_prompt_type` choices:**

| Value | Description |
|-------|-------------|
| `points` | Random points from support GT mask — scattered spatial tokens, each capturing a point's appearance and (unless `--skip_coords`) position |
| `box` | Bounding box of a connected blob (largest or smallest, controlled by `--blob_selection`). The geometry encoder encodes the box as a single token (or two corner tokens if `encode_boxes_as_points=True`) using ROI-align pooling to aggregate visual content within the box region. When `--skip_coords` is False, coordinate embeddings are also added; when True, only the ROI-pooled visual content is encoded |

**Notes:**
- With `--support_prompt_type box`, connected-components is always run; `--blob_selection` then picks the target blob.
- `--blob_selection smallest` ignores blobs smaller than 10 px to skip noise artifacts.
- `--blob_selection` is only meaningful when `--support_prompt_type box`; it has no effect for `points`.
- Incompatible with `--sampling_inputs text_only` (no visual tokens are built in that mode).
- Not applicable to `--experiment_mode matcher`, `self_matching`, or `dense_cross_attn`.

### Experiment Selector

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--experiment_mode` | str | `random` | Selects which experiment / point-selection strategy to run. See experiment descriptions below. Choices: `random`, `matcher`, `self_matching`, `attn_prior`, `dense_cross_attn`, `self_attn` |

**`--experiment_mode` choices:**

| Value | Experiment | Description |
|-------|-----------|-------------|
| `random` | Exp 1 / 2 | Random points sampled from support GT masks (default baseline) |
| `matcher` | Exp 3 | Bipartite patch matching support→query to compute prompt points |
| `self_matching` | Exp 4 | Query self-matching: match query features with/without support embeddings |
| `attn_prior` | Exp 5 | Top-k from Fusion Encoder cross-attention map |
| `dense_cross_attn` | Exp 6 | Dense cross-attention to support foreground patches |
| `self_attn` | Exp 7 | Bottom-k (or top-k) from Fusion Encoder self-attention map |

### Matcher Controls

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--use_fused_matcher_features` | bool | False | Use fusion encoder output features for matching instead of raw backbone features. Only relevant for `--experiment_mode matcher`; for all other fusion-encoder experiments the fusion encoder is always used |
| `--sampling` | str | `random` | Subsampling strategy applied to the matcher's candidate points. Only active for `experiment_mode=matcher` |

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

### Attention Map Aggregation & Sampling (Experiments 5, 6, 7)

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--attn_layers` | str | `last` | Which Fusion Encoder layers to aggregate attention from, for both point selection (Exp 5/6/7) and map saving (all experiments). `last` = final layer only (most semantic); `all` = mean over all 6 layers |
| `--attention_aggregate_function` | str | `sum` | Aggregation function used to aggregate attention matrices on the key dimension. Choices: `sum`, `mean`, `max`, `min`, `top-*-mean` (e.g. `top-5-mean`) |
| `--attn_sampling_mode` | str | `top-k` | Point sampling method for attention priors. Choices: `top-k`, `bottom-k` |

### Text Pooling Injection (all experiments)

The pooled-text bias is the projected mean-pooled text embedding (`text_pooling_proj(pool_text_feat(...))`), shape `[1, 256]`. It mirrors the text conditioning applied during SAM3 training. Injection is controlled by one master switch plus three sub-flags:

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--inject_text_pooling` | bool | False | Master on/off. When False, all sub-flags are no-ops. When True, the bias is added to query image patches at the stages selected by `--injection_text_pooling_stage`; support-side targets are only biased when the corresponding `in_prompts_*` flag is set. |
| `--injection_text_pooling_stage` | str | `point_sampling` | Which forward pass(es) receive the bias. Choices: `point_sampling` (Fusion Encoder pass that extracts the attention map for Exp 5/6/7), `inference_pass` (final SAM3 inference forward), `both`. |
| `--injection_text_pooling_in_prompts_sampling` | bool | False | During the point-sampling pass, also bias support-side tokens: visual prompt tokens for Exp 5/7, dense support spatial K/V for Exp 6. No effect on Exp 1/2/3/4 (no separate sampling pass). |
| `--injection_text_pooling_in_prompts_inference` | bool | False | During the inference forward, also bias the aggregated visual prompt fed to the encoder. Applies to all experiments. For Exp 1, this replaces the previous per-shot bias loop and is mathematically equivalent. |

**Effect matrix (when master flag is on):**

| Target | Image patches (query) | Visual prompt tokens (Exp 5/7) | Dense support K/V (Exp 6) | Aggregated visual prompt (inference, all Exp) |
|--------|-----------------------|--------------------------------|---------------------------|------------------------------------------------|
| Sampling pass | `stage ∈ {point_sampling, both}` | `stage ∈ {point_sampling, both} AND in_prompts_sampling` | `stage ∈ {point_sampling, both} AND in_prompts_sampling` | — |
| Inference pass | `stage ∈ {inference_pass, both}` | — | — | `stage ∈ {inference_pass, both} AND in_prompts_inference` |

`pool_text_feat` is computed at most once per `cross_image_prediction` call regardless of how many stages or targets are biased.

### Sampling Pass Input Control (Experiments 5 and 7)

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--sampling_inputs` | str | `both` | Controls what goes into the Fusion Encoder during the **point-selection forward pass** (the dedicated pass that extracts the attention map used for point selection). Only relevant for `--experiment_mode attn_prior` and `self_attn`. `support_only` also applies to `matcher`: it replaces the class label with the `"visual"` sentinel in the fusion feature extraction pass. The final SAM3 inference pass is always run with text + support visual tokens |

**`--sampling_inputs` choices:**

| Value | Description |
|-------|-------------|
| `both` | Text label + support visual prompt tokens (default — same as inference pass). For `matcher`: class label is used in the fusion feature extraction pass |
| `text_only` | Text label only; no support visual tokens in the sampling pass. Only valid for `attn_prior`/`self_attn` |
| `support_only` | Support visual tokens + `"visual"` sentinel token (class label suppressed). For `attn_prior`/`self_attn`: also excludes class label from the fusion sampling pass. For `matcher`: replaces the class label with `"visual"` in the fusion feature extraction pass (`--use_fused_matcher_features` only) |

This flag is an ablation tool to isolate which input signal is responsible for the attention map's localization quality. Requires `--nshot > 0` when using `support_only`.

#### Suppressing the class label in both passes

To run a fully class-label-free experiment on Exp 5/7 — where SAM3 uses only the `"visual"` sentinel (no custom text label) in **both** the sampling pass and the inference pass — combine both flags:

```
--sampling_inputs support_only   # sampling pass: "visual" sentinel + support visual tokens
--disable_text_inference          # inference pass: "visual" sentinel + aggregated visual prompt
```

The `"visual"` token is always present (SAM3 requires a text input for architectural reasons), but no custom class label is used anywhere in the pipeline.

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

| Map | Exp 1 / 2 / 3 / 4 | Exp 5 — attention prior | Exp 6 — dense cross-attn | Exp 7 — self-attn |
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
| `--experiment_mode self_attn --sampling_inputs support_only` requires `--nshot > 0` |
| `--experiment_mode attn_prior --sampling_inputs support_only` requires `--nshot > 0` |
| `--sampling_inputs text_only` only applies to `--experiment_mode attn_prior` or `self_attn` |
| `--sampling_inputs support_only` applies to `--experiment_mode attn_prior`, `self_attn`, or `matcher` |
| `--support_prompt_type box` is incompatible with `--sampling_inputs text_only` |
| `--support_prompt_type box` only applies to `--experiment_mode attn_prior`, `self_attn`, or `random` |

---

## Experiments

### Experiment 1 — Random support points (baseline)

**Flags:**
```
--experiment_mode random
--nshot N
[--support_prompt_type points|box]
[--disable_text_inference]
[--skip_coords]
[--sample_points_from_image]
```

**Flow:**
```
For each support image:
    if --support_prompt_type points (default):
        if --sample_points_from_image:
            sample --num_points_from_mask random points from the full support image (uniform over all pixels)
        else:
            sample --num_points_from_mask random points from GT mask (default)
        encode points via SAM3 geometry encoder → 1 token per point
    if --support_prompt_type box:
        run connected-components on GT mask → select largest blob
        compute bounding box [cx, cy, w, h] (normalized [0,1])
        encode box via SAM3 geometry encoder (ROI-align) → 1 token per support image
        (--skip_coords: use ROI-align content only, drop coordinate embeddings)
Aggregate all support tokens → visual prompt
Combine with text label tokens (or "visual" sentinel if --disable_text_inference)
    → final prompt fed to Fusion Encoder on query image
Fusion Encoder: query patches cross-attend to final prompt
SAM3 decoder → mask
```

**Notes:** The visual prompt encodes *where* and *what* is in the support. `--skip_coords` tests whether removing spatial information hurts (it typically does, because the exemplar encoder partially encodes object appearance via position). With `--support_prompt_type box`, a single ROI-pooled token represents the whole object region, collapsing N scattered point tokens into one holistic representation. `--sample_points_from_image` replaces the foreground-constrained sampling with uniform image-wide sampling — useful only as an ablation for Exp 1 (what if we give random positions on the support image?) and as the main variant for Exp 2 (see below).

---

### Experiment 2 — Self-support (query image as its own support)

Two variants depending on where the point prompts are sampled from.

#### Variant A — GT-mask points (canonical upper bound)

**Flags:**
```
--experiment_mode random
--nshot 1
--use_query_as_support
[--disable_text_inference]
[--skip_coords]
```

**Flow:** Identical to Experiment 1, but the support image is replaced by the query image itself. The GT mask of the query is used to sample the support points. This tests what the model can do when it "sees" the answer — an upper-bound sanity check.

#### Variant B — Fully random image points

**Flags:**
```
--experiment_mode random
--nshot 1
--use_query_as_support
--sample_points_from_image
[--disable_text_inference]
[--skip_coords]
```

**Flow:** Same substitution as Variant A (query image replaces the support), but the point prompts are sampled **uniformly from the entire query image canvas** — no foreground mask constraint. The model receives `--num_points_from_mask` random point hints with no localization bias, plus the text label. Variant B tests a purely text-guided prompt regime, measuring what information the geometric encoder can contribute from randomly placed, semantically uninformed points.

**Implementation note:** `--sample_points_from_image` replaces the foreground mask with an all-ones mask before calling `get_random_points_from_mask`, so `np.where` returns all pixel positions and `np.random.choice` samples from the full image. The selected pixel coordinates are relative to the transformed image resolution (1008 × 1008).

---

### Experiment 3a — Matcher with backbone features

**Flags:**
```
--experiment_mode matcher
--nshot N
[--sampling random|top-k|...]
[--disable_text_inference]
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
[--sampling_inputs support_only]
[--disable_text_inference]
[--skip_coords]
```

**Flow:** Same as 3a, but both feature volumes are obtained from the Fusion Encoder output (after cross-attention with text tokens). Text conditioning makes features more semantically aligned with the class label before matching.

> [!NOTE]
> `--sampling_inputs support_only` replaces the class label with the `"visual"` sentinel in the **fusion feature extraction pass** (both support and query feature volumes computed by `compute_box`). Combined with `--disable_text_inference`, neither the matching features nor the final inference uses the real class label. For Exp 3a (backbone features), `--sampling_inputs support_only` has no effect since backbone features are text-agnostic.

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
--attn_layers last | all
--attention_aggregate_function sum | mean | max | min | top-*-mean
--attn_sampling_mode top-k | bottom-k
[--sampling_inputs both | text_only | support_only]
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

**Flow:**
```
Resolve sampling-pass inputs from --sampling_inputs
    both:         use text label + support visual tokens
    text_only:    use text label only (no visual_prompt)
    support_only: use support visual tokens + "visual" sentinel (class label suppressed; SAM3 still receives its required text token)
Run Fusion Encoder on query — capture cross-attn weights (aggregated via --attention_aggregate_function)
Construct attention map [5184] from target layers
Sample patches based on --attn_sampling_mode (top-k or bottom-k, k=--num_points_from_mask) → patch indices → pixel centers → normalized coords
Encode selected points on query image → visual prompt
Combine with text tokens → Fusion Encoder on query → mask
```

Does NOT use the bipartite matcher. Does NOT go through `--sampling` (the attention map ranking is the final selection).

---

### Experiment 6 — Dense Dual Cross-Attention

**Flags:**
```
--experiment_mode dense_cross_attn
--nshot N
--attn_layers all | last
--attention_aggregate_function sum | mean | max | min | top-*-mean
--attn_sampling_mode top-k | bottom-k
[--inject_text_pooling
 --injection_text_pooling_stage point_sampling|inference_pass|both
 --injection_text_pooling_in_prompts_sampling
 --injection_text_pooling_in_prompts_inference]
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

**Flow:**

```
Aggregate support visual tokens (random points from GT masks)
Run modified Fusion Encoder pass with dense_support_feats active
    → capture pre-softmax self-attn logits from target layers (aggregated via --attention_aggregate_function)
Construct heatmap [72, 72]
Sample patches based on --attn_sampling_mode (top-k or bottom-k, k=--num_points_from_mask) → patch indices → pixel centers → normalized coords
Encode selected points on query image → visual prompt
Combine with text tokens → standard Fusion Encoder on query → mask
```

Does NOT use the bipartite matcher. The heatmap ranking is the final selection — `--sampling` is ignored.

**Implementation notes:**
- `dense_support_feats` is set as per-layer state on each `TransformerEncoderLayer` before the encoder forward pass; `capture_self_attn_weights=True` is set on target layers to record pre-softmax logits into `last_self_attn_weights`
- `return_pre_softmax=True` in `MultiheadAttentionWrapper.forward()` skips the softmax and returns raw scaled dot-products; `average_attn_weights=True` averages over heads → shape `[batch, tgt_len, src_len]`
- The cross-attention layers of the encoder (query→prompt) are unchanged; only self-attention is replaced
- Feature space note: at layers > 0, query tokens have been transformed by prior cross-attention, while support K/V remain at backbone level — a progressive mismatch that is acknowledged and evaluated empirically

---

### Experiment 7 — Self-attention

**Flags:**
```
--experiment_mode self_attn
--nshot N
--attn_layers last | all
--attention_aggregate_function sum | mean | max | min | top-*-mean
--attn_sampling_mode top-k | bottom-k
[--sampling_inputs both | text_only | support_only]
[--support_prompt_type points|box]
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
Resolve sampling-pass inputs:
    --sampling_inputs both:         run Fusion Encoder with text label + support visual tokens
    --sampling_inputs text_only:    run Fusion Encoder with text label only (no visual_prompt)
    --sampling_inputs support_only: run Fusion Encoder with support visual tokens + "visual" sentinel (class label suppressed)

    Support visual tokens built from (when sampling_inputs is both or support_only):
        --support_prompt_type points (default): random points from GT mask
        --support_prompt_type box:              bounding box of largest blob (ROI-align token)

Run Fusion Encoder on query — capture self-attn weights
Construct self-attention heatmap [5184] from target layers
Sample patches based on --attn_sampling_mode (top-k or bottom-k, k=--num_points_from_mask)
    → patch indices → pixel centers → normalized coords
Convert patch index → pixel center: px = (idx % 72) * 14 + 7,  py = (idx // 72) * 14 + 7
Normalize: [px / 1008, py / 1008]

Encode selected points on query image → visual prompt
Combine with text tokens → Fusion Encoder on query → mask
```

Does NOT use the bipartite matcher or the cross-attention map. `--sampling` is not applied.

**`--sampling_inputs` ablation purpose:**
- `both` (default): replicates the full prompt context used in inference; tests whether the self-attention map is driven by the combined signal
- `text_only`: isolates the contribution of the text label to the self-attention localization (incompatible with `--support_prompt_type box`)
- `support_only`: isolates the contribution of the support visual tokens; uses `"visual"` as the text sentinel (no class label) — SAM3 still receives the text token, just not the actual class name

**`--support_prompt_type box` ablation purpose:**
Tests whether a holistic ROI-pooled box token (capturing the whole object region) produces a more focused self-attention prior than a set of scattered point tokens.

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
    --attn_layers                 last (default) or all
    --sampling_inputs             controls fusion encoder inputs in sampling pass
    --attention_aggregate_function controls aggregation of attention matrices
    --attn_sampling_mode          top-k (default) or bottom-k
    --inject_text_pooling         master on/off (see Text Pooling Injection table)
    --injection_text_pooling_stage             point_sampling | inference_pass | both
    --injection_text_pooling_in_prompts_sampling   bias visual prompt tokens during sampling pass
    --injection_text_pooling_in_prompts_inference  bias aggregated visual prompt during inference

--experiment_mode dense_cross_attn
    requires: --nshot > 0
    --inject_text_pooling         master on/off (see Text Pooling Injection table)
    --injection_text_pooling_stage             point_sampling | inference_pass | both
    --injection_text_pooling_in_prompts_sampling   bias dense support K/V during sampling pass
    --injection_text_pooling_in_prompts_inference  bias aggregated visual prompt during inference
    --attn_layers                 last (default) or all
    --attention_aggregate_function controls aggregation of attention matrices
    --attn_sampling_mode          top-k (default) or bottom-k

--experiment_mode self_attn
    --attn_layers                 last (default) or all
    --sampling_inputs             controls fusion encoder inputs in sampling pass
    --sampling_inputs support_only requires --nshot > 0
    --attention_aggregate_function controls aggregation of attention matrices
    --attn_sampling_mode          top-k or bottom-k (default)
    --support_prompt_type         points (default) or box; incompatible with sampling_inputs=text_only
    --blob_selection              largest (default) or smallest; only meaningful with support_prompt_type=box
    --inject_text_pooling         master on/off (see Text Pooling Injection table)
    --injection_text_pooling_stage             point_sampling | inference_pass | both
    --injection_text_pooling_in_prompts_sampling   bias visual prompt tokens during sampling pass
    --injection_text_pooling_in_prompts_inference  bias aggregated visual prompt during inference

--experiment_mode attn_prior
    --support_prompt_type         points (default) or box; incompatible with sampling_inputs=text_only
    --blob_selection              largest (default) or smallest; only meaningful with support_prompt_type=box

--experiment_mode random
    --support_prompt_type         points (default) or box
    --blob_selection              largest (default) or smallest; only meaningful with support_prompt_type=box
    --sample_points_from_image    when support_prompt_type=points: sample from the full image canvas instead of GT mask

--sampling_inputs                 applies only to --experiment_mode attn_prior or self_attn
--attn_layers                     affects point selection for Exp 5/6/7 and attention map saving for all experiments
--attention_aggregate_function    applies to Exp 5/6/7 for attention map aggregation
--attn_sampling_mode              applies to Exp 5/6/7 for point selection from attention maps
--blob_selection                  applies whenever --support_prompt_type box is active
```

---

## Quick Reference: Experiment Flag Combinations

| Experiment | Key flags |
|-----------|-----------|
| 1 — random support | `--experiment_mode random --nshot N` |
| 2a — self-support (GT-mask points) | `--experiment_mode random --nshot 1 --use_query_as_support` |
| 2b — self-support (random image points) | `--experiment_mode random --nshot 1 --use_query_as_support --sample_points_from_image` |
| 3a — matcher + backbone | `--experiment_mode matcher --nshot N` |
| 3b — matcher + fusion features | `--experiment_mode matcher --nshot N --use_fused_matcher_features` |
| 3b-no-label — matcher + fusion features + no class label | `--experiment_mode matcher --nshot N --use_fused_matcher_features --sampling_inputs support_only --disable_text_inference` |
| 3c — matcher + structured sampling | `--experiment_mode matcher --nshot N [--use_fused_matcher_features] --sampling k-medoids-points` |
| 4 — query self-matching (failed) | `--experiment_mode self_matching --nshot N` |
| 5a — attention prior topk | `--experiment_mode attn_prior --nshot N --attn_sampling_mode top-k --attn_layers last` |
| 5b — attention prior bottomk | `--experiment_mode attn_prior --nshot N --attn_sampling_mode bottom-k --attn_layers last` |
| 5c — attention prior, text only sampling | `--experiment_mode attn_prior --nshot N --sampling_inputs text_only --attn_layers last` |
| 6a — dense cross-attn topk | `--experiment_mode dense_cross_attn --nshot N --attn_sampling_mode top-k --attn_layers all` |
| 6b — dense cross-attn bottomk | `--experiment_mode dense_cross_attn --nshot N --attn_sampling_mode bottom-k --attn_layers all` |
| 6a+text — dense cross-attn topk + text pooling (image patches only, sampling) | `--experiment_mode dense_cross_attn --nshot N --attn_sampling_mode top-k --attn_layers all --inject_text_pooling --injection_text_pooling_stage point_sampling` |
| 6a+text-prompts-sampling — dense cross-attn topk + text pooling on image+support K/V | `--experiment_mode dense_cross_attn --nshot N --attn_sampling_mode top-k --attn_layers all --inject_text_pooling --injection_text_pooling_stage point_sampling --injection_text_pooling_in_prompts_sampling` |
| 6b+text — dense cross-attn bottomk + text pooling | `--experiment_mode dense_cross_attn --nshot N --attn_sampling_mode bottom-k --attn_layers all --inject_text_pooling --injection_text_pooling_stage point_sampling` |
| 1+text-inf — random + text pooling on inference (image patches only) | `--experiment_mode random --nshot N --inject_text_pooling --injection_text_pooling_stage inference_pass` |
| 1+text-inf-prompts — random + text pooling on inference (image + aggregated prompt) | `--experiment_mode random --nshot N --inject_text_pooling --injection_text_pooling_stage inference_pass --injection_text_pooling_in_prompts_inference` |
| 7+text-both-mixed — self-attn + text pooling sampling-prompts only, inference image-only | `--experiment_mode self_attn --nshot N --attn_sampling_mode bottom-k --inject_text_pooling --injection_text_pooling_stage both --injection_text_pooling_in_prompts_sampling` |
| 7a — self-attn bottom-k (both) | `--experiment_mode self_attn --nshot N --attn_sampling_mode bottom-k --attn_layers last` |
| 7b — self-attn bottom-k, text only | `--experiment_mode self_attn --nshot N --sampling_inputs text_only --attn_layers last` |
| 7c — self-attn bottom-k, support only | `--experiment_mode self_attn --nshot N --sampling_inputs support_only --attn_layers last` |
| 7c-no-label — self-attn, no class label in both passes | `--experiment_mode self_attn --nshot N --sampling_inputs support_only --disable_text_inference --attn_layers last` |
| 5c-no-label — attn prior, no class label in both passes | `--experiment_mode attn_prior --nshot N --sampling_inputs support_only --disable_text_inference --attn_layers last` |
| 1-no-label — random support, no class label at inference | `--experiment_mode random --nshot N --disable_text_inference` |
| 1-box — random + box support | `--experiment_mode random --nshot N --support_prompt_type box` |
| 1-box-small — random + smallest blob | `--experiment_mode random --nshot N --support_prompt_type box --blob_selection smallest` |
| 5-box — attn prior + box support | `--experiment_mode attn_prior --nshot N --attn_sampling_mode top-k --attn_layers last --support_prompt_type box` |
| 5-box-small — attn prior + smallest blob | `--experiment_mode attn_prior --nshot N --attn_sampling_mode top-k --attn_layers last --support_prompt_type box --blob_selection smallest` |
| 7-box — self-attn + box support (both) | `--experiment_mode self_attn --nshot N --attn_sampling_mode bottom-k --attn_layers last --sampling_inputs both --support_prompt_type box` |
| 7-box-small — self-attn + smallest blob (both) | `--experiment_mode self_attn --nshot N --attn_sampling_mode bottom-k --attn_layers last --sampling_inputs both --support_prompt_type box --blob_selection smallest` |
| 7-box-all — self-attn + box support (all layers) | `--experiment_mode self_attn --nshot N --attn_sampling_mode bottom-k --attn_layers all --sampling_inputs both --support_prompt_type box` |
| 7-box-all-small — self-attn + smallest blob (all layers) | `--experiment_mode self_attn --nshot N --attn_sampling_mode bottom-k --attn_layers all --sampling_inputs both --support_prompt_type box --blob_selection smallest` |
