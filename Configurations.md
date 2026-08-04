# Configurations — `test_SAM3_CROSS_IMAGE.py`

This file documents every flag, every experiment, and the inference flow of the SAM3 cross-image few-shot segmentation pipeline.

> **Naming.** Experiments are referred to by descriptive names throughout this document. The `--experiment_mode` values in the code are unchanged and are given next to every name, so every command in this file is runnable as-is.

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
- **What prompt tokens** reach the Fusion Encoder (random support points / matched points / cross-attention prior points / dense support cross-attention points / self-attention prior points)
- **What feature volumes** are used for matching (backbone vs. fusion encoder output)
- **How many final points** reach the decoder and how they are selected (`--sampling`)
- **Which attention operator supplies the localization prior** — cross-attention to prompt tokens (Exp 5), the self-attention slot rewired to attend to dense support patches (Exp 6), or plain query self-attention (Exp 7)

---

## Experiment Index

The three attention-based experiments differ in exactly one respect: **which attention map the point prior is read from, and what the keys/values are.**

| # | Name | `--experiment_mode` | Prior read from | Q | K/V |
|---|------|---------------------|-----------------|---|-----|
| 1 | Random Support Points (baseline) | `random` | — points drawn from the support GT mask | — | — |
| 2 | Self-Support Oracle | `random` + `--use_query_as_support` | — points drawn from the query's own GT mask | — | — |
| 3 | Bipartite Patch Matching | `matcher` | cosine similarity between two patch feature volumes | — | — |
| 4 | Query Self-Matching *(deprecated)* | `self_matching` | cosine similarity between two fusions of the same query | — | — |
| 5 | **Prompt Cross-Attention Prior** | `attn_prior` | **cross-attention** | query patches | prompt tokens (text + support visual) — the *visual* columns are the decision signal |
| 6 | **Dense Support Cross-Attention Prior** | `dense_cross_attn` | **self-attention slot, run as cross-attention** | query patches | support **foreground patches** (dense, backbone-level) |
| 7 | **Query Self-Attention Prior** | `self_attn` | **self-attention** | query patches | query patches |

> [!IMPORTANT]
> In Experiment 6 the keys/values are the support images' **foreground patches taken from the backbone feature map** (`sup_feats_spatial`, `MatcherBoxCalculator.get_dense_cross_attn_map`), *not* the encoded geometric prompt tokens. The prompt tokens continue to feed the encoder's cross-attention layers, which Experiment 6 leaves untouched — only the self-attention operator is substituted.

Sub-variants:

| Label | Name | Label | Name |
|-------|------|-------|------|
| 2a | Self-Support Oracle (GT-mask points) | 5b | Valley-Attention Selection (`bottom-k`) |
| 2b | Uniform Query Points | 5c | Text-Only Prior |
| 3a | Backbone-Feature Matching | 6a | Peak Selection |
| 3b | Fusion-Feature Matching | 6b | Valley Selection |
| 3c | Structured Subsampling | 7a | Full-Context Prior |
| 5a | Peak-Attention Selection (`top-k`) | 7b | Text-Only Prior |
| | | 7c | Support-Only Prior |

---

## All Flags

### Infrastructure

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--checkpoint` | str | None | Path to SAM3 model checkpoint |
| `--benchmark` | str | `youtube-fsvos` ⚠️ | Dataset benchmark to evaluate on. Choices: `youtube_fsvos`, `minivspw`, `coco`, `lvis`, `ade20k`, `pascal`, `coco-20i`, `pascal-5i`, `lvis-92i`. **The default is not a member of its own choices list** (`youtube-fsvos` with a hyphen vs. `youtube_fsvos` with an underscore). argparse does not validate defaults, so omitting the flag passes the invalid string through to `ImageDataset.build_dataset` and fails — treat `--benchmark` as mandatory |
| `--session_name` | str | None | Name for this run (used in output paths and logs). Auto-suffixed with `_shard{i}of{N}` when sharding is active |
| `--dataset_path` | str | None | Root path to the dataset |
| `--data_list_path` | str | None | Path to the data list / split file (MiniVSPW only) |
| `--output_dir` | str | `./output` | Directory for output masks, frames, bounding boxes and attention maps |
| `--fold` | int | 1 | Cross-validation fold (used by COCO-20i, Pascal-5i, LVIS-92i, MiniVSPW) |
| `--frame_num` | int | 1 | Number of query frames to evaluate per episode |
| `--nshot` | int | 1 | Number of support images per episode (N-shot) |
| `--run_n` | int | 0 | Run index (for repeated experiments with different seeds) |
| `--seed` | int | 0 | Random seed for reproducibility. The per-item seed is `seed + run_n * 10000 + idx`, set *before* data loading so support-image sampling is identical across experiments |
| `--log_dir` | str | `/megaverse/storage/samele/FSS-SAM3/experiment_results_logs` | Root directory for the result CSVs |
| `--num_shards` | int | 1 | Number of parallel shards to split the class list into. `1` = no sharding. See [Sharded execution](#sharded-execution) |
| `--shard_id` | int | 0 | Zero-based index of this shard. Range-checked **only when `--num_shards > 1`** |

### Class Label Handling

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--use_synset_names` | bool | False | Use WordNet synset names as text labels instead of raw class names. Replaces `idx_to_classname[idx]` with the mapping's `selected_lemma` and populates `class_idx_to_all_lemmas[idx]`. `idx_to_ground_truth_label` always retains the original dataset name for CSV cross-referencing |
| `--synset_mapping_folder_path` | str | `/megaverse/storage/samele/FSS-SAM3/datasets/synset_mappings/leaf` | Directory holding the synset mapping **CSV** files — one per benchmark, named `{benchmark}.csv`, `sep="\|"`, columns `idx` / `selected_lemma` / `lemmas`. The few-shot variants reuse the base dataset's mapping (`coco-20i`→`coco`, `lvis-92i`→`lvis`, `pascal-5i`→`pascal`) |
| `--use_grouping_ade20k` | bool | False | Group ADE20K classes using `ADE20K_grouping.json`, loaded from the *same directory* as the synset mapping CSV. Several raw ADE20K pixel labels collapse into one grouped class. **Takes precedence over `--use_synset_names`** when both are set. Silent no-op on every other benchmark |
| `--all_lemmas` | bool | False | Evaluate every WordNet lemma of a class as a separate virtual class, instead of just the canonical one. See [All-lemmas evaluation protocol](#all-lemmas-evaluation-protocol) |
| `--disable_text_inference` | bool | False | Replace the real class label with the `"visual"` sentinel token **at inference time only**. SAM3 still receives a text token (`"visual"`) but no custom class label. To suppress the class label in the sampling pass too, combine with `--sampling_inputs support_only` (Exp 5 and 7 only) — see [Suppressing the class label in both passes](#suppressing-the-class-label-in-both-passes) |

### Point Prompt Controls

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--num_points_from_mask` | int | 20 | Number of points to sample / select as geometric prompts for the decoder. Despite the name, for Exp 5/6/7 the points come from an attention map, not from a mask |
| `--skip_coords` | bool | False | Skip spatial coordinate embeddings when encoding point prompts. When False, the exemplar encoder receives both appearance and position information; when True, position is dropped. **Scope:** support-side encoding only — see the note below |
| `--use_query_as_support` | bool | False | Use the query image itself as the support image (1-shot self-support). Incompatible with `matcher` and `self_matching` |
| `--sample_points_from_image` | bool | False | With `--experiment_mode random` and `--support_prompt_type points`: sample the point prompts **uniformly from the entire image canvas** instead of only from the foreground mask region. Applies to Exp 1 (random points over the full support image) and Exp 2b (random points over the full query image, via `--use_query_as_support`). Silent no-op for `box`, and for every other experiment mode |
| `--fix_sampled_points` | bool | False | Freeze the first lemma's sampled points and replay them for every subsequent lemma of the same sample. Only meaningful with `--all_lemmas` — see [All-lemmas evaluation protocol](#all-lemmas-evaluation-protocol) |

> [!NOTE]
> **`--skip_coords` does not affect the final query-side prompt.** The sampled query points are always encoded with `skip_coords=True`, hardcoded in all five point-selection branches of `get_prompt_tokens_from_support` (prior experiments showed that keeping coordinates is markedly worse when the points already live on the query image). The flag therefore governs only:
> - the support-side encoding in Exp 1 / 2 (points *and* boxes),
> - the support-token encoding of the Exp 5/6/7 sampling pass,
> - the matcher's `compute_box` feature extraction,
> - the dummy geometric prompt built inside `get_fused_image_features` when `--sampling_inputs text_only` (the only path on which `skip_coords` reaches that function's body).

### Support Prompt Type

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--support_prompt_type` | str | `points` | Type of visual prompt built from support images. `points`: random points sampled from the GT mask (default). `box`: bounding box of a connected foreground blob (see `--blob_selection`), encoded via the geometry encoder's ROI-align path. Applies to `random` (direct support encoding) and `attn_prior` / `self_attn` (sampling pass) |
| `--blob_selection` | str | `largest` | Which connected foreground blob to use when `--support_prompt_type box`. `largest` (default): biggest blob by pixel area. `smallest`: smallest blob with area ≥ 10 px (blobs smaller than 10 px are treated as noise and ignored; falls back to the absolute smallest if none qualify) |

**`--support_prompt_type` choices:**

| Value | Description |
|-------|-------------|
| `points` | Random points from support GT mask — scattered spatial tokens, each capturing a point's appearance and (unless `--skip_coords`) position |
| `box` | Bounding box of a connected blob (largest or smallest, controlled by `--blob_selection`). The geometry encoder encodes the box as a single token (or two corner tokens if `encode_boxes_as_points=True`) using ROI-align pooling to aggregate visual content within the box region. When `--skip_coords` is False, coordinate embeddings are also added; when True, only the ROI-pooled visual content is encoded |

**Notes:**
- With `--support_prompt_type box`, connected-components is always run; `--blob_selection` then picks the target blob.
- `--blob_selection` is only meaningful when `--support_prompt_type box`; it has no effect for `points`.
- Incompatible with `--sampling_inputs text_only` (no visual tokens are built in that mode).
- Not applicable to `matcher`, `self_matching`, or `dense_cross_attn`.
- **No experiment uses box tokens in both passes.** For Exp 5/7 the boxes condition the *sampling* pass only — the tokens that reach inference are the query-side point tokens sampled from the resulting attention map. For Exp 1 there is no sampling pass, so the box tokens *are* the final prompt.

### Experiment Selector

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--experiment_mode` | str | `random` | Selects which experiment / point-selection strategy to run. Choices: `random`, `matcher`, `self_matching`, `attn_prior`, `dense_cross_attn`, `self_attn` |

**`--experiment_mode` choices:**

| Value | Experiment | Description |
|-------|-----------|-------------|
| `random` | Exp 1 / 2 — Random Support Points / Self-Support Oracle | Random points sampled from support GT masks (default baseline) |
| `matcher` | Exp 3 — Bipartite Patch Matching | Bipartite patch matching support→query to compute prompt points |
| `self_matching` | Exp 4 — Query Self-Matching | Query self-matching: match query features with/without support embeddings *(deprecated)* |
| `attn_prior` | Exp 5 — Prompt Cross-Attention Prior | Points from the Fusion Encoder **cross-attention** map (K/V = prompt tokens) |
| `dense_cross_attn` | Exp 6 — Dense Support Cross-Attention Prior | Points from the self-attention slot rewired as cross-attention (K/V = dense support foreground patches) |
| `self_attn` | Exp 7 — Query Self-Attention Prior | Points from the Fusion Encoder **self-attention** map (K/V = query patches) |

### Matcher Controls

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--use_fused_matcher_features` | bool | False | Use fusion encoder output features for matching instead of raw backbone features. Only relevant for `matcher`; for all other fusion-encoder experiments the fusion encoder is always used |
| `--sampling` | str | `random` | Subsampling strategy applied to the matcher's candidate points. Active for `matcher` **and `self_matching`** (Exp 4 routes through the same `_postprocess_matcher_output` helper). Ignored by every other experiment mode |

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
| `--attn_layers` | str | **`all`** | Which Fusion Encoder layers to aggregate attention from, for both point selection (Exp 5/6/7) and map saving (all experiments). `last` = final layer only (most semantic); `all` = mean over all 6 layers. Also used as the filename suffix of the saved heatmaps |
| `--attention_aggregate_function` | str | `sum` | Aggregation function used to reduce attention matrices along the key dimension. Choices: `sum`, `mean`, `max`, `min`, `top-*-mean` (e.g. `top-5-mean`; `k` is clamped to the key count). An unrecognised value silently falls back to `sum` |
| `--attn_sampling_mode` | str | **`bottom-k`** | Point sampling method for attention priors. Choices: `top-k`, `bottom-k` |

### Text Pooling Injection (all experiments)

The pooled-text bias is the projected mean-pooled text embedding (`text_pooling_proj(pool_text_feat(...))`), shape `[1, 256]`. It mirrors the text conditioning applied during SAM3 training. Injection is controlled by one master switch plus three sub-flags:

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--inject_text_pooling` | bool | False | Master on/off. When False, all sub-flags are no-ops. When True, the bias is added to query image patches at the stages selected by `--injection_text_pooling_stage`; support-side targets are only biased when the corresponding `in_prompts_*` flag is set |
| `--injection_text_pooling_stage` | str | `point_sampling` | Which forward pass(es) receive the bias. Choices: `point_sampling` (Fusion Encoder pass that extracts the attention map for Exp 5/6/7), `inference_pass` (final SAM3 inference forward), `both` |
| `--injection_text_pooling_in_prompts_sampling` | bool | False | During the point-sampling pass, also bias support-side tokens: visual prompt tokens for Exp 5/7, dense support spatial K/V for Exp 6. No effect on Exp 1/2/3/4 (no separate sampling pass) |
| `--injection_text_pooling_in_prompts_inference` | bool | False | During the inference forward, also bias the aggregated visual prompt fed to the encoder. Applies to all experiments. For Exp 1, this replaces the previous per-shot bias loop and is mathematically equivalent |

**Effect matrix (when master flag is on):**

| Target | Image patches (query) | Visual prompt tokens (Exp 5/7) | Dense support K/V (Exp 6) | Aggregated visual prompt (inference, all Exp) |
|--------|-----------------------|--------------------------------|---------------------------|------------------------------------------------|
| Sampling pass | `stage ∈ {point_sampling, both}` | `stage ∈ {point_sampling, both} AND in_prompts_sampling` | `stage ∈ {point_sampling, both} AND in_prompts_sampling` | — |
| Inference pass | `stage ∈ {inference_pass, both}` | — | — | `stage ∈ {inference_pass, both} AND in_prompts_inference` |

`pool_text_feat` is computed at most once per `cross_image_prediction` call regardless of how many stages or targets are biased.

> [!WARNING]
> **The two "pooled text" vectors are not the same quantity.** The *image-patch* injection is applied inside the encoder (`encoder.inject_text_pooling`), where the pooling runs over the **whole prompt sequence** — text tokens *plus* the concatenated visual/support tokens. The `in_prompts_*` bias uses `_compute_pooled_text`, which pools the **text encoder output only**. Whenever visual prompts are present the two differ, so `--injection_text_pooling_stage point_sampling` and `--injection_text_pooling_in_prompts_sampling` do not add the same vector to their respective targets.
>
> Additionally, under `--sampling_inputs support_only` the class label has already been overwritten with `"visual"` before the bias is computed, so the `in_prompts_*` bias is the pooled embedding of the literal string `"visual"`.

> [!NOTE]
> The argparse `--help` text for `--injection_text_pooling_in_prompts_sampling` claims it also biases "support prompt tokens for Exp 1". It does not: the sampling-pass visual prompt is only built for `attn_prior` and `self_attn`, so the flag is a no-op for `random`. Exp 1 can only be reached by the *inference* variant. The table above is authoritative; the help string is stale.

### Sampling Pass Input Control (Experiments 5 and 7)

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--sampling_inputs` | str | `both` | Controls what goes into the Fusion Encoder during the **point-selection forward pass** (the dedicated pass that extracts the attention map used for point selection). Only relevant for `attn_prior` and `self_attn`. `support_only` also applies to `matcher`: it replaces the class label with the `"visual"` sentinel in the fusion feature extraction pass. The final SAM3 inference pass is always run with text + support visual tokens |

**`--sampling_inputs` choices:**

| Value | Description |
|-------|-------------|
| `both` | Text label + support visual prompt tokens (default — same as inference pass). For `matcher`: class label is used in the fusion feature extraction pass |
| `text_only` | Text label only; no support visual tokens in the sampling pass. Only valid for `attn_prior`/`self_attn` |
| `support_only` | Support visual tokens + `"visual"` sentinel token (class label suppressed). For `attn_prior`/`self_attn`: also excludes the class label from the fusion sampling pass. For `matcher`: replaces the class label with `"visual"` in the fusion feature extraction pass (`--use_fused_matcher_features` only) |

This flag is an ablation tool to isolate which input signal is responsible for the attention map's localization quality. Requires `--nshot > 0` when using `support_only`.

Note that `support_only` suppresses the *class label*, not the text token: SAM3 always receives a text input, and the `"visual"` sentinel is still concatenated to the prompt sequence in the sampling pass.

#### Suppressing the class label in both passes

To run a fully class-label-free experiment on Exp 5/7 — where SAM3 uses only the `"visual"` sentinel (no custom text label) in **both** the sampling pass and the inference pass — combine both flags:

```
--sampling_inputs support_only   # sampling pass: "visual" sentinel + support visual tokens
--disable_text_inference         # inference pass: "visual" sentinel + aggregated visual prompt
```

The `"visual"` token is always present (SAM3 requires a text input for architectural reasons), but no custom class label is used anywhere in the pipeline.

### Attention Map Saves

Attention maps are written per frame under:

```
<output_dir>/<dir_name>_<eval_id>_<idx>/attention_maps/
    ├── sampling_maps/     # only when the experiment runs a dedicated sampling pass
    ├── inference_maps/    # always written
    └── frame_{tag}_cross_points_{attn_layers}_sampled.png
```

- **`sampling_maps/`** captures the *decision* pass — the forward run whose attention map is used to select the prompt points. Written only when `has_dedicated_pass` is true: Exp 5, Exp 6, Exp 7, and Exp 3b (`matcher --use_fused_matcher_features`).
- **`inference_maps/`** captures the final SAM3 inference forward and is written for every experiment. For Exp 1/2/3a/4 this is the only directory with content.
- The `_sampled.png` overlay stays in the parent `attention_maps/` directory.

Each of the two directories receives:

| Filename | Content |
|----------|---------|
| `frame_{tag}_cross_total_{attn_layers}.png` | Cross-attn sum over **all** prompt token columns (text + visual) |
| `frame_{tag}_cross_text_{attn_layers}.png` | Cross-attn sum over **text** token columns only |
| `frame_{tag}_cross_points_{attn_layers}.png` | Cross-attn sum over **visual/point** token columns only — decision signal in Exp 5 |
| `frame_{tag}_self_{attn_layers}.png` | Self-attn row-sum (standard Q=K=V=query patches, or dense support cross-attn in Exp 6) — decision signal in Exp 7 |
| `frame_{tag}_layer_{n}_{self\|cross\|cross_text\|cross_points}.png` | The same four maps **per encoder layer**, before the across-layer mean. Only populated for the layers selected by `--attn_layers` |

All maps use **pre-softmax** weights. `{attn_layers}` in the filename is the literal value of `--attn_layers` (`last` or `all`).

**What each map means per experiment:**

| Map | Exp 1 / 2 / 3 / 4 | Exp 5 — Prompt Cross-Attn Prior | Exp 6 — Dense Support Cross-Attn Prior | Exp 7 — Query Self-Attn Prior |
|-----|--------------------|-------------------------|--------------------------|---------------------------|
| `cross_total` | Query patches → all prompt tokens | Query patches → all prompt tokens | Query patches → all prompt tokens | Query patches → all prompt tokens |
| `cross_text` | Query patches → text tokens only | Query patches → text tokens only | Query patches → text tokens only | Query patches → text tokens only |
| `cross_points` | Query patches → visual/point tokens | **Decision signal** — top-k/bottom-k of this map selects the prompt points | Diagnostic only | Diagnostic only |
| `self` | Standard query self-attention | Standard query self-attention | **Decision signal** — dense cross-attn: Q=query patches, K/V=support foreground patches | **Decision signal** — row-sum of self-attn; bottom-k selects the prompt points |
| `cross_points_sampled` | Support mask points overlaid on `cross_points` | Points selected from `cross_points` overlaid on the map | Points selected from `self` overlaid on `cross_points` | Points selected from `self` overlaid on the map |

**Capture pass by experiment:**
- Exp 1 / 2 / 3a / 4: flags armed before the SAM3 inference forward pass; no extra encoder run. Only `inference_maps/` is populated
- Exp 3b: sampling maps captured inside `compute_box`; both directories populated
- Exp 5 / 7: sampling maps captured inside `get_fused_image_features` (the decision pass — no extra forward pass); both directories populated
- Exp 6: sampling maps captured inside `get_dense_cross_attn_map` (the decision pass — no extra forward pass); both directories populated

When `--sampling_inputs text_only` is used, no visual tokens exist in the sampling pass, so point selection falls back to the **text** cross-attention map instead of the points map.

### Debug / Visualization

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--visualize_embeddings` | bool | False | Save t-SNE plots of patch embeddings and matched points. Effective only for `matcher` — the output path is only constructed for that mode, and the plot needs ≥ 30 candidate points |

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

Checked later, in `main()` rather than `validate_args()`:

| Rule |
|------|
| `0 <= --shard_id < --num_shards` — **only enforced when `--num_shards > 1`**. `--num_shards 1 --shard_id 7` passes silently and simply disables sharding |

---

## Experiments

### Experiment 1 — Random Support Points (baseline) · `--experiment_mode random`

**Flags:**
```
--experiment_mode random
--nshot N
[--support_prompt_type points|box]
[--blob_selection largest|smallest]
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
        run connected-components on GT mask → select blob per --blob_selection
        compute bounding box [cx, cy, w, h] (normalized [0,1])
        encode box via SAM3 geometry encoder (ROI-align) → 1 token per support image
        (--skip_coords: use ROI-align content only, drop coordinate embeddings)
Aggregate all support tokens → visual prompt
Combine with text label tokens (or "visual" sentinel if --disable_text_inference)
    → final prompt fed to Fusion Encoder on query image
Fusion Encoder: query patches cross-attend to final prompt
SAM3 decoder → mask
```

**Notes:** The visual prompt encodes *where* and *what* is in the support. `--skip_coords` tests whether removing spatial information hurts (it typically does, because the exemplar encoder partially encodes object appearance via position). With `--support_prompt_type box`, a single ROI-pooled token represents the whole object region, collapsing N scattered point tokens into one holistic representation. `--sample_points_from_image` replaces the foreground-constrained sampling with uniform image-wide sampling — an ablation for Exp 1 (what if we give random positions on the support image?) and the main variant for Exp 2b.

---

### Experiment 2 — Self-Support Oracle · `--experiment_mode random --use_query_as_support`

Two variants depending on where the point prompts are sampled from.

#### Variant 2a — Self-Support Oracle (GT-mask points)

**Flags:**
```
--experiment_mode random
--nshot 1
--use_query_as_support
[--disable_text_inference]
[--skip_coords]
```

**Flow:** Identical to Experiment 1, but the support image is replaced by the query image itself. The GT mask of the query is used to sample the support points. This tests what the model can do when it "sees" the answer — an upper-bound sanity check.

#### Variant 2b — Uniform Query Points

**Flags:**
```
--experiment_mode random
--nshot 1
--use_query_as_support
--sample_points_from_image
[--disable_text_inference]
[--skip_coords]
```

**Flow:** Same substitution as Variant 2a (query image replaces the support), but the point prompts are sampled **uniformly from the entire query image canvas** — no foreground mask constraint. The model receives `--num_points_from_mask` random point hints with no localization bias, plus the text label. Variant 2b tests a purely text-guided prompt regime, measuring what information the geometric encoder can contribute from randomly placed, semantically uninformed points.

**Implementation note:** `--sample_points_from_image` replaces the foreground mask with an all-ones mask before calling `get_random_points_from_mask`, so `np.where` returns all pixel positions and `np.random.choice` samples from the full image. Coordinates are normalized by the **mask's own H×W** inside that helper, so the result is resolution-independent — the dataset transform resizes to 518 px, while the 1008 × 1008 / 72 × 72 / patch-14 grid applies only to attention-derived points (`matcher_calculator.resolution`).

---

### Experiment 3a — Backbone-Feature Matching · `--experiment_mode matcher`

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

### Experiment 3b — Fusion-Feature Matching · `--experiment_mode matcher --use_fused_matcher_features`

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

This is the only matcher variant that has a dedicated pass, so it is also the only one that populates `sampling_maps/`.

---

### Experiment 3c — Structured Subsampling · `--experiment_mode matcher --sampling <method>`

**Flags:**
```
--experiment_mode matcher
--nshot N
[--use_fused_matcher_features]
--sampling k-medoids-points   # or any non-random choice
```

**Flow:** Same as 3a/3b, but the final point selection from matcher candidates uses a structured method (k-medoids, k-means, patch-core, top-k). These strategies improve spatial diversity or feature coverage compared to random sampling.

---

### Experiment 4 — Query Self-Matching *(failed / deprecated)* · `--experiment_mode self_matching`

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

### Experiment 5 — Prompt Cross-Attention Prior · `--experiment_mode attn_prior`

**Flags:**
```
--experiment_mode attn_prior
--nshot N
--attn_layers all | last
--attention_aggregate_function sum | mean | max | min | top-*-mean
--attn_sampling_mode top-k | bottom-k
[--sampling_inputs both | text_only | support_only]
[--support_prompt_type points|box]
[--blob_selection largest|smallest]
[--inject_text_pooling ...]
```

**Motivation:** Instead of bipartite matching, use the **cross-attention** weights of the Fusion Encoder as a localization prior. Cross-attention runs with **Q = query image patches** and **K/V = prompt tokens** (text + support visual embeddings). Patches with high attention weight are likely to contain the object referred to by the prompts.

**Attention map construction:**
```
For each target layer (last or all 6) — armed via capture_cross_attn_weights=True:
    cross_attn_image fires return_pre_softmax=True, average_attn_weights=True
    pre-softmax logits → [1, 5184, seq_prompt]
    Columns 0..num_text_tokens-1 = text tokens; columns num_text_tokens.. = visual/point tokens
    Reduce along the key dimension with --attention_aggregate_function
    Split: text_map = reduce over text columns; points_map = reduce over visual columns
Mean over target layers → [5184] → reshape [72, 72] heatmap per split

Point selection uses last_cross_attn_points_map (visual token prior only).
With --sampling_inputs text_only there are no visual columns, so selection
falls back to last_cross_attn_text_map.
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
Encode selected points on query image → visual prompt (always skip_coords=True)
Combine with text tokens → Fusion Encoder on query → mask
```

Does NOT use the bipartite matcher. Does NOT go through `--sampling` (the attention map ranking is the final selection).

---

### Experiment 6 — Dense Support Cross-Attention Prior · `--experiment_mode dense_cross_attn`

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

**Motivation:** In the standard pipeline, the query image only sees the support object through a compressed visual prompt in the Fusion Encoder's cross-attention. Experiment 6 replaces the Fusion Encoder's **self-attention** with a dense cross-attention in which the query image patches directly attend to the support object's foreground patches at patch resolution, enabling explicit patch-level correspondence.

**Why the keys are patches, not prompt tokens:** the substitution happens in the *self-attention* slot, whose K/V would normally be the query patches themselves. They are replaced by the support images' foreground patches taken straight from the backbone feature map. The encoder's cross-attention layers are untouched and still attend to the text + support visual prompt tokens.

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
    reduce over support patches via --attention_aggregate_function → [5184]
Mean over target layers → [5184] → reshape [72, 72] heatmap
```

**Flow:**

```
Aggregate support visual tokens (random points from GT masks) → conditions the untouched cross-attn layers
Run modified Fusion Encoder pass with dense_support_feats active
    → capture pre-softmax self-attn logits from target layers
Construct heatmap [72, 72]
Sample patches based on --attn_sampling_mode (top-k or bottom-k, k=--num_points_from_mask) → patch indices → pixel centers → normalized coords
Encode selected points on query image → visual prompt (always skip_coords=True)
Combine with text tokens → standard Fusion Encoder on query → mask
```

Does NOT use the bipartite matcher. The heatmap ranking is the final selection — `--sampling` is ignored.

**Implementation notes:**
- `dense_support_feats` is set as per-layer state on each `TransformerEncoderLayer` before the encoder forward pass; `capture_self_attn_weights=True` is set on target layers to record pre-softmax logits into `last_self_attn_weights`
- `return_pre_softmax=True` in `MultiheadAttentionWrapper.forward()` skips the softmax and returns raw scaled dot-products; `average_attn_weights=True` averages over heads → shape `[batch, tgt_len, src_len]`
- The cross-attention layers of the encoder (query→prompt) are unchanged; only self-attention is replaced
- Feature space note: at layers > 0, query tokens have been transformed by prior cross-attention, while support K/V remain at backbone level — a progressive mismatch that is acknowledged and evaluated empirically
- `--support_prompt_type box` does not apply: the support visual prompt for this mode is always built from random mask points, and `validate_args()` rejects `box` for `dense_cross_attn`
- `--injection_text_pooling_in_prompts_sampling` biases the **dense support spatial features** (`sup_feats_spatial`) here, not the cross-attention prompt tokens

---

### Experiment 7 — Query Self-Attention Prior · `--experiment_mode self_attn`

**Flags:**
```
--experiment_mode self_attn
--nshot N
--attn_layers all | last
--attention_aggregate_function sum | mean | max | min | top-*-mean
--attn_sampling_mode bottom-k | top-k
[--sampling_inputs both | text_only | support_only]
[--support_prompt_type points|box]
[--blob_selection largest|smallest]
[--inject_text_pooling ...]
```

**Motivation:** The Fusion Encoder's **self-attention** map (standard Q = K = V = query patches) empirically shows low row-sum values at the target object's location. Experiment 7 exploits this by selecting the k query patches with the *lowest* self-attention activation as prompt points, bypassing explicit cross-image matching entirely.

**Self-attention map construction:**
```
For each target layer (last or all 6) — armed via capture_self_attn_weights=True:
    self_attn(Q=K=V=query patches) fires return_pre_softmax=True, average_attn_weights=True
    pre-softmax logits → [1, 5184, 5184] (heads averaged)
    Row reduction: for each query patch i, reduce over j via --attention_aggregate_function → [5184]
Mean over target layers → [5184] → reshape [72, 72] heatmap
```

All weights are **pre-softmax** raw scaled dot-products averaged over heads. This produces the same heatmap as `matcher_calculator.last_self_attn_map` (saved as `frame_{tag}_self_{attn_layers}.png`).

**Flow:**
```
Resolve sampling-pass inputs:
    --sampling_inputs both:         run Fusion Encoder with text label + support visual tokens
    --sampling_inputs text_only:    run Fusion Encoder with text label only (no visual_prompt)
    --sampling_inputs support_only: run Fusion Encoder with support visual tokens + "visual" sentinel (class label suppressed)

    Support visual tokens built from (when sampling_inputs is both or support_only):
        --support_prompt_type points (default): random points from GT mask
        --support_prompt_type box:              bounding box of the selected blob (ROI-align token)

Run Fusion Encoder on query — capture self-attn weights
Construct self-attention heatmap [5184] from target layers
Sample patches based on --attn_sampling_mode (bottom-k by default, k=--num_points_from_mask)
    → patch indices → pixel centers → normalized coords
Convert patch index → pixel center: px = (idx % 72) * 14 + 7,  py = (idx // 72) * 14 + 7
Normalize: [px / 1008, py / 1008]

Encode selected points on query image → visual prompt (always skip_coords=True)
Combine with text tokens → Fusion Encoder on query → mask
```

Does NOT use the bipartite matcher or the cross-attention map. `--sampling` is not applied.

**`--sampling_inputs` ablation purpose:**
- `both` (7a, default): replicates the full prompt context used in inference; tests whether the self-attention map is driven by the combined signal
- `text_only` (7b): isolates the contribution of the text label to the self-attention localization (incompatible with `--support_prompt_type box`)
- `support_only` (7c): isolates the contribution of the support visual tokens; uses `"visual"` as the text sentinel (no class label) — SAM3 still receives the text token, just not the actual class name

**`--support_prompt_type box` ablation purpose:**
Tests whether a holistic ROI-pooled box token (capturing the whole object region) produces a more focused self-attention prior than a set of scattered point tokens. Note that the box only conditions the sampling pass — inference still receives the query-side point tokens selected from the resulting map.

---

## Evaluation Protocols

### All-lemmas evaluation protocol

`--all_lemmas` is handled **entirely in `test_SAM3_CROSS_IMAGE.py`**, not in the dataset classes. It expands each real class into one **virtual class id** per WordNet lemma and re-runs the whole prediction once per lemma.

```
For each real class id cid in the (possibly sharded) class list:
    lemmas = dataset.class_idx_to_all_lemmas.get(cid, [class_dic[cid]])
    for lemma in lemmas:
        assign a new virtual id; record virtual_to_original[vid] = cid
Evaluator is constructed on the virtual class list
```

Key properties:

- **Canonical lemma is forced first.** `lemma_entries` is sorted so the dataset's `selected_lemma` always occupies the lowest virtual id and the first iteration, guarded by an `assert`. This matters for Exp 5/6/7, where the text prompt determines the attention map and therefore the sampled point coordinates — a different ordering would produce different points. For Exp 1/3 the ordering is a practical no-op but is kept for reproducibility.
- **Missing lemma lists degrade gracefully** to a single-element list rather than raising.
- **CSV rows carry both identities**: `class_id` / `class_name` are the *virtual* ones, `original_class_idx` / `original_class_name` the real class, and `original_ground_truth_class_*` the untouched dataset label.
- Without `--use_synset_names`, most datasets leave `class_idx_to_all_lemmas` empty, so `--all_lemmas` degenerates to one lemma per class. ADE20K and the few-shot variants (COCO-20i / LVIS-92i / PASCAL-5i) always populate at least a single-element list.

#### `--fix_sampled_points`

Freezes the first lemma's sampled points and replays them for every subsequent lemma of the same sample. The point of the flag is to isolate the effect of the **label** from the effect of the **points**: without it, an attention-based experiment resamples different points for every lemma, so a per-lemma IoU difference conflates the two.

- **Cache scope:** `first_pts` is reset once per **dataset item**, outside the lemma loop *and* outside the frame loop. It therefore freezes points across lemmas **and across frames** — invisible at the default `--frame_num 1`, but material for `--frame_num > 1` on the video benchmarks.
- **Honoured by:** Exp 1 (`points` only), Exp 5, Exp 6, Exp 7.
- **Silent no-op for:** Exp 3 (`matcher`), Exp 4 (`self_matching`), and Exp 1 with `--support_prompt_type box`. For Exp 3/4 the cache is still *captured* — it is simply never read back.

> [!WARNING]
> In Exp 1 with `--nshot > 1`, `norm_pts_sampled` is overwritten on every support shot, so the cache ends up holding the **last** shot's points. On every subsequent lemma that single normalised point set is replayed onto *all* shots — collapsing per-shot point diversity and potentially placing points outside the other shots' masks. Prefer `--nshot 1` when using `--fix_sampled_points` with Exp 1, or restrict the flag to the attention experiments, where the points genuinely live on the query image.

**Typical pairing:**

```
# FREE — points resampled per lemma
--all_lemmas --use_synset_names

# FIXED — same points for every lemma of a sample
--all_lemmas --use_synset_names --fix_sampled_points
```

### Sharded execution

`--num_shards` / `--shard_id` split one evaluation across N parallel jobs.

- Splitting is done on the **sorted class-id list**, not on the raw item index, so every image of a given class is evaluated by exactly one shard. Remainder classes are distributed to the lowest-indexed shards.
- Items outside the shard are still *loaded* before being skipped, so per-shard wall time still pays the dataset I/O for every item — the saving is in model forward passes, not in data loading.
- **Seeding is shard-invariant.** The per-item seed uses the global `idx`, so a class's episodes get identical seeds no matter which shard runs them; a sharded run reproduces an unsharded one.
- `--session_name` is auto-suffixed `_shard{i}of{N}` (or becomes `shard{i}of{N}` if unset), so each shard writes its own results directory under `<log_dir>/<benchmark>/`.
- The final printed `Mean IoU` covers that shard's classes only.

Each shard writes six CSVs (`sep=';'`) into its own directory:

| File | Granularity |
|------|-------------|
| `{benchmark}_size_scores.csv` | per class × size bucket (SMALL / MEDIUM / LARGE) |
| `{benchmark}_sample_scores.csv` | per evaluated sample |
| `{benchmark}_blob_scores.csv` | per connected blob within a sample |
| `{benchmark}_class_scores.csv` | per class (IoU + point accuracy, micro and macro) |
| `{benchmark}_box_sizes_scores.csv` | per matcher box (empty for non-matcher modes) |
| `{benchmark}_point_features.csv` | per (sample, lemma) — prompt-point geometry features |

> [!WARNING]
> **With `--all_lemmas`, virtual class ids are not comparable across shards.** Virtual ids are enumerated over the *sharded* class list, so shard 1's `vid=0` is a different lemma from shard 0's `vid=0`. Join on `original_class_idx` / `original_class_name`, or use [`merge_shards.py`](merge_shards.py), which renumbers ids globally and reproduces exactly what an unsharded run would have written.

```bash
python test_SAM3_CROSS_IMAGE.py ... --num_shards 4 --shard_id 0   # repeat for 1, 2, 3
python merge_shards.py --log_dir <log_dir> --benchmark ade20k --session_name <name> --auto_discover
```

---

## Encoder Layer Capture API

`TransformerEncoderLayer` exposes per-layer state attributes for extracting attention weights without extra forward passes:

| Attribute | Set before pass | Read after pass | Shape stored |
|-----------|----------------|-----------------|--------------|
| `capture_cross_attn_weights` | `True` | `last_cross_attn_weights` | `[batch, 5184, seq_prompt]` |
| `capture_self_attn_weights` | `True` | `last_self_attn_weights` | `[batch, 5184, src_len]` — `src_len` is 5184 (standard) or K support patches (Exp 6) |

Both use `return_pre_softmax=True, average_attn_weights=True` internally — raw scaled dot-products averaged over heads. Always clear both flag and storage after reading (set flag to `False`, storage to `None`) to avoid stale tensors on subsequent passes; `MatcherBoxCalculator.collect_inference_attn` does this for the inference path.

`capture_self_attn_weights` captures whichever keys are active during the forward pass it was armed for: standard self-attention (Q = K = V = query patches) or dense cross-attention to support foreground patches when `dense_support_feats` is injected (Exp 6). No extra forward pass is required for either case.

---

## Flag Dependency Map

```
--experiment_mode random                      (Exp 1 / 2 — Random Support Points / Self-Support Oracle)
    --support_prompt_type         points (default) or box
    --blob_selection              largest (default) or smallest; only meaningful with support_prompt_type=box
    --sample_points_from_image    when support_prompt_type=points: sample from the full image canvas instead of GT mask
    --use_query_as_support        makes this Exp 2
    --fix_sampled_points          honoured for points only; see the nshot>1 caveat

--experiment_mode matcher                     (Exp 3 — Bipartite Patch Matching)
    --use_fused_matcher_features  optional (backbone features used if False) → makes this Exp 3b
    --sampling                    active for all matcher sub-modes
    --sampling_inputs support_only replaces the class label with "visual" in the fusion feature pass (3b only)
    --attn_layers                 forwarded to compute_box
    --visualize_embeddings        t-SNE plots (this mode only)
    incompatible with: --use_query_as_support
    --fix_sampled_points          no-op

--experiment_mode self_matching               (Exp 4 — Query Self-Matching, deprecated)
    requires: --nshot > 0
    --sampling                    active
    incompatible with: --use_query_as_support
    --fix_sampled_points          no-op

--experiment_mode attn_prior                  (Exp 5 — Prompt Cross-Attention Prior)
    --attn_layers                 all (default) or last
    --attention_aggregate_function controls reduction along the key dimension
    --attn_sampling_mode          bottom-k (default) or top-k
    --sampling_inputs             controls fusion encoder inputs in sampling pass
    --sampling_inputs support_only requires --nshot > 0
    --support_prompt_type         points (default) or box; incompatible with sampling_inputs=text_only
    --blob_selection              largest (default) or smallest; only meaningful with support_prompt_type=box
    --inject_text_pooling         master on/off (see Text Pooling Injection table)
    --injection_text_pooling_stage             point_sampling | inference_pass | both
    --injection_text_pooling_in_prompts_sampling   bias visual prompt tokens during sampling pass
    --injection_text_pooling_in_prompts_inference  bias aggregated visual prompt during inference

--experiment_mode dense_cross_attn            (Exp 6 — Dense Support Cross-Attention Prior)
    requires: --nshot > 0
    --attn_layers                 all (default) or last
    --attention_aggregate_function controls reduction along the key dimension
    --attn_sampling_mode          bottom-k (default) or top-k
    --inject_text_pooling         master on/off (see Text Pooling Injection table)
    --injection_text_pooling_stage             point_sampling | inference_pass | both
    --injection_text_pooling_in_prompts_sampling   bias dense support K/V during sampling pass
    --injection_text_pooling_in_prompts_inference  bias aggregated visual prompt during inference
    --support_prompt_type         not applicable (box rejected by validate_args)

--experiment_mode self_attn                   (Exp 7 — Query Self-Attention Prior)
    --attn_layers                 all (default) or last
    --attention_aggregate_function controls reduction along the key dimension
    --attn_sampling_mode          bottom-k (default) or top-k
    --sampling_inputs             controls fusion encoder inputs in sampling pass
    --sampling_inputs support_only requires --nshot > 0
    --support_prompt_type         points (default) or box; incompatible with sampling_inputs=text_only
    --blob_selection              largest (default) or smallest; only meaningful with support_prompt_type=box
    --inject_text_pooling         master on/off (see Text Pooling Injection table)
    --injection_text_pooling_stage             point_sampling | inference_pass | both
    --injection_text_pooling_in_prompts_sampling   bias visual prompt tokens during sampling pass
    --injection_text_pooling_in_prompts_inference  bias aggregated visual prompt during inference

Cross-cutting
--sampling_inputs                 text_only: attn_prior/self_attn only; support_only: + matcher
--attn_layers                     affects point selection for Exp 5/6/7, attention map saving and
                                  filenames for all experiments
--attention_aggregate_function    applies to Exp 5/6/7 for attention map aggregation
--attn_sampling_mode              applies to Exp 5/6/7 for point selection from attention maps
--blob_selection                  applies whenever --support_prompt_type box is active
--skip_coords                     support-side encoding only; query-side points always use skip_coords=True
--all_lemmas                      re-runs every experiment once per lemma (virtual classes)
--fix_sampled_points              requires --all_lemmas to be meaningful
--num_shards / --shard_id         orthogonal to every experiment; merge with merge_shards.py
```

---

## Quick Reference: Experiment Flag Combinations

Flag sets assume the current defaults (`--attn_layers all`, `--attn_sampling_mode bottom-k`); layer and selection flags are still written out explicitly where they carry the meaning of the row.

### Baselines and matching

| Experiment | Key flags |
|-----------|-----------|
| 1 — Random Support Points | `--experiment_mode random --nshot N` |
| 1-no-label — no class label at inference | `--experiment_mode random --nshot N --disable_text_inference` |
| 1-box — box support prompt | `--experiment_mode random --nshot N --support_prompt_type box` |
| 1-box-small — smallest blob | `--experiment_mode random --nshot N --support_prompt_type box --blob_selection smallest` |
| 2a — Self-Support Oracle (GT-mask points) | `--experiment_mode random --nshot 1 --use_query_as_support` |
| 2b — Uniform Query Points | `--experiment_mode random --nshot 1 --use_query_as_support --sample_points_from_image` |
| 3a — Backbone-Feature Matching | `--experiment_mode matcher --nshot N` |
| 3b — Fusion-Feature Matching | `--experiment_mode matcher --nshot N --use_fused_matcher_features` |
| 3b-no-label — fusion features, no class label | `--experiment_mode matcher --nshot N --use_fused_matcher_features --sampling_inputs support_only --disable_text_inference` |
| 3c — Structured Subsampling | `--experiment_mode matcher --nshot N [--use_fused_matcher_features] --sampling k-medoids-points` |
| 4 — Query Self-Matching *(deprecated)* | `--experiment_mode self_matching --nshot N` |

### Exp 5 — Prompt Cross-Attention Prior

| Variant | Key flags |
|---------|-----------|
| 5a — Peak-Attention Selection | `--experiment_mode attn_prior --nshot N --attn_sampling_mode top-k --attn_layers all` |
| 5b — Valley-Attention Selection | `--experiment_mode attn_prior --nshot N --attn_sampling_mode bottom-k --attn_layers all` |
| 5c — Text-Only Prior | `--experiment_mode attn_prior --nshot N --sampling_inputs text_only --attn_layers all` |
| 5c-no-label — no class label in both passes | `--experiment_mode attn_prior --nshot N --sampling_inputs support_only --disable_text_inference --attn_layers all` |
| 5-box — box support prompt | `--experiment_mode attn_prior --nshot N --attn_sampling_mode top-k --attn_layers all --support_prompt_type box` |
| 5-box-small — smallest blob | `--experiment_mode attn_prior --nshot N --attn_sampling_mode top-k --attn_layers all --support_prompt_type box --blob_selection smallest` |
| 5a-last — layer ablation (final layer only) | `--experiment_mode attn_prior --nshot N --attn_sampling_mode top-k --attn_layers last` |

### Exp 6 — Dense Support Cross-Attention Prior

| Variant | Key flags |
|---------|-----------|
| 6a — Peak Selection | `--experiment_mode dense_cross_attn --nshot N --attn_sampling_mode top-k --attn_layers all` |
| 6b — Valley Selection | `--experiment_mode dense_cross_attn --nshot N --attn_sampling_mode bottom-k --attn_layers all` |
| 6a+text — text pooling, image patches only (sampling) | `--experiment_mode dense_cross_attn --nshot N --attn_sampling_mode top-k --attn_layers all --inject_text_pooling --injection_text_pooling_stage point_sampling` |
| 6a+text-prompts-sampling — image patches + dense support K/V | `--experiment_mode dense_cross_attn --nshot N --attn_sampling_mode top-k --attn_layers all --inject_text_pooling --injection_text_pooling_stage point_sampling --injection_text_pooling_in_prompts_sampling` |
| 6b+text — valley selection + text pooling | `--experiment_mode dense_cross_attn --nshot N --attn_sampling_mode bottom-k --attn_layers all --inject_text_pooling --injection_text_pooling_stage point_sampling` |
| 6a-last — layer ablation (final layer only) | `--experiment_mode dense_cross_attn --nshot N --attn_sampling_mode top-k --attn_layers last` |

### Exp 7 — Query Self-Attention Prior

| Variant | Key flags |
|---------|-----------|
| 7a — Full-Context Prior | `--experiment_mode self_attn --nshot N --attn_sampling_mode bottom-k --attn_layers all` |
| 7b — Text-Only Prior | `--experiment_mode self_attn --nshot N --sampling_inputs text_only --attn_layers all` |
| 7c — Support-Only Prior | `--experiment_mode self_attn --nshot N --sampling_inputs support_only --attn_layers all` |
| 7c-no-label — no class label in both passes | `--experiment_mode self_attn --nshot N --sampling_inputs support_only --disable_text_inference --attn_layers all` |
| 7-box — box support prompt | `--experiment_mode self_attn --nshot N --attn_sampling_mode bottom-k --attn_layers all --sampling_inputs both --support_prompt_type box` |
| 7-box-small — smallest blob | `--experiment_mode self_attn --nshot N --attn_sampling_mode bottom-k --attn_layers all --sampling_inputs both --support_prompt_type box --blob_selection smallest` |
| 7a-last — layer ablation (final layer only) | `--experiment_mode self_attn --nshot N --attn_sampling_mode bottom-k --attn_layers last` |
| 7-box-last — box support, final layer only | `--experiment_mode self_attn --nshot N --attn_sampling_mode bottom-k --attn_layers last --sampling_inputs both --support_prompt_type box` |
| 7+text-both-mixed — text pooling on sampling prompts, inference image-only | `--experiment_mode self_attn --nshot N --attn_sampling_mode bottom-k --inject_text_pooling --injection_text_pooling_stage both --injection_text_pooling_in_prompts_sampling` |

### Text pooling on the inference pass (any experiment)

| Variant | Key flags |
|---------|-----------|
| 1+text-inf — image patches only | `--experiment_mode random --nshot N --inject_text_pooling --injection_text_pooling_stage inference_pass` |
| 1+text-inf-prompts — image patches + aggregated prompt | `--experiment_mode random --nshot N --inject_text_pooling --injection_text_pooling_stage inference_pass --injection_text_pooling_in_prompts_inference` |

### Protocols (compose with any row above)

| Protocol | Key flags |
|----------|-----------|
| All lemmas, FREE points | `--all_lemmas --use_synset_names` |
| All lemmas, FIXED points | `--all_lemmas --use_synset_names --fix_sampled_points` |
| Sharded run (4 jobs) | `--num_shards 4 --shard_id {0,1,2,3}`, then `merge_shards.py --auto_discover` |
