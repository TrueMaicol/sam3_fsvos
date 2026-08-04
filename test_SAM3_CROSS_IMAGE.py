import argparse
import random
import json
from SAM3_IMAGE_TEXT import SAM3_IMAGE_PREDICTOR
from datasets.YoutubeFSVOS.YoutubeFSVOS_IMAGE import YTVOSDataset_Image
from datasets.MiniVSPW.nminivspw_dataset_IMAGE import NMiniVSPWEpisodicData_IMAGE
from datasets.YoutubeFSVOS.transform import TestTransform
from datasets.ImageDataset import ImageDataset
import torch 
import numpy as np
from PIL import Image, ImageDraw 
import cv2
import os
from utils.Evaluator import Evaluator
import time
import pandas as pd
from torch.utils.data import DataLoader
from MatcherBoxCalculator import MatcherBoxCalculator
from PatchCoreSampler import GreedyCoresetSampler
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn_extra.cluster import KMedoids
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
import matplotlib.pyplot as plt
import re
from scipy import ndimage as scipy_ndimage

from sam3.model.encoder import pool_text_feat

def agg_function_arg(value):
    fixed_values = {"sum", "mean", "max", "min"}
    if value in fixed_values:
        return value
    pattern = r"top-(\d+)-mean"
    match = re.fullmatch(pattern, value)
    if not match:
        raise argparse.ArgumentTypeError(
            f"Invalid input: '{value}'. Expected 'sum', 'mean', 'max', 'min', or 'top-*-mean'."
        )
    else:
        return value

def fix_randseed(seed):
    r""" Set random seeds for reproducibility """
    if seed is None:
        seed = int(random.random() * 1e5)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def validate_args(args):
    em = args.experiment_mode

    if args.frame_num <= 0:
        raise Exception("--frame_num must be > 0")

    if em in ("matcher", "self_matching", "dense_cross_attn"):
        if not args.nshot > 0:
            raise Exception(f"--experiment_mode={em} requires --nshot > 0")

    if em in ("matcher", "self_matching"):
        if args.use_query_as_support:
            raise Exception(f"--experiment_mode={em} is incompatible with --use_query_as_support")

    if args.sampling_inputs == "support_only" and em in ("self_attn", "attn_prior"):
        if not args.nshot > 0:
            raise Exception(
                f"--experiment_mode={em} with --sampling_inputs=support_only requires --nshot > 0"
            )

    if args.sampling_inputs != "both":
        # text_only and support_only are fully meaningful for attn_prior and self_attn.
        # support_only is also valid for matcher: it replaces the class label with the "visual"
        # sentinel in the fusion encoder's feature extraction pass (Exp 3b).
        # text_only for matcher has no additional effect and is therefore disallowed.
        allowed = em in ("attn_prior", "self_attn") or (
            em == "matcher" and args.sampling_inputs == "support_only"
        )
        if not allowed:
            raise Exception(
                "--sampling_inputs text_only/support_only only applies to "
                "--experiment_mode=attn_prior or self_attn; "
                "--sampling_inputs support_only also applies to --experiment_mode=matcher "
                "(replaces class label with 'visual' sentinel in the fusion feature extraction pass)"
            )

    if args.support_prompt_type == "box":
        if args.sampling_inputs == "text_only":
            raise Exception("--support_prompt_type=box has no effect with --sampling_inputs=text_only")
        if em not in ("attn_prior", "self_attn", "random"):
            raise Exception("--support_prompt_type=box only applies to --experiment_mode=attn_prior, self_attn, or random")


def get_arguments():
        parser = argparse.ArgumentParser(description='FSVOS')
        parser.add_argument("--checkpoint", type=str, default=None)
        parser.add_argument("--benchmark", type=str, default="youtube-fsvos", choices=["youtube_fsvos", "minivspw", "coco", "lvis", "ade20k", "pascal", "coco-20i", "pascal-5i", "lvis-92i"])
        parser.add_argument("--session_name", type=str, default=None)
        parser.add_argument("--dataset_path", type=str, default=None)
        parser.add_argument("--data_list_path", type=str, default=None)
        parser.add_argument("--output_dir", type=str, default="./output")
        parser.add_argument("--fold", type=int, default=1)
        parser.add_argument("--frame_num", type=int, default=1)
        parser.add_argument("--nshot", type=int, default=1)
        parser.add_argument("--use_synset_names", action="store_true", default=False)
        parser.add_argument("--synset_mapping_folder_path", type=str, default="/megaverse/storage/samele/FSS-SAM3/datasets/synset_mappings/leaf")
        parser.add_argument("--use_grouping_ade20k", action="store_true", default=False, help="Enable grouping of classes using JSON [ONLY ON ADE20K].")
        parser.add_argument("--all_lemmas", action="store_true", default=False, help="Iterate over all lemmas, instead of just the one selected inside the mapping")
        parser.add_argument("--experiment_mode", type=str, default="random",
            choices=["random", "matcher", "self_matching", "attn_prior", "dense_cross_attn", "self_attn"],
            help=(
                "Experiment to run. "
                "'random': random points from support masks (default). "
                "'matcher': bipartite point matching support→query (requires --nshot > 0). "
                "'self_matching': query self-matching with support embeddings, Exp 4. "
                "'attn_prior': top-k from fusion encoder cross-attention map, Exp 5. "
                "'dense_cross_attn': dense cross-attention to support foreground patches, Exp 6. "
                "'self_attn': bottom-k from fusion encoder self-attention map, Exp 7."
            ))
        parser.add_argument("--sampling_inputs", type=str, default="both",
            choices=["both", "text_only", "support_only"],
            help=(
                "Controls what goes into the fusion encoder during the point-selection sampling pass. "
                "Applies to attn_prior and self_attn. "
                "'both': text label + support visual prompts (default). "
                "'text_only': text label only (no visual prompts). "
                "'support_only': support visual prompts + 'visual' sentinel token (class label suppressed; SAM3 still receives 'visual' as its required text input, no custom class name is used in the sampling pass)."
            ))
        parser.add_argument("--run_n", type=int, default=0)
        parser.add_argument("--skip_coords", action="store_true", default=False, help="Skip coordinate-based embeddings when generating prompt tokens from support images")
        parser.add_argument("--use_fused_matcher_features", action="store_true", default=False, help="Use fused features from the fusion encoder instead of native PE backbone features for matcher (only relevant for experiment_mode=matcher).")
        parser.add_argument("--attn_layers", type=str, default="all", choices=["last", "all"], help="Which fusion encoder layers to aggregate attention from for both point selection and saving")
        parser.add_argument("--attention_aggregate_function", type=agg_function_arg, default="sum", help="Aggregation function to be used to aggregate attention matrices on the key dimension")
        parser.add_argument("--attn_sampling_mode", type=str, default="bottom-k", choices=["top-k", "bottom-k"], help="Point sampling method for attention priors")
        
        # Injection of Text Pooling Bias options
        # parser.add_argument("--inject_text_pooling", action="store_true", default=False, help="If set, add the pooled-text bias to every image patch (query and support) before the Fusion Encoder self-attention layers, mirroring the text conditioning applied during SAM3 training.")
        parser.add_argument("--inject_text_pooling", action="store_true", default=False, help="Inject text pooling bias into the fusion encoder")
        parser.add_argument("--injection_text_pooling_stage", type=str, default="point_sampling", choices=["point_sampling", "inference_pass", "both"], help="If --inject_text_pooling is set, this option determines when to inject the text pooling bias into the fusion encoder; either into the point sampling stage, the inference stage, or both.")
        parser.add_argument("--injection_text_pooling_in_prompts_sampling", action="store_true", default=False, help="If --inject_text_pooling is set and stage covers point_sampling, also bias support-side tokens during the sampling pass (visual prompt tokens for Exp 5/7, dense support spatial K/V for Exp 6, support prompt tokens for Exp 1). Default False = image patches only.")
        parser.add_argument("--injection_text_pooling_in_prompts_inference", action="store_true", default=False, help="If --inject_text_pooling is set and stage covers inference_pass, also bias the aggregated visual prompt fed to the inference forward. Default False = image patches only.")
        
        #
        parser.add_argument("--num_points_from_mask", type=int, default=20)
        parser.add_argument("--use_query_as_support", action="store_true", default=False, help="Use the query image as support image (only for 1-shot)")
        parser.add_argument("--disable_text_inference", action="store_true", default=False, help="Disable text prompts")
        parser.add_argument("--sampling", type=str, default="random", choices=["random", "top-k", "patch-core", "k-means-embeddings", "k-means-points", "k-medoids-embeddings", "k-medoids-points"], help="Sampling strategy for Matcher points")
        parser.add_argument("--visualize_embeddings", action="store_true", default=False, help="Generate t-SNE plots of the embeddings")
        # Random state management
        parser.add_argument('--seed', type=int, default=0)
        parser.add_argument("--support_prompt_type", type=str, default="points",
            choices=["points", "box"],
            help=(
                "Type of visual prompt built from support images. "
                "'points': random points from support mask (default). "
                "'box': bounding box of a connected blob (see --blob_selection). "
                "Applies to --experiment_mode=attn_prior, self_attn (sampling pass), and random (direct support encoding)."
        ))
        parser.add_argument("--blob_selection", type=str, default="largest",
            choices=["largest", "smallest"],
            help=(
                "Which connected blob to use when --support_prompt_type=box. "
                "'largest' (default): bounding box of the biggest foreground blob. "
                "'smallest': bounding box of the smallest blob with area >= 10 px "
                "(noise artifacts smaller than 10 px are ignored)."
        ))
        parser.add_argument("--sample_points_from_image", action="store_true", 
            default=False, 
            help="When sampling random points from the support image, or the query image (exp1, exp2). Sample randomly from the entire image instead of the mask region."
        )
        parser.add_argument("--fix_sampled_points", action="store_true",
            default=False, help=(
                "In the ALL_LEMMAS experiments, for the same synset, use always the same sampled points"
                "'query_as_support' use the points sampled randomly on the first lemma (the points don't depend on the lemma itself, is random)"
                "'attention_experiments' use the points sampled from the 'selected lemma' that is provided by the dataloader as the first lemma"
                "the script fixes the first set of points computed and reuse it for all the other lemmas"
            )
        )
        # Loggin arguments
        parser.add_argument('--log_dir', type=str, default='/megaverse/storage/samele/FSS-SAM3/experiment_results_logs')
        # Sharding: split inference across N parallel jobs by class, not by raw index.
        # Each shard receives a disjoint, contiguous slice of the sorted class list so
        # that no class is evaluated by more than one job.
        # Usage: --num_shards 3 --shard_id 0  (shard 0 of 3)
        parser.add_argument('--num_shards', type=int, default=1,
            help="Total number of parallel shards to divide the class list into (default: 1 = no sharding).")
        parser.add_argument('--shard_id', type=int, default=0,
            help="Zero-based index of this shard (0 <= shard_id < num_shards, default: 0).")
        return parser.parse_args()

def save_image_with_box(image, box, output_path):
    """
    Draws a red bounding box on the image and saves it.
    The box is expected to be in [x1, y1, x2, y2] format.
    """
    # Create a copy to avoid modifying the original
    res_image = Image.fromarray(image.copy())
    draw = ImageDraw.Draw(res_image)
    
    x1, y1, x2, y2 = box
    draw.rectangle(
        [x1 ,y1 ,x2 ,y2], 
        outline="red", 
        width=5
    )
    res_image.save(output_path)

def save_image_with_all_and_sampled_points(image, all_points, sampled_points, output_path):
    """
    Draws all points in blue and sampled points in red.
    """
    res_image = Image.fromarray(image.copy())
    draw = ImageDraw.Draw(res_image)
    
    if all_points is not None:
        if all_points.ndim == 3:
            all_points = all_points[:, 0, :]
        # Draw all points (blue, smaller)
        for (x, y) in all_points:
            draw.ellipse(
                [x-3, y-3, x+3, y+3], 
                fill="blue", 
                outline="blue"
            )
        
    if sampled_points is not None:
        if sampled_points.ndim == 3:
            sampled_points = sampled_points[:, 0, :]
        # Draw sampled points (red, larger)
        for (x, y) in sampled_points:
            draw.ellipse(
                [x-3, y-3, x+3, y+3], 
                fill="red", 
                outline="red"
            )
    res_image.save(output_path)

def rescale_to_pixel(data, image_size):
    """
    Rescale normalized [0,1] coordinates to pixel coordinates.
    Works for both points (shape [N,2] or [N,1,2]) and boxes (shape [4]).
    image_size: (W, H) tuple or int.
    """
    if data is None:
        return None
    if isinstance(image_size, int):
        w, h = image_size, image_size
    else:
        w, h = image_size

    data = np.array(data, dtype=np.float64)
    rescaled = data.copy()
    if rescaled.ndim == 1:  # Box [x1, y1, x2, y2]
        rescaled[0] *= w
        rescaled[2] *= w
        rescaled[1] *= h
        rescaled[3] *= h
    elif rescaled.ndim == 3:  # Points [N, 1, 2]
        rescaled[:, :, 0] *= w
        rescaled[:, :, 1] *= h
    else:  # Points [N, 2]
        rescaled[:, 0] *= w
        rescaled[:, 1] *= h
    return rescaled

def convert_norm_box_to_sam3_format(norm_box):
    """
    Convert a normalized [x1, y1, x2, y2] box (already in [0,1]) to
    the SAM3 format: [center_x, center_y, width, height] in [0,1].
    """
    x1, y1, x2, y2 = norm_box
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    width = x2 - x1
    height = y2 - y1
    return [center_x, center_y, width, height]


def save_attn_heatmap(img_numpy, attn_map_2d, save_path):
    h, w = img_numpy.shape[:2]
    attn_up = cv2.resize(attn_map_2d, (w, h), interpolation=cv2.INTER_LINEAR)
    attn_norm = (attn_up - attn_up.min()) / (attn_up.max() - attn_up.min() + 1e-8)
    heatmap_bgr = cv2.applyColorMap((attn_norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
    
    # Blend with original image
    img_bgr = cv2.cvtColor(img_numpy, cv2.COLOR_RGB2BGR)
    overlay = cv2.addWeighted(img_bgr, 0.5, heatmap_bgr, 0.5, 0)
    cv2.imwrite(save_path, overlay)


def save_mask_overlay(image, mask, output_path):
    """Save image with mask overlay"""
    # Ensure image is numpy array
    if isinstance(image, Image.Image):
        image = np.array(image)
    if torch.is_tensor(image):
        image = image.numpy()
    if torch.is_tensor(mask):
        mask = mask.numpy()
    
    # Handle float32 images from transform (values in range 0-1)
    if image.dtype == np.float32:
        image = (image * 255).astype(np.uint8)

    # print(f"Image shape: {image.shape[:2]}, Mask shape: {mask.shape[:2]}")
    
    # Ensure mask is 2D boolean array
    mask = mask.squeeze()  # Remove any extra dimensions
    if mask.ndim > 2:
        mask = mask[:, :, 0] if mask.shape[2] == 1 else mask.max(axis=2)
    
    if mask.shape[:2] != image.shape[:2]:
        # print(f"Warning: Mask shape {mask.shape} doesn't match image shape {image.shape}")
        mask = cv2.resize(mask.astype(np.uint8), (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
    
    # Ensure mask is boolean
    mask = mask.astype(bool)
    
    # Create colored mask overlay
    colored_mask = np.zeros_like(image, dtype=np.uint8)
    colored_mask[mask] = [255, 0, 0]  # Red overlay
    
    # Blend with original image
    overlay = cv2.addWeighted(image.astype(np.uint8), 0.65, colored_mask.astype(np.uint8), 0.35, 0)
    cv2.imwrite(output_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

def save_image(image, path):
    """Save a numpy array or PIL image to the specified path."""
    if torch.is_tensor(image):
        image = image.numpy()
    if isinstance(image, np.ndarray):
        # Handle boolean arrays (masks)
        if image.dtype == bool:
            image = (image * 255).astype(np.uint8)
        # Handle float32 images from transform
        elif image.dtype == np.float32:
            # Only multiply by 255 if values are in 0-1 range
            # Masks from ADE20K/PASCAL have values 0-255, don't multiply again
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            else:
                image = image.astype(np.uint8)
        image = Image.fromarray(image)
    image.save(path)

def save_results(class_dic, evaluator, args, virtual_to_original=None, original_class_dic=None, box_coordinates=None, point_scores=None, grond_truth_class_dic=None):
    # Save results to file
    results_dir = os.path.join(args.log_dir, args.benchmark)
    if args.session_name is not None:
        results_dir = os.path.join(results_dir, args.session_name)
    os.makedirs(results_dir, exist_ok=True)

    class_list_idx = list(class_dic.keys())
    class_list_names = list(class_dic.values())

    # Helper to get original class info
    def _orig(cid):
        if virtual_to_original is not None and original_class_dic is not None:
            orig_cid = virtual_to_original.get(cid, cid)
            return orig_cid, original_class_dic.get(orig_cid, '')
        return None, None
    
    def _ground_truth_orig(cid):
        if grond_truth_class_dic is not None:
            if virtual_to_original is not None:
                orig_cid = virtual_to_original.get(cid, cid)
            else:
                orig_cid = cid
            return orig_cid, grond_truth_class_dic.get(orig_cid, '')
        return None, None


    size_data = []
    for label, scores in [('SMALL', evaluator.iou_small_score), ('MEDIUM', evaluator.iou_medium_score), ('LARGE', evaluator.iou_large_score)]:
        for cid, cname, s in zip(class_list_idx, class_list_names, scores):
            row = {'size': label, 'class_id': cid, 'class_name': cname, 'score': float(s)}
            orig_cid, orig_cname = _orig(cid)
            if orig_cid is not None:
                row['original_class_idx'] = orig_cid
                row['original_class_name'] = orig_cname
            orig_g_cid, orig_g_cname = _ground_truth_orig(cid)
            if orig_g_cid is not None:
                row['original_ground_truth_class_idx'] = orig_g_cid
                row['original_ground_truth_class_name'] = orig_g_cname
            size_data.append(row)
    size_df = pd.DataFrame(size_data)
    size_csv_path = os.path.join(results_dir, f"{args.benchmark}_size_scores.csv")
    size_df.to_csv(size_csv_path, index=False, sep=';')
    print(f"Size scores saved to {size_csv_path}")

    sample_data = []
    blob_data = []
    for class_idx, class_name, sample_list in zip(class_list_idx, class_list_names, evaluator.sample_details):
        for sample in sample_list:
            row_pt_score = sample.get('point_score', -1.0)
            if row_pt_score is None:
                row_pt_score = -1.0
                
            row = {
                'size': sample['size_category'],
                'class_id': class_idx,
                'class_name': class_name,
                'j_score': float(sample['j_score']),
                'pixel_ratio': float(sample['pixel_ratio']),
                'sample_id': sample['sample_id'],
                'point_score': float(row_pt_score),
                'all_point_score': float(sample.get('all_point_score', -1.0)),
                'num_total_points': int(sample.get('num_total_points', -1)),
            }
            blob_rows = [{
                'class_id': class_idx,
                'class_name': class_name,
                'sample_id': sample['sample_id'],
                **blob,
            } for blob in sample['blob_results']]
            
            orig_cid, orig_cname = _orig(class_idx)
            orig_g_cid, orig_g_cname = _ground_truth_orig(class_idx)
            if orig_cid is not None:
                row['original_class_idx'] = orig_cid
                row['original_class_name'] = orig_cname
                for blob in blob_rows:
                    blob['original_class_idx'] = orig_cid
                    blob['original_class_name'] = orig_cname
            if orig_g_cid is not None:
                row['original_ground_truth_class_idx'] = orig_g_cid
                row['original_ground_truth_class_name'] = orig_g_cname
                for blob in blob_rows:
                    blob['original_ground_truth_class_idx'] = orig_g_cid
                    blob['original_ground_truth_class_name'] = orig_g_cname
            
            sample_data.append(row)
            blob_data.extend(blob_rows)
            
    sample_df = pd.DataFrame(sample_data)
    sample_csv_path = os.path.join(results_dir, f"{args.benchmark}_sample_scores.csv")
    sample_df.to_csv(sample_csv_path, index=False, sep=';')
    print(f"Sample scores saved to {sample_csv_path}")

    blob_df = pd.DataFrame(blob_data)
    blob_csv_path = os.path.join(results_dir, f"{args.benchmark}_blob_scores.csv")
    blob_df.to_csv(blob_csv_path, index=False, sep=';')
    print(f"Blob scores saved to {blob_csv_path}")

    # helper for mean
    def _mean(l):
        return np.mean(l) if len(l) > 0 else -1.0

    # Class-level scores CSV
    class_scores_data = []
    for ic, (cid, cname, iou) in enumerate(zip(class_list_idx, class_list_names, evaluator.iou_list)):
        row = {
            'class_id': cid, 
            'class_name': cname, 
            'iou_score': float(iou),
            'point_accuracy_micro': float(evaluator.pt_accuracy[ic]),
            'all_point_accuracy_micro': float(evaluator.all_point_accuracy[ic]),
            'point_accuracy_macro': float(evaluator.pt_accuracy_macro[ic]),
            'all_point_accuracy_macro': float(evaluator.all_point_accuracy_macro[ic])
        }
        if virtual_to_original is not None and original_class_dic is not None:
            original_cid = virtual_to_original.get(cid, cid)
            row['original_class_idx'] = original_cid
            row['original_class_name'] = original_class_dic.get(original_cid, '')
            orig_g_cid, orig_g_cname = _ground_truth_orig(cid)
            if orig_g_cid is not None:
                row['original_ground_truth_class_idx'] = orig_g_cid
                row['original_ground_truth_class_name'] = orig_g_cname
        class_scores_data.append(row)
        
    class_scores_df = pd.DataFrame(class_scores_data)
    class_scores_csv_path = os.path.join(results_dir, f"{args.benchmark}_class_scores.csv")
    class_scores_df.to_csv(class_scores_csv_path, index=False, sep=';')
    print(f"Class scores saved to {class_scores_csv_path}")
    
    if box_coordinates is not None:
        box_sizes_df = pd.DataFrame(box_coordinates, columns=["center_x", "center_y", "width", "height"])
        box_sizes_csv_path = os.path.join(results_dir, f"{args.benchmark}_box_sizes_scores.csv")
        box_sizes_df.to_csv(box_sizes_csv_path, index=False, sep=";")
        print(f"Box sizes saved to {box_sizes_csv_path}")


def compute_point_features(mask, pts_xy):
    """
    Compute geometry features for a set of positive prompt points against a GT mask.

    Args:
        mask : 2-D numpy array (H, W), any numeric dtype — will be binarised > 0.
        pts_xy : numpy array of shape [N, 2] with (x, y) in the SAME pixel frame as mask.
                 Negative or out-of-bounds points are clipped before use.

    Returns a dict with keys matching the CSV schema.  All features are NaN when
    the mask is empty or its distance transform has zero interior radius.

    Coordinate-frame contract
    -------------------------
    This function is called with `mask = ground_truth` (original-image H×W)
    and `pts_xy = rescaled_sampled_pts` (positive prompt points, same frame).
    Both variables are already in the original-image pixel coordinate system
    before this function is invoked (see the call-site comment in main()).

    Multi-instance handling
    -----------------------
    When the GT mask contains multiple disconnected instances (blobs), EDT-based
    metrics (object_radius_px, coverage_gap_*, dt_depth_*) are computed **per
    blob** using only the pixels and points that belong to that blob.  The
    reported scalar is a blob-area-weighted average across all blobs that
    received at least one prompt point.  Blobs with no points are excluded from
    the EDT aggregation — point-placement quality is only assessed for the blobs
    that were actually prompted.

    Global metrics (frac_offmask, dispersion_norm, centroid_offset_norm) still
    use the full mask/point set, but dispersion_norm and centroid_offset_norm are
    normalised by the area-weighted mean blob radius (= object_radius_px) so the
    scale remains comparable across scenes.  For the single-instance case the
    behaviour is identical to the previous implementation.
    """
    nan = float("nan")
    feat = dict(
        n_points=0, n_neg_points=0, frac_offmask=nan,
        object_radius_px=nan,
        coverage_gap_mean=nan, coverage_gap_p95=nan,
        dt_depth_mean=nan, dt_depth_min=nan,
        dispersion_norm=nan, centroid_offset_norm=nan,
        points_xy="[]",
    )

    # ── Normalise mask ────────────────────────────────────────────────────────
    mask_arr = np.array(mask)
    if mask_arr.ndim > 2:
        mask_arr = mask_arr.squeeze()
    M = (mask_arr > 0).astype(np.uint8)  # binary {0, 1}

    H, W = M.shape

    # ── Normalise points ──────────────────────────────────────────────────────
    if pts_xy is None or len(pts_xy) == 0:
        return feat

    pts = np.array(pts_xy, dtype=np.float64)
    if pts.ndim == 3:          # [N, 1, 2] → [N, 2]
        pts = pts[:, 0, :]
    assert pts.ndim == 2 and pts.shape[1] == 2, f"pts_xy must be (N,2), got {pts.shape}"

    n_pts = len(pts)
    feat["n_points"] = n_pts
    feat["n_neg_points"] = 0
    feat["points_xy"] = json.dumps([[float(x), float(y)] for x, y in pts])

    # Clip to image bounds for indexing (may be slightly off due to rounding)
    px = np.clip(np.round(pts[:, 0]).astype(int), 0, W - 1)   # x → col
    py = np.clip(np.round(pts[:, 1]).astype(int), 0, H - 1)   # y → row

    # ── frac_offmask (global — whole mask) ───────────────────────────────────
    on_mask_flags = M[py, px] > 0
    n_on = int(on_mask_flags.sum())
    feat["frac_offmask"] = float(1.0 - n_on / n_pts)

    if M.sum() == 0:
        return feat  # empty mask → NaN for all geometry features

    # ── Connected-component labelling ─────────────────────────────────────────
    # label_map: 0 = background, 1..K = individual foreground blobs.
    n_labels, label_map, cc_stats, _ = cv2.connectedComponentsWithStats(
        M, connectivity=8
    )

    # Which blob does each point fall in? (0 = off-mask / background)
    pt_blob_ids = label_map[py, px]  # shape [N], values 0..K

    # ── Per-blob EDT metrics ──────────────────────────────────────────────────
    # Iterate over blobs that contain ≥1 on-mask point; compute all EDT-based
    # quantities restricted to that blob, normalised by that blob's own
    # inscribed-circle radius (= max of the blob-local EDT).

    blob_radii    = []   # r per prompted blob
    blob_areas    = []   # pixel area per prompted blob  (weight)
    blob_cov_gap_means = []
    blob_cov_gap_p95s  = []
    blob_dt_means = []
    blob_dt_mins  = []
    blob_centroids = []  # (cx, cy) centre-of-mass per prompted blob

    for blob_label in range(1, n_labels):   # label 0 is background
        blob_pt_flags = pt_blob_ids == blob_label   # which of the N points land here
        if not blob_pt_flags.any():
            continue   # no prompt point in this blob → skip entirely

        # Mask restricted to this single blob
        blob_M = (label_map == blob_label).astype(np.uint8)
        blob_area = int(blob_M.sum())

        # EDT and inscribed-circle radius for this blob only
        blob_dt = scipy_ndimage.distance_transform_edt(blob_M)
        r_blob = float(blob_dt.max())

        blob_radii.append(r_blob)
        blob_areas.append(blob_area)
        cy_b, cx_b = scipy_ndimage.center_of_mass(blob_M)
        blob_centroids.append((cx_b, cy_b))

        if r_blob == 0:
            # Degenerate single-pixel blob — EDT metrics are meaningless
            blob_cov_gap_means.append(nan)
            blob_cov_gap_p95s.append(nan)
            blob_dt_means.append(nan)
            blob_dt_mins.append(nan)
            continue

        # Coverage gap: nearest-point distance field measured inside this blob only
        blob_px = px[blob_pt_flags]
        blob_py = py[blob_pt_flags]
        seed = np.ones((H, W), dtype=np.uint8)
        for xi, yi in zip(blob_px, blob_py):
            seed[yi, xi] = 0
        dfield = scipy_ndimage.distance_transform_edt(seed)
        inside = dfield[blob_M > 0]
        blob_cov_gap_means.append(float(inside.mean()  / r_blob))
        blob_cov_gap_p95s.append(float(np.percentile(inside, 95) / r_blob))

        # DT depth of the on-blob points, normalised by this blob's radius
        depths = blob_dt[blob_py, blob_px] / r_blob
        blob_dt_means.append(float(depths.mean()))
        blob_dt_mins.append(float(depths.min()))

    # ── Area-weighted aggregation across prompted blobs ───────────────────────
    def _weighted_mean(values, weights):
        """Weighted mean, ignoring NaN entries; returns NaN if no valid entry."""
        v = np.array(values, dtype=np.float64)
        w = np.array(weights, dtype=np.float64)
        ok = ~np.isnan(v)
        if not ok.any():
            return nan
        return float(np.average(v[ok], weights=w[ok]))

    if blob_radii:
        areas_arr = np.array(blob_areas, dtype=np.float64)
        feat["object_radius_px"]  = _weighted_mean(blob_radii,          areas_arr)
        feat["coverage_gap_mean"] = _weighted_mean(blob_cov_gap_means,  areas_arr)
        feat["coverage_gap_p95"]  = _weighted_mean(blob_cov_gap_p95s,   areas_arr)
        feat["dt_depth_mean"]     = _weighted_mean(blob_dt_means,       areas_arr)
        feat["dt_depth_min"]      = _weighted_mean(blob_dt_mins,        areas_arr)
    # else: no on-mask points → all EDT features remain NaN

    # Area-weighted mean blob radius used as global normaliser below.
    r_global = feat["object_radius_px"]   # NaN if no prompted blob

    # ── Dispersion (global point set, normalised by mean blob radius) ──────────
    if n_pts >= 2:
        diffs = pts[:, None, :] - pts[None, :, :]          # [N, N, 2]
        pair_dists = np.sqrt((diffs ** 2).sum(axis=2))      # [N, N]
        upper = pair_dists[np.triu_indices(n_pts, k=1)]
        if not np.isnan(r_global) and r_global > 0:
            feat["dispersion_norm"] = float(upper.mean() / (2.0 * r_global))
        # else remain NaN (all points off-mask or degenerate)
    else:
        feat["dispersion_norm"] = 0.0

    # ── Centroid offset (mean-pt vs each prompted blob's centroid, weighted) ───
    if blob_centroids and not np.isnan(r_global) and r_global > 0:
        mean_pt = pts.mean(axis=0)   # (x_mean, y_mean) of ALL prompt points
        offsets = [
            np.sqrt((mean_pt[0] - cx_b) ** 2 + (mean_pt[1] - cy_b) ** 2)
            for (cx_b, cy_b) in blob_centroids
        ]
        areas_arr = np.array(blob_areas, dtype=np.float64)
        weighted_offset = _weighted_mean(offsets, areas_arr)
        feat["centroid_offset_norm"] = float(weighted_offset / r_global) \
            if not np.isnan(weighted_offset) else nan

    return feat


def save_point_features_csv(point_feat_rows, args):
    """Write the point_features CSV alongside the existing sample_scores CSV."""
    if not point_feat_rows:
        print("[PointFeatures] No rows to save — skipping point_features CSV.")
        return
    results_dir = os.path.join(args.log_dir, args.benchmark)
    if args.session_name is not None:
        results_dir = os.path.join(results_dir, args.session_name)
    os.makedirs(results_dir, exist_ok=True)
    df = pd.DataFrame(point_feat_rows)
    csv_path = os.path.join(results_dir, f"{args.benchmark}_point_features.csv")
    df.to_csv(csv_path, index=False, sep=";")
    print(f"Point features saved to {csv_path}")


def get_random_points_from_mask(mask, num_points):
    mask_arr = mask.cpu().numpy() if hasattr(mask, 'cpu') else np.array(mask)
    if mask_arr.ndim > 2:
        mask_arr = mask_arr.squeeze()
        
    y_coords, x_coords = np.where(mask_arr > 0)
    if len(x_coords) == 0:
        print("Warning mask is empty")
        return None, None
        
    # Sample N points
    indices = np.random.choice(len(x_coords), min(num_points, len(x_coords)), replace=False)
    pts_actual = np.stack([x_coords[indices], y_coords[indices]], axis=1)
    pts_norm = np.stack([x_coords[indices] / mask_arr.shape[1], y_coords[indices] / mask_arr.shape[0]], axis=1)
    return pts_norm[:, None, :], pts_actual

def get_bbox_from_blob(mask, blob_selection="largest"):
    """
    Returns the bounding box of a connected foreground blob as
    normalized [cx, cy, w, h] in [0, 1] (SAM3 add_geometric_prompt format).

    Args:
        mask: 2-D binary mask (tensor or numpy array).
        blob_selection: "largest" (default) selects the biggest blob;
                        "smallest" selects the smallest blob with area >= 10 px
                        to skip single-pixel noise artifacts.
    Returns None if the mask is empty or no qualifying blob is found.
    """
    mask_arr = mask.cpu().numpy() if hasattr(mask, 'cpu') else np.array(mask)
    mask_arr = (mask_arr.squeeze() > 0).astype(np.uint8)
    n_labels, _, stats, _ = cv2.connectedComponentsWithStats(mask_arr, connectivity=8)
    if n_labels <= 1:
        return None
    areas = stats[1:, cv2.CC_STAT_AREA]  # exclude background label 0
    if blob_selection == "smallest":
        MIN_AREA = 10  # ignore noise blobs
        valid = np.where(areas >= MIN_AREA)[0]
        if len(valid) == 0:
            # Fallback to the absolute smallest blob if none are >= 10px
            best = int(np.argmin(areas)) + 1
        else:
            best = int(valid[np.argmin(areas[valid])]) + 1
    else:  # "largest"
        best = int(np.argmax(areas)) + 1
    x1 = stats[best, cv2.CC_STAT_LEFT]
    y1 = stats[best, cv2.CC_STAT_TOP]
    bw = stats[best, cv2.CC_STAT_WIDTH]
    bh = stats[best, cv2.CC_STAT_HEIGHT]
    H, W = mask_arr.shape
    cx = (x1 + bw / 2) / W
    cy = (y1 + bh / 2) / H
    nw = bw / W
    nh = bh / H
    return [cx, cy, nw, nh]

# Backwards-compatible alias
def get_bbox_from_largest_blob(mask):
    return get_bbox_from_blob(mask, blob_selection="largest")

def plot_embeddings_tsne(all_features, matched_indices, sampled_indices, output_path):
    """
    Plots a t-SNE visualization of the embeddings.
    - all_features: Full set of target embeddings [N x 256]
    - matched_indices: Indices in all_features that were found by the matcher
    - sampled_indices: Indices in matched_features that were sampled by coreset
    """

    print(f"[Visualization] Running PCA and t-SNE for {len(all_features)} points...")
    features_np = all_features.cpu().numpy()
    
    # 1. PCA to 50 components
    pca = PCA(n_components=min(50, features_np.shape[0], features_np.shape[1]))
    features_pca = pca.fit_transform(features_np)
    
    # 2. t-SNE
    tsne = TSNE(n_components=2, perplexity=30, max_iter=300, random_state=0)
    features_2d = tsne.fit_transform(features_pca)
    
    # 3. Plot
    plt.figure(figsize=(10, 10))
    # All features in gray
    plt.scatter(features_2d[:, 0], features_2d[:, 1], c='lightgray', s=10, alpha=0.3, label='Full Image Features')
    
    if matched_indices is not None and len(matched_indices) > 0:
        # Matcher features in blue
        m_idx = matched_indices.cpu().numpy() if hasattr(matched_indices, 'cpu') else matched_indices
        plt.scatter(features_2d[m_idx, 0], features_2d[m_idx, 1], c='blue', s=30, alpha=0.6, label='Points via Bipartite Matcher')
        
        if sampled_indices is not None and len(sampled_indices) > 0:
            # Core-set sampled features in red
            final_s_idx = m_idx[sampled_indices]
            plt.scatter(features_2d[final_s_idx, 0], features_2d[final_s_idx, 1], c='red', s=30, label='Final Core-set Samples')

    plt.title("t-SNE Embedding Space Visualization")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"[Visualization] t-SNE plot saved to {output_path}")

def _postprocess_matcher_output(box, points, matched_features, all_target_features, matched_indices_in_all, res, num_points, sampling, visualize, visual_output_path):
    """
    Shared post-processing for matcher output: normalize, sample, visualize, normalize box.
    Returns (pts_norm_sampled, all_pts_norm, norm_box) or (None, None, None) if no points.
    """
    if len(points) == 0:
        return None, None, None

    all_pts_norm = points.astype(np.float64).copy()
    all_pts_norm[:, 0] /= res
    all_pts_norm[:, 1] /= res

    print(f"[Sampling] strategy='{sampling}' | {len(all_pts_norm)} candidates → target {num_points} points")
    if len(all_pts_norm) > num_points:
        if sampling == "patch-core":
            sampler = GreedyCoresetSampler(device=matched_features.device, n_samples=num_points)
            _, sampled_indices = sampler.run(matched_features)
            pts_norm_sampled = all_pts_norm[sampled_indices][:, None, :]
        elif sampling == "top-k":
            sampled_indices = np.arange(num_points)
            pts_norm_sampled = all_pts_norm[:num_points][:, None, :]
        elif sampling == "random":
            sampled_indices = np.random.choice(len(all_pts_norm), num_points, replace=False)
            pts_norm_sampled = all_pts_norm[sampled_indices][:, None, :]
        elif sampling == "k-means-embeddings":
            feat_np = matched_features.cpu().numpy()
            kmeans = KMeans(n_clusters=num_points, random_state=0, n_init=10).fit(feat_np)
            dists = pairwise_distances(kmeans.cluster_centers_, feat_np)
            sampled_indices = np.argmin(dists, axis=1)
            pts_norm_sampled = all_pts_norm[sampled_indices][:, None, :]
        elif sampling == "k-means-points":
            kmeans = KMeans(n_clusters=num_points, random_state=0, n_init=10).fit(all_pts_norm)
            dists = pairwise_distances(kmeans.cluster_centers_, all_pts_norm)
            sampled_indices = np.argmin(dists, axis=1)
            pts_norm_sampled = all_pts_norm[sampled_indices][:, None, :]
        elif sampling == "k-medoids-embeddings":
            feat_np = matched_features.cpu().numpy()
            kmedoids = KMedoids(n_clusters=num_points, random_state=0, init='k-medoids++').fit(feat_np)
            sampled_indices = kmedoids.medoid_indices_
            pts_norm_sampled = all_pts_norm[sampled_indices][:, None, :]
        elif sampling == "k-medoids-points":
            kmedoids = KMedoids(n_clusters=num_points, random_state=0, init='k-medoids++').fit(all_pts_norm)
            sampled_indices = kmedoids.medoid_indices_
            pts_norm_sampled = all_pts_norm[sampled_indices][:, None, :]
    else:
        pts_norm_sampled = all_pts_norm[:, None, :]
        sampled_indices = np.arange(len(all_pts_norm))

    print(f"[Sampling] selected {len(pts_norm_sampled)} points")
    if visualize and len(points) >= 30 and visual_output_path:
        plot_embeddings_tsne(all_target_features, matched_indices_in_all, sampled_indices, visual_output_path)

    norm_box = [box[0] / res, box[1] / res, box[2] / res, box[3] / res] if box is not None else None

    return pts_norm_sampled, all_pts_norm, norm_box


def get_points_from_matcher(support_imgs=None, support_masks=None, query_img=None, num_points=20, matcher_calculator=None, text_prompt="visual", use_fused_matcher_features=False, skip_coords=False, sampling="random", visualize=False, visual_output_path=None, reference_visual_prompt=None, reference_visual_mask=None, attn_layers="all"):
    """
    Returns all data in normalized [0,1] coordinates:
      - pts_norm_sampled: [N, 1, 2] normalized sampled points (for SAM3)
      - all_pts_norm:     [M, 2]    normalized full point set
      - norm_box:         [4]       normalized box [x1, y1, x2, y2] or None
    """
    if support_imgs is None:
        raise Exception("Support images are not specified")
    if support_masks is None:
        raise Exception("Support masks are not specified")
    if query_img is None:
        raise Exception("Query image is not specified")
    if matcher_calculator is None:
        raise Exception("Matcher calculator is not specified")

    print(f"[Matcher] support→query | nshot={support_imgs.shape[0]} | text='{text_prompt}' | fused={use_fused_matcher_features}")
    box, points, matched_features, all_target_features, matched_indices_in_all = matcher_calculator.compute_box(
        reference_image=support_imgs,
        target_image=query_img,
        reference_mask=support_masks,
        text_prompt=text_prompt,
        use_fused_matcher_features=use_fused_matcher_features,
        skip_coords=skip_coords,
        reference_visual_prompt=reference_visual_prompt,
        reference_visual_mask=reference_visual_mask,
        attn_layers=attn_layers,
    )
    print(f"[Matcher] compute_box returned {len(points)} candidate points")
    if len(points) == 0:
        print("[WARNING] - Matcher returned 0 points.")
        return None, None, None

    return _postprocess_matcher_output(
        box, points, matched_features, all_target_features, matched_indices_in_all,
        float(matcher_calculator.resolution), num_points, sampling, visualize, visual_output_path
    )

# Helper function to encode point prompts (reused in get_prompt_tokens_from_support and get_query_self_matching_points)
def encode_pts_prompts(processor, image, pts, skip_coords):
    state = processor.set_image(image)
    state = processor.add_point_prompts(pts, [True]*len(pts), state)
    state = processor._encode_current_prompts(state, encode_text=False, skip_coords=skip_coords)
    return state

def encode_box_prompts(processor, image, box_cxcywh, skip_coords):
    state = processor.set_image(image)
    state = processor.add_box_prompts(box_cxcywh, True, state)
    state = processor._encode_current_prompts(state, encode_text=False, skip_coords=skip_coords)
    return state

def encode_support_visual_tokens(processor, support_imgs, support_masks, num_points, skip_coords, tag="Support"):
    """Encode random points from each support mask as visual tokens. Returns (agg_prompt, agg_mask)."""
    assert support_imgs.shape[0] == support_masks.shape[0]
    visual_tokens = []
    visual_masks = []
    for idx in range(support_imgs.shape[0]):
        pts_norm, _ = get_random_points_from_mask(mask=support_masks[idx], num_points=num_points)
        if pts_norm is None:
            raise Exception(f"Mask for support image {idx} is empty. No points have been returned.")
        print(f"[{tag}] support {idx+1}/{support_imgs.shape[0]} - encoding {len(pts_norm)} random points from mask")
        state = encode_pts_prompts(processor, support_imgs[idx], pts_norm, skip_coords)
        visual_tokens.append(state["prompt"])
        visual_masks.append(state["prompt_mask"])
    if not visual_tokens:
        raise ValueError("No valid support shots found.")
    return torch.cat(visual_tokens, dim=0), torch.cat(visual_masks, dim=1)


def encode_support_box_tokens(processor, support_imgs, support_masks, skip_coords,
                              tag="SupportBox", blob_selection="largest"):
    """Encode a blob's bounding box from each support mask as visual tokens.
    Args:
        blob_selection: "largest" (default) or "smallest" — which connected blob to use.
    Returns (agg_prompt, agg_mask, boxes_list) where boxes_list is [[cx,cy,w,h], ...] per shot."""
    assert support_imgs.shape[0] == support_masks.shape[0]
    visual_tokens = []
    visual_masks = []
    boxes_list = []
    for idx in range(support_imgs.shape[0]):
        box_cxcywh = get_bbox_from_blob(support_masks[idx], blob_selection=blob_selection)
        if box_cxcywh is None:
            raise Exception(f"Mask for support image {idx} is empty — no bounding box available.")
        print(f"[{tag}] support {idx+1}/{support_imgs.shape[0]} ({blob_selection} blob) - encoding box {[f'{v:.3f}' for v in box_cxcywh]}")
        state = encode_box_prompts(processor, support_imgs[idx], box_cxcywh, skip_coords)
        visual_tokens.append(state["prompt"])
        visual_masks.append(state["prompt_mask"])
        boxes_list.append(box_cxcywh)
    if not visual_tokens:
        raise ValueError("No valid support shots found.")
    return torch.cat(visual_tokens, dim=0), torch.cat(visual_masks, dim=1), boxes_list


def get_query_self_matching_points(processor=None, support_imgs=None, support_masks=None, query_img=None, num_points=20, matcher_calculator=None, text_prompt="visual", skip_coords=False, sampling="random", visualize=False, visual_output_path=None):
    """
    Query self-matching: compute matcher points between two feature volumes of the query image.
    Reference features: query + support embeddings + text
    Target features: query + text only

    Returns all data in normalized [0,1] coordinates:
      - pts_norm_sampled: [N, 1, 2] normalized sampled points (for SAM3)
      - all_pts_norm:     [M, 2]    normalized full point set
      - norm_box:         [4]       normalized box [x1, y1, x2, y2] or None
    """
    if processor is None:
        raise Exception("Processor is not specified")
    if support_imgs is None:
        raise Exception("Support images are not specified")
    if support_masks is None:
        raise Exception("Support masks are not specified")
    if query_img is None:
        raise Exception("Query image is not specified")
    if matcher_calculator is None:
        raise Exception("Matcher calculator is not specified")

    # Step 1: Extract visual embeddings from support images (random points from masks)
    aggregated_visual_prompt, aggregated_visual_mask = encode_support_visual_tokens(
        processor, support_imgs, support_masks, num_points, skip_coords, tag="QuerySelfMatching"
    )
    print(f"[QuerySelfMatching] aggregated visual prompt: {aggregated_visual_prompt.shape} | running self-matching on query | text='{text_prompt}'")

    # Step 2: Call matcher with query self-matching mode
    # We pass the query image as target_image and use a dummy reference_mask (all ones)
    # The reference_mask is used to pool features, but in self-matching we are matching query_with_support and query
    dummy_reference_mask = torch.ones((1008, 1008), device=query_img.device)


    # 2 Fused feature volumes
    # Reference: query features fused with support points and text label
    # Target: query features fused with text label
    box, points, matched_features, all_target_features, matched_indices_in_all = matcher_calculator.compute_box(
        reference_image=query_img,  # Not used in query self-matching, but required for API
        target_image=query_img,
        reference_mask=dummy_reference_mask,
        text_prompt=text_prompt,
        use_fused_matcher_features=True,  # Required for query self-matching
        skip_coords=skip_coords,
        use_query_self_matching=True,
        reference_visual_prompt=aggregated_visual_prompt,
        reference_visual_mask=aggregated_visual_mask,
    )

    print(f"[QuerySelfMatching] compute_box returned {len(points)} candidate points")
    if len(points) == 0:
        print("[WARNING] - Query self-matching returned 0 points.")
        return None, None, None

    return _postprocess_matcher_output(
        box, points, matched_features, all_target_features, matched_indices_in_all,
        float(matcher_calculator.resolution), num_points, sampling, visualize, visual_output_path
    )

def get_attn_prior_points(query_img, text_prompt, num_points, matcher_calculator, attn_layers,
                           visual_prompt=None, visual_prompt_mask=None, attention_aggregate_function="sum",
                           attn_sampling_mode="top-k", inject_into_image_patches=False, skip_coords=False):
    """
    Sample num_points prompt points from the fusion encoder visual/points cross-attention map.
    Returns norm_pts [num_points, 2] in normalized (x, y) coords.
    Bypasses compute_box and _postprocess_matcher_output. Attention ranking is the final selection.
    """
    matcher_calculator.get_fused_image_features(
        query_img,
        text_prompt=text_prompt,
        visual_prompt=visual_prompt,
        visual_prompt_mask=visual_prompt_mask,
        attn_layers=attn_layers,
        agg_function=attention_aggregate_function,
        inject_into_image_patches=inject_into_image_patches,
        skip_coords=skip_coords
    )
    # Point selection uses attention to visual/point tokens (not text). But if no support prompts are provided, points tokens are not present, therefore sample from the text map.
    if visual_prompt is not None:
        pts_map = matcher_calculator.last_cross_attn_points_map  # [72, 72] numpy
    else:
        pts_map = matcher_calculator.last_cross_attn_text_map  # [72, 72] numpy
    H = W = matcher_calculator.encoder_feat_size  # 72
    patch_size = matcher_calculator.encoder_patch_size  # 14
    resolution = float(matcher_calculator.resolution)  # 1008

    attn_flat = torch.from_numpy(pts_map).flatten()  # [5184]
    sampled_idx = sample_points_from_attn_map(attn_flat, sampling_method=attn_sampling_mode, k=num_points)
    px = (sampled_idx % W) * patch_size + patch_size // 2
    py = (sampled_idx // W) * patch_size + patch_size // 2
    norm_pts = torch.stack([px.float() / resolution, py.float() / resolution], dim=1)
    print(f"[AttnPrior] sampled {num_points} points from cross-attention map (layers={attn_layers}, sampling_method={attn_sampling_mode})")
    return norm_pts.cpu().numpy()

def sample_points_from_attn_map(attn_map, sampling_method="top-k", k=5):
    if sampling_method == "bottom-k":
        return torch.topk(attn_map, k=k, largest=False).indices
    elif sampling_method == "top-k":
        return torch.topk(attn_map, k=k, largest=True).indices
    else:
        # fallback to top-k
        return torch.topk(attn_map, k=k, largest=True).indices

def get_self_attn_points(query_img, text_prompt, num_points, matcher_calculator,
                                  attn_layers, visual_prompt=None, visual_prompt_mask=None,
                                  include_text_in_prompt=True, attention_aggregate_function="sum",
                                  attn_sampling_mode="bottom-k", skip_coords=False,
                                  inject_into_image_patches=False):
    """
    Exp 7: select num_points prompt points from the fusion encoder self-attention map,
    choosing the bottom-k patches (lowest row-sum = empirically the object region).
    Returns norm_pts [num_points, 2] in normalized (x, y) coords.
    """
    matcher_calculator.get_fused_image_features(
        query_img,
        text_prompt=text_prompt,
        visual_prompt=visual_prompt,
        visual_prompt_mask=visual_prompt_mask,
        attn_layers=attn_layers,
        include_text_in_prompt=include_text_in_prompt,
        agg_function=attention_aggregate_function,
        skip_coords=skip_coords,
        inject_into_image_patches=inject_into_image_patches,
    )
    self_map = matcher_calculator.last_self_attn_map  # [72, 72] numpy float32
    W = matcher_calculator.encoder_feat_size           # 72
    patch_size = matcher_calculator.encoder_patch_size # 14
    resolution = float(matcher_calculator.resolution)  # 1008.0

    attn_flat = torch.from_numpy(self_map).flatten()
    sampled_idx = sample_points_from_attn_map(attn_flat, sampling_method=attn_sampling_mode, k=num_points)
    px = (sampled_idx % W) * patch_size + patch_size // 2
    py = (sampled_idx // W) * patch_size + patch_size // 2
    norm_pts = torch.stack([px.float() / resolution, py.float() / resolution], dim=1)
    print(f"[SelfAttnPoints] sampled {num_points} pts from self-attn map "
          f"(layers={attn_layers}, include_text={include_text_in_prompt}, sampling_method={attn_sampling_mode})")
    return norm_pts.cpu().numpy()  # [num_points, 2]


def get_dense_cross_attn_points(processor=None, support_imgs=None, support_masks=None,
                                 query_img=None, num_points=20, matcher_calculator=None,
                                 text_prompt="visual", skip_coords=False,
                                 attn_layers="all",
                                 inject_into_image_patches=False,
                                 inject_into_support_feats=False,
                                 pooled_text=None,
                                 attention_aggregate_function="sum",
                                 attn_sampling_mode="top-k",
                                 visual_prompt=None, visual_prompt_mask=None,
                                 ):
    """
    Exp 6: Extract prompt points using dense cross-attention between query patches and
    foreground support patches (pre-softmax logits aggregated into a localization heatmap).

    Returns:
        norm_pts [N, 2] (heatmap stored on matcher_calculator)
    """
    if processor is None:
        raise Exception("Processor is not specified")
    if support_imgs is None or support_masks is None:
        raise Exception("Support images and masks are required for dense cross-attention")
    if query_img is None:
        raise Exception("Query image is required for dense cross-attention")
    if matcher_calculator is None:
        raise Exception("Matcher calculator is required for dense cross-attention")

    # Build support visual prompt to condition the cross-attention layers (Exp 6 still
    # uses cross-attn to text+visual; only self-attn is replaced with dense support attn)
    if visual_prompt is None or visual_prompt_mask is None:
        # visual_prompt, visual_prompt_mask = encode_support_visual_tokens(
        #     processor, support_imgs, support_masks, num_points, skip_coords, tag="DenseCA"
        # )
        raise ValueError("Visual prompt and mask are required for dense cross-attention")

    # Run the dense cross-attention pass and extract the heatmap
    heatmap_2d = matcher_calculator.get_dense_cross_attn_map(
        query_img=query_img,
        support_imgs=support_imgs,
        support_masks=support_masks,
        text_prompt=text_prompt,
        skip_coords=skip_coords,
        attn_layers=attn_layers,
        visual_prompt=visual_prompt,
        visual_prompt_mask=visual_prompt_mask,
        inject_into_image_patches=inject_into_image_patches,
        inject_into_support_feats=inject_into_support_feats,
        pooled_text=pooled_text,
        agg_function=attention_aggregate_function,
    )

    H = W = matcher_calculator.encoder_feat_size  # 72
    patch_size = matcher_calculator.encoder_patch_size  # 14
    resolution = float(matcher_calculator.resolution)  # 1008

    heatmap_tensor = torch.from_numpy(heatmap_2d).flatten()  # [5184]

    # if we are not reranking the matcher, we sample points from the heatmap, therefore call sample_points_from_attn_map
    sampled_idx = sample_points_from_attn_map(heatmap_tensor, sampling_method=attn_sampling_mode, k=num_points)
    px = (sampled_idx % W) * patch_size + patch_size // 2
    py = (sampled_idx // W) * patch_size + patch_size // 2
    norm_pts = torch.stack([px.float() / resolution, py.float() / resolution], dim=1)
    print(f"[DenseCrossAttnPoints] sampled {num_points} points from pre-softmax heatmap (mode={attn_sampling_mode}, layers={attn_layers})")
    return norm_pts.numpy()

def _compute_pooled_text(processor, text_prompt):
    """Compute the projected pooled-text bias once per episode. Returns [1, 256]."""
    text_outputs = processor.model.backbone.forward_text([text_prompt], device=processor.device)
    text_feats = text_outputs["language_features"]
    text_mask  = text_outputs["language_mask"]
    encoder = processor.model.transformer.encoder
    pooled_text = pool_text_feat(text_feats, text_mask, encoder.pool_text_with_mask)
    pooled_text = encoder.text_pooling_proj(pooled_text)
    print(f"[PooledText] computed pooled-text bias for prompt='{text_prompt}'")
    return pooled_text

def _resolve_injection(inject_text_pooling, stage, in_prompts_sampling, in_prompts_inference):
    """Resolve the four injection booleans from the master flag + stage + per-stage in_prompts flags."""
    inj_sampling  = bool(inject_text_pooling) and stage in ("point_sampling", "both")
    inj_inference = bool(inject_text_pooling) and stage in ("inference_pass", "both")
    return (
        inj_sampling,
        inj_inference,
        inj_sampling  and bool(in_prompts_sampling),
        inj_inference and bool(in_prompts_inference),
    )


def get_prompt_tokens_from_support(processor=None, support_imgs=None, support_masks=None, query_img=None, skip_coords=False, num_points=20, matcher_calculator=None, text_prompt="visual", use_fused_matcher_features=False, sampling="random", visualize=False, visual_output_path=None, experiment_mode="random", attn_layers="all", inject_text_pooling=False, injection_text_pooling_stage="point_sampling", injection_text_pooling_in_prompts_sampling=False, injection_text_pooling_in_prompts_inference=False, sampling_inputs="both", attention_aggregate_function="sum", attn_sampling_mode="top-k", support_prompt_type="points", blob_selection="largest", sample_points_from_image=False, fixed_first_pts=None):
    if processor is None:
        raise Exception("Processor is not specified")
    if support_imgs is None:
        raise Exception("Support images are not specified")
    if support_masks is None:
        raise Exception("Support masks are not specified")
    _needs_query = experiment_mode in ("matcher", "self_matching", "attn_prior", "dense_cross_attn", "self_attn")
    if query_img is None and _needs_query:
        raise Exception(f"Query image is required for experiment_mode={experiment_mode}")
    if matcher_calculator is None and _needs_query:
        raise Exception(f"matcher_calculator is required for experiment_mode={experiment_mode}")

    if experiment_mode == "dense_cross_attn":
        _mode = f"dense_cross_attn_{attn_sampling_mode}"
    elif experiment_mode == "self_matching":
        _mode = "query_self_matching"
    elif experiment_mode == "attn_prior":
        _mode = f"attn_prior_{attn_sampling_mode}"
    elif experiment_mode == "matcher":
        _mode = "matcher_points"
    elif experiment_mode == "self_attn":
        _mode = f"self_attn ({sampling_inputs})"
    else:
        _mode = "random_support"
    print(f"[PromptTokens] mode='{_mode}' | nshot={support_imgs.shape[0]} | text='{text_prompt}'")

    all_visual_tokens = []
    all_visual_masks = []


    assert support_imgs.shape[0] == support_masks.shape[0]

    # Resolve sampling-pass inputs for experiments that query the fusion encoder
    _sampling_vp, _sampling_vm, _include_text = None, None, True
    support_boxes = None  # list of [cx,cy,w,h] per shot, only set when support_prompt_type=="box"
    if experiment_mode in ("attn_prior", "self_attn"):
        # if the sampling needs support embeddings, run the geometric encoder
        if sampling_inputs in ("both", "support_only"):
            if support_prompt_type == "box":
                _sampling_vp, _sampling_vm, support_boxes = encode_support_box_tokens(
                    processor, support_imgs, support_masks, skip_coords,
                    tag="Sampling", blob_selection=blob_selection
                )
            else:
                _sampling_vp, _sampling_vm = encode_support_visual_tokens(
                    processor, support_imgs, support_masks, num_points, skip_coords, tag="Sampling"
                )
        # if the sampling_inputs is support only, then the text_prompt is "visual" (text must not be disabled in the literal sense, if no text_prompt is given, "visual" is used to tell SAM3 to segment the exemplars)

    # support_only: replace class label with "visual" sentinel in the sampling/matching pass.
    # Applies to attn_prior and self_attn (fusion encoder sampling pass) AND matcher
    # (fusion encoder feature extraction pass when use_fused_matcher_features=True).
    if sampling_inputs == "support_only" and experiment_mode in ("attn_prior", "self_attn", "matcher"):
        text_prompt = "visual"

    # Resolve injection booleans + compute pooled_text once per call (shared across all stages and branches).
    inj_sampling, inj_inference, inj_prompts_sampling, inj_prompts_inference = _resolve_injection(
        inject_text_pooling, injection_text_pooling_stage,
        injection_text_pooling_in_prompts_sampling, injection_text_pooling_in_prompts_inference,
    )
    pooled_text = _compute_pooled_text(processor, text_prompt) if inject_text_pooling else None

    # Sampling-stage support-side bias for Exp 5/7 visual prompt tokens.
    # _sampling_vp shape: [L, 1, 256]; pooled_text [1, 256] → unsqueeze(0) [1, 1, 256] broadcasts onto every token.
    if inj_prompts_sampling and _sampling_vp is not None:
        _sampling_vp = _sampling_vp + pooled_text.unsqueeze(0)

    # All point/box data returned is in normalized [0,1] coordinates
    norm_pts_sampled = None
    all_norm_pts = None
    norm_box = None
    actual_pts_sampled = None

    if experiment_mode == "dense_cross_attn":
        # Exp 6: dense cross-attention heatmap → sample prompt points
        if fixed_first_pts is None:
            result = get_dense_cross_attn_points(
                processor=processor,
                support_imgs=support_imgs,
                support_masks=support_masks,
                visual_prompt=_sampling_vp, visual_prompt_mask=_sampling_vm,
                query_img=query_img,
                num_points=num_points,
                matcher_calculator=matcher_calculator,
                text_prompt=text_prompt,
                skip_coords=skip_coords,
                attn_layers=attn_layers,
                inject_into_image_patches=inj_sampling,
                inject_into_support_feats=inj_prompts_sampling,
                pooled_text=pooled_text,
                attention_aggregate_function=attention_aggregate_function,
                attn_sampling_mode=attn_sampling_mode,
            )
        else:
            result = fixed_first_pts
        pts_norm = result
        if pts_norm is not None:
            # previous experiments showed that skip_coords=False is a lot worse for points on the query image
            # skip_coords=True is usually better when encoding prompts from support to query fusion
            # here we are encoding points from the query image itself
            # state = encode_pts_prompts(processor, query_img, pts_norm, skip_coords)
            state = encode_pts_prompts(processor, query_img, pts_norm, True)
            all_visual_tokens.append(state["prompt"])
            all_visual_masks.append(state["prompt_mask"])
            norm_pts_sampled = pts_norm
            all_norm_pts = pts_norm
        else:
            raise Exception("No points sampled for query image")
    elif experiment_mode == "self_attn":
        # Exp 7: bottom-k points from fusion encoder self-attention map
        if fixed_first_pts is None:
            pts_norm = get_self_attn_points(
                query_img, text_prompt, num_points, matcher_calculator, attn_layers,
                visual_prompt=_sampling_vp, visual_prompt_mask=_sampling_vm,
                # include_text_in_prompt=_include_text,
                attention_aggregate_function=attention_aggregate_function,
                attn_sampling_mode=attn_sampling_mode,
                skip_coords=skip_coords,
                inject_into_image_patches=inj_sampling,
            )
        else:
            pts_norm = fixed_first_pts
        if pts_norm is not None:
            # previous experiments showed that skip_coords=False is a lot worse for points on the query image
            # skip_coords=True is usually better when encoding prompts from support to query fusion
            # here we are encoding points from the query image itself
            # state = encode_pts_prompts(processor, query_img, pts_norm, skip_coords)
            state = encode_pts_prompts(processor, query_img, pts_norm, True)
            all_visual_tokens.append(state["prompt"])
            all_visual_masks.append(state["prompt_mask"])
            norm_pts_sampled = pts_norm
            all_norm_pts = pts_norm
        else:
            raise Exception("No points sampled for query image")
    elif experiment_mode == "attn_prior":
        # Exp 5: top-k points from fusion encoder cross-attention map
        if fixed_first_pts is None:
            pts_norm = get_attn_prior_points(
                query_img, text_prompt, num_points, matcher_calculator, attn_layers,
                visual_prompt=_sampling_vp, visual_prompt_mask=_sampling_vm,
                attention_aggregate_function=attention_aggregate_function,
                attn_sampling_mode=attn_sampling_mode,
                inject_into_image_patches=inj_sampling,
                skip_coords=skip_coords
            )
        else:
            pts_norm = fixed_first_pts
        if pts_norm is not None:
            # previous experiments showed that skip_coords=False is a lot worse for points on the query image
            # skip_coords=True is usually better when encoding prompts from support to query fusion
            # here we are encoding points from the query image itself
            # state = encode_pts_prompts(processor, query_img, pts_norm, skip_coords)
            state = encode_pts_prompts(processor, query_img, pts_norm, True)
            all_visual_tokens.append(state["prompt"])
            all_visual_masks.append(state["prompt_mask"])
            norm_pts_sampled = pts_norm
            all_norm_pts = pts_norm
        else:
            raise Exception("No points sampled for query image")
    elif experiment_mode == "self_matching":
        # Exp 4: query self-matching mode
        pts_norm, all_pts_norm, box_norm = get_query_self_matching_points(
            processor=processor,
            support_imgs=support_imgs,
            support_masks=support_masks,
            query_img=query_img,
            num_points=num_points,
            matcher_calculator=matcher_calculator,
            text_prompt=text_prompt,
            skip_coords=skip_coords,
            sampling=sampling,
            visualize=visualize,
            visual_output_path=visual_output_path
        )
        if pts_norm is not None:
            state = encode_pts_prompts(processor, query_img, pts_norm, True)
            all_visual_tokens.append(state["prompt"])
            all_visual_masks.append(state["prompt_mask"])
            norm_pts_sampled = pts_norm
            all_norm_pts = all_pts_norm
            norm_box = box_norm
        
    elif experiment_mode == "matcher":
        # Standard matcher mode: bipartite point matching support→query
        pts_norm, all_pts_norm, box_norm = get_points_from_matcher(
            support_imgs=support_imgs, support_masks=support_masks, query_img=query_img,
            num_points=num_points, matcher_calculator=matcher_calculator,
            text_prompt=text_prompt, use_fused_matcher_features=use_fused_matcher_features,
            skip_coords=skip_coords, sampling=sampling, visualize=visualize,
            visual_output_path=visual_output_path,
            attn_layers=attn_layers
        )
        if pts_norm is not None:
            state = encode_pts_prompts(processor, query_img, pts_norm, True)
            all_visual_tokens.append(state["prompt"])
            all_visual_masks.append(state["prompt_mask"])
            norm_pts_sampled = pts_norm
            all_norm_pts = all_pts_norm
            norm_box = box_norm
    else:  # "random"
        _random_boxes = []

        for idx in range(support_imgs.shape[0]):
            support_img = support_imgs[idx]
            support_mask = support_masks[idx]

            if support_prompt_type == "box":
                box_cxcywh = get_bbox_from_blob(support_mask, blob_selection=blob_selection)
                if box_cxcywh is None:
                    raise Exception(f"Mask for support image {idx} is empty — no bounding box available.")
                print(f"[PromptTokens] support {idx+1}/{support_imgs.shape[0]} - encoding box {[f'{v:.3f}' for v in box_cxcywh]}")
                state = encode_box_prompts(processor, support_img, box_cxcywh, skip_coords)
                _random_boxes.append(box_cxcywh)
                prompt = state["prompt"]
                mask = state["prompt_mask"]
            else:
                mask = support_mask
                if sample_points_from_image:
                    mask = torch.ones(support_mask.shape, dtype=torch.float32)
                if fixed_first_pts is None:
                    pts_norm, pts_actual = get_random_points_from_mask(mask=mask, num_points=num_points)
                else:
                    pts_norm = fixed_first_pts
                    # Recompute pixel-space coordinates from the fixed normalised points.
                    # pts_norm: [N, 1, 2] floats in [0,1] (x, y); mask shape: (..., H, W).
                    # Must match the shape returned by get_random_points_from_mask: [N, 2] int.
                    _H, _W = support_mask.shape[-2], support_mask.shape[-1]
                    _pts2d = np.asarray(pts_norm)[:, 0, :]  # [N, 2]
                    pts_actual = np.stack(
                        [(_pts2d[:, 0] * _W).astype(int), (_pts2d[:, 1] * _H).astype(int)], axis=1
                    )  # [N, 2]
                if pts_norm is None:
                    raise Exception(f"Mask for support image {idx} is empty. No points have been returned.")
                print(f"[PromptTokens] support {idx+1}/{support_imgs.shape[0]} - encoding {len(pts_norm)} random points from mask")
                state = encode_pts_prompts(processor, support_img, pts_norm, skip_coords)
                norm_pts_sampled = pts_norm
                actual_pts_sampled = pts_actual
                prompt = state["prompt"]
                mask = state["prompt_mask"]

            all_visual_tokens.append(prompt)
            all_visual_masks.append(mask)
        if support_prompt_type == "box":
            support_boxes = _random_boxes
            
    if not all_visual_tokens:
        raise ValueError("No valid support shots found.")

    

    # Aggregate visual knowledge
    aggregated_visual_prompt = torch.cat(all_visual_tokens, dim=0)
    aggregated_visual_mask = torch.cat(all_visual_masks, dim=1)

    # Inference-stage support-side bias: aggregated_visual_prompt has shape [total_seq, 1, 256];
    # pooled_text [1, 256] → unsqueeze(0) [1, 1, 256] broadcasts onto every visual token.
    # Equivalent to per-shot biasing before cat (sum is associative).
    if inj_prompts_inference:
        aggregated_visual_prompt = aggregated_visual_prompt + pooled_text.unsqueeze(0)

    return aggregated_visual_prompt, aggregated_visual_mask, norm_pts_sampled, all_norm_pts, norm_box, actual_pts_sampled, support_boxes, inj_inference

def aggregate_prompt_with_text_tokens(processor=None, state=None, text_prompt="visual", aggregated_visual_prompt=None, aggregated_visual_mask=None):
    if processor is None:
        raise Exception("Processor is not specified")
    elif aggregated_visual_prompt is None or aggregated_visual_mask is None:
        raise Exception("Aggregated visual prompt or mask is not specified")
    elif text_prompt is None:
        raise Exception("Text prompt is not specified")
    
    # generate text tokens
    # if no text_prompt is specified use "visual" as default (used by SAM3 to encode visual knowledge without a text prompt)
    text_outputs = processor.model.backbone.forward_text(
        [text_prompt], device=processor.device
    )
    # text_outputs['language_features'] is usually [L, 1, C]
    # text_outputs['language_mask'] is usually [1, L]
    txt_tokens = text_outputs["language_features"][:, [0]] # select first text prompt
    txt_mask = text_outputs["language_mask"][[0]]

    # combine text tokens with visual tokens
    final_prompt = torch.cat([txt_tokens, aggregated_visual_prompt], dim=0)
    final_mask = torch.cat([txt_mask, aggregated_visual_mask], dim=1)
    
    return final_prompt, final_mask, text_outputs

def update_state_with_support_prompt(state=None, prompt=None, prompt_mask=None, text_outputs=None):
    if state is None:
        raise Exception("State is not specified")
    elif prompt is None or prompt_mask is None:
        raise Exception("Prompt or prompt mask is not specified")
    elif text_outputs is None:
        raise Exception("Text outputs is not specified")
    
    state["prompt"] = prompt
    state["prompt_mask"] = prompt_mask
    state["backbone_out"].update(text_outputs)
    
    return state

def cross_image_prediction(sam3=None, query_frame=None, support_imgs=None, support_masks=None, text_prompt="visual", skip_coords=False, num_points=20, matcher_calculator=None, use_fused_matcher_features=False, sampling="random", visualize=False, visual_output_path=None, experiment_mode="random", attn_layers="all", inject_text_pooling=False, injection_text_pooling_stage="point_sampling", injection_text_pooling_in_prompts_sampling=False, injection_text_pooling_in_prompts_inference=False, sampling_inputs="both", attention_aggregate_function="sum", attn_sampling_mode="top-k", support_prompt_type="points", blob_selection="largest", sample_points_from_image=False, disable_text_inference=False, fixed_first_pts=None, has_dedicated_pass=False, attention_maps_output_dir=None, img_numpy=None, frame_tag=None):
    if sam3 is None:
        raise Exception("SAM3 is not specified")
    elif query_frame is None:
        raise Exception("Query frame is not specified")
    elif support_imgs is None:
        raise Exception("Support images are not specified")
    elif support_masks is None:
        raise Exception("Support masks are not specified")
    elif matcher_calculator is None and experiment_mode in (
        "matcher", "self_matching", "attn_prior", "dense_cross_attn", "self_attn"
    ):
        raise Exception(f"matcher_calculator is required for experiment_mode={experiment_mode}")

    visual_prompt, visual_mask, norm_pts_sampled, all_norm_pts, norm_box, actual_pts_sampled, support_boxes, inj_inference = get_prompt_tokens_from_support(
        processor=sam3.processor,
        support_imgs=support_imgs,
        support_masks=support_masks,
        query_img=query_frame,
        skip_coords=skip_coords,
        num_points=num_points,
        matcher_calculator=matcher_calculator,
        text_prompt=text_prompt,
        use_fused_matcher_features=use_fused_matcher_features,
        sampling=sampling,
        visualize=visualize,
        visual_output_path=visual_output_path,
        experiment_mode=experiment_mode,
        attn_layers=attn_layers,
        sampling_inputs=sampling_inputs,
        attention_aggregate_function=attention_aggregate_function,
        attn_sampling_mode=attn_sampling_mode,
        support_prompt_type=support_prompt_type,
        blob_selection=blob_selection,
        inject_text_pooling=inject_text_pooling,
        injection_text_pooling_stage=injection_text_pooling_stage,
        injection_text_pooling_in_prompts_sampling=injection_text_pooling_in_prompts_sampling,
        injection_text_pooling_in_prompts_inference=injection_text_pooling_in_prompts_inference,
        sample_points_from_image=sample_points_from_image,
        fixed_first_pts=fixed_first_pts
    )

    # save the sampling pass attention maps
    if has_dedicated_pass and matcher_calculator is not None:
        # matcher_calculator.collect_inference_attn(
        #     attn_layers, matcher_calculator.last_num_text_tokens, attention_aggregate_function
        # )
        save_attention_maps(matcher_calculator, attention_maps_output_dir, attn_layers, frame_tag, img_numpy)
        matcher_calculator.arm_inference_capture(attn_layers)

    final_prompt, final_mask, text_outputs = aggregate_prompt_with_text_tokens(
        processor=sam3.processor,
        text_prompt=text_prompt if not disable_text_inference else "visual",
        aggregated_visual_prompt=visual_prompt,
        aggregated_visual_mask=visual_mask
    )
    # Store num_text_tokens so the inference-path capture can split cross-attn by token type
    if matcher_calculator is not None and "language_features" in text_outputs:
        matcher_calculator.last_num_text_tokens = text_outputs["language_features"].shape[0]
    state = sam3.processor.set_image(query_frame)
    state = update_state_with_support_prompt(
        state=state,
        prompt=final_prompt,
        prompt_mask=final_mask,
        text_outputs=text_outputs
    )

    encoder = sam3.processor.model.transformer.encoder
    if inj_inference:
        encoder.inject_text_pooling = True
        print(f"[SAM3_CROSS_IMAGE] Inject text pooling on inference (experiment_mode={experiment_mode})")
    try:
        state = sam3.processor._forward_with_encoded_prompt(state)
    finally:
        encoder.inject_text_pooling = False
    print(f"Found {len(state['scores'])} objects")
    merged_mask = np.any(np.array(state['masks'].cpu()), axis=0).squeeze(0)
    return merged_mask, norm_pts_sampled, all_norm_pts, norm_box, actual_pts_sampled, support_boxes

def save_attention_maps(matcher_calculator=None, vid_attn_dir=None, _sfx=None, frame_tag=0, img_numpy=None):
    if matcher_calculator is None or vid_attn_dir is None or _sfx is None or frame_tag is None or img_numpy is None:
        raise ValueError("Missing arguments for saving attention maps")
    
    for _map_name, _attn_map in [
        ("cross_total",  matcher_calculator.last_cross_attn_map),
        ("cross_text",   matcher_calculator.last_cross_attn_text_map),
        ("cross_points", matcher_calculator.last_cross_attn_points_map),
        ("self",         matcher_calculator.last_self_attn_map),
    ]:
        if _attn_map is not None:
            save_attn_heatmap(
                img_numpy, _attn_map,
                os.path.join(vid_attn_dir, f"frame_{frame_tag}_{_map_name}_{_sfx}.png")
            )

    for _map_name, _attn_map_list in [
        ("self", matcher_calculator.all_self_attn_maps),
        ("cross", matcher_calculator.all_cross_attn_maps),
        ("cross_text", matcher_calculator.all_cross_attn_text_maps),
        ("cross_points", matcher_calculator.all_cross_attn_points_maps),
    ]:
        if _attn_map_list is not None:
            for num_layer, _attn_map in enumerate(_attn_map_list):
                if _attn_map is not None:
                    save_attn_heatmap(
                        img_numpy, _attn_map,
                        os.path.join(vid_attn_dir, f"frame_{frame_tag}_layer_{num_layer}_{_map_name}.png")
                    )

def main():
    args = get_arguments()
    
    validate_args(args)

    # Validate arguments
    print('Running parameters:\n')
    print(json.dumps(vars(args), indent=4, separators=(',', ':')))

    sam3 = SAM3_IMAGE_PREDICTOR(checkpoint=args.checkpoint)

    random.seed(args.seed)
    fix_randseed(args.seed)
            
    matcher_calculator = MatcherBoxCalculator(sam3_model=sam3.model, sam3_processor=sam3.processor)

    # create the dataset from the builder
    loader = ImageDataset(args.benchmark, args)
    # dataloader = DataLoader(
    #     loader,
    #     batch_size=None,
    #     num_workers=1,
    #     shuffle=False
    # )

    dataset = loader.dataset

    # ── Shard: slice class list into num_shards disjoint parts ───────────────
    # Sort deterministically so every shard gets a stable, reproducible subset.
    # Slicing is done on the *class* axis (not the raw item index) so that every
    # image of a given class always belongs to exactly one shard.
    class_list = sorted(dataset.get_class_ids())
    if args.num_shards > 1:
        if not (0 <= args.shard_id < args.num_shards):
            raise ValueError(
                f"--shard_id {args.shard_id} is out of range for --num_shards {args.num_shards}"
            )
        total_classes = len(class_list)
        # Compute inclusive [start_c, end_c) slice for this shard.
        # Integer division distributes remainder classes to the first shards.
        base, remainder = divmod(total_classes, args.num_shards)
        start_c = args.shard_id * base + min(args.shard_id, remainder)
        end_c   = start_c + base + (1 if args.shard_id < remainder else 0)
        class_list = class_list[start_c:end_c]
        print(
            f"[Sharding] shard {args.shard_id}/{args.num_shards}: "
            f"classes {start_c}..{end_c-1} of {total_classes} total "
            f"({len(class_list)} classes)"
        )
        # Auto-suffix session_name so each shard writes to its own log dir.
        if args.session_name is not None:
            args.session_name = f"{args.session_name}_shard{args.shard_id}of{args.num_shards}"
        else:
            args.session_name = f"shard{args.shard_id}of{args.num_shards}"
    # Set of class IDs to process in this shard (O(1) membership test in the loop)
    shard_class_set = set(class_list)
    # ─────────────────────────────────────────────────────────────────────────

    class_dic = dataset.idx_to_classname
    grond_truth_class_dic = dataset.idx_to_ground_truth_label
    original_class_dic = dict(class_dic)  # Save before overwriting
    if args.all_lemmas:
        # Build mapping: original class_id → [(virtual_id, lemma_text), ...]
        class_id_to_virtual = {}
        virtual_class_list = []
        virtual_class_dic = {}
        virtual_to_original = {}
        vid = 0
        for cid in class_list:
            lemmas = dataset.class_idx_to_all_lemmas.get(cid, [class_dic[cid]])
            class_id_to_virtual[cid] = []
            for lemma in lemmas:
                class_id_to_virtual[cid].append((vid, lemma))
                virtual_class_list.append(vid)
                virtual_class_dic[vid] = lemma
                virtual_to_original[vid] = cid
                vid += 1
        evaluator = Evaluator(class_list=virtual_class_list)
        class_dic = virtual_class_dic
    else:
        virtual_to_original = None
        evaluator = Evaluator(class_list=class_list)

    os.makedirs(args.output_dir, exist_ok=True)

    start_time = time.perf_counter()

    box_coordinates = []
    # Accumulates one dict per (sample_id, eval_id) for the point_features CSV.
    point_feat_rows = []
    
    print("STARTING SEGMENTATION")
    print("-" * 50)
    with torch.no_grad():
        for idx in range(len(loader)):
            # Deterministic seed set BEFORE data loading so support image sampling
            # (np.random.choice inside load_frame) is identical across all experiments.
            current_seed = args.seed + (args.run_n * 10000) + idx
            random.seed(current_seed)
            np.random.seed(current_seed)
            torch.manual_seed(current_seed)

            data = loader[idx]

            # Skip items whose class does not belong to this shard.
            if data['class_id'] not in shard_class_set:
                continue

            query_imgs = data['query_imgs']
            query_masks = data['query_masks']
            support_imgs = data['support_imgs']
            support_masks = data['support_masks']
            begin_new = data['begin_new']
            class_id = data['class_id']
            class_name = data['class_name']
            dir_name = data['dir_name']
            chosen_frames = data['chosen_frames']

            print(f"query_imgs shape: {query_imgs.shape}")
            print(f"query_masks shape: {query_masks.shape}")

            if support_imgs is not None and support_masks is not None:
                print(f"support_imgs shape: {support_imgs.shape}")
                print(f"support_masks shape: {support_masks.shape}")

            # assert len(query_imgs) == len(query_masks)
            assert query_imgs.shape[0] == query_masks.shape[0]

            # class_name = dataset.idx_to_classname[class_id]
            print(f"Segmenting image {dir_name} on class '{class_name}' with id {class_id}")
            
            # Determine lemma entries for this image
            if args.all_lemmas:
                lemma_entries = class_id_to_virtual[class_id]
            else:
                lemma_entries = [(class_id, class_name)]

            # Always place the canonical selected lemma first whenever all_lemmas is
            # active. This guarantees a consistent virtual-class ordering across all
            # experiment types (FREE, FIXED, random, attention-based, etc.) so that
            # the selected lemma always occupies the lowest virtual ID and the first
            # iteration of the inner lemma loop. For attention experiments this is
            # critical: the first lemma's text prompt determines the attention map and
            # therefore the sampled point coordinates, so a different ordering produces
            # different points. For random/matcher experiments the ordering is a no-op
            # in practice (points are drawn from the mask, not from the lemma text),
            # but consistent ordering is still preferable for reproducibility.
            if args.all_lemmas:
                selected_lemma_text = original_class_dic[class_id]
                lemma_entries = sorted(
                    lemma_entries,
                    key=lambda e: 0 if e[1] == selected_lemma_text else 1
                )
                print(f"  [Lemma ordering] Selected lemma '{selected_lemma_text}' placed first")
                assert lemma_entries[0][1] == selected_lemma_text, (
                    f"[BUG] Selected lemma '{selected_lemma_text}' is NOT first for class {class_id} "
                    f"(original name: '{original_class_dic.get(class_id, '?')}')! "
                    f"First entry is '{lemma_entries[0][1]}'. "
                    f"Full order: {[e[1] for e in lemma_entries]}"
                )

            first_pts = None
            for eval_id, lemma in lemma_entries:
                if args.all_lemmas:
                    print(f"  Prompting with lemma: '{lemma}' (eval_id: {eval_id})")

                vid_output_dir = os.path.join(args.output_dir, f"{dir_name}_{eval_id}_{idx}", "output")
                vid_ground_truth_dir = os.path.join(args.output_dir, f"{dir_name}_{eval_id}_{idx}", "ground_truth")
                vid_frames_dir = os.path.join(args.output_dir, f"{dir_name}_{eval_id}_{idx}", "frames")
                vid_attn_dir = os.path.join(args.output_dir, f"{dir_name}_{eval_id}_{idx}", "attention_maps")
                vid_attn_sampling_dir = os.path.join(vid_attn_dir, f"sampling_maps")
                vid_attn_inference_dir = os.path.join(vid_attn_dir, f"inference_maps")
                if args.experiment_mode == "matcher":
                    vid_box_dir = os.path.join(args.output_dir, f"{dir_name}_{eval_id}_{idx}", "bounding_box")

                os.makedirs(vid_output_dir, exist_ok=True)
                os.makedirs(vid_ground_truth_dir, exist_ok=True)
                os.makedirs(vid_frames_dir, exist_ok=True)
                os.makedirs(vid_attn_dir, exist_ok=True)
                os.makedirs(vid_attn_sampling_dir, exist_ok=True)
                os.makedirs(vid_attn_inference_dir, exist_ok=True)
                if args.experiment_mode == "matcher":
                    os.makedirs(vid_box_dir, exist_ok=True)

                predictions = []
                ground_truths = []
                point_scores_list = []
                all_point_scores_list = []
                # Per-frame geometry feature accumulator for this (sample, lemma).
                # Each entry is a feat-dict from compute_point_features().
                _frame_feat_list = []

                # Save support images and their mask overlays
                if support_imgs is not None:
                    for s_idx in range(support_imgs.shape[0]):
                        s_img_numpy = (support_imgs[s_idx] * 255).permute(1, 2, 0).to(torch.uint8).cpu().numpy()
                        if support_masks is not None:
                            s_mask = support_masks[s_idx].squeeze()
                            save_mask_overlay(s_img_numpy, s_mask, os.path.join(vid_frames_dir, f"support_{s_idx}_overlay.png"))

                for frame_idx in range(len(query_imgs)):
                    print(f"Processing frame {chosen_frames[frame_idx]}")

                    # img_pil = Image.fromarray((query_imgs[frame_idx] * 255).astype(np.uint8))
                    query_frame = query_imgs[frame_idx]
                    img_numpy = (query_frame * 255).permute(1, 2, 0).to(torch.uint8).cpu().numpy()
                    ground_truth = query_masks[frame_idx].squeeze()
                    print(f"query_frame shape: {query_frame.shape}")
                    print(f"ground_truth shape: {ground_truth.shape}")
                    
                    _text = lemma
                    if args.use_query_as_support:
                        _sup_imgs  = query_frame.unsqueeze(0)
                        _sup_masks = ground_truth.unsqueeze(0)
                    else:
                        _sup_imgs  = support_imgs
                        _sup_masks = support_masks

                    # has_dedicated_pass == True if the experiment requires a sampling pass. 
                    # Already set the SAM3 fusion encoder layers to store attention maps
                    # If instead the exp requires the sampling, the storing and saving is delegated to the sampling function in "cross_image_prediction"
                    has_dedicated_pass = args.experiment_mode in (
                        "attn_prior", "dense_cross_attn", "self_attn"
                    ) or (args.experiment_mode == "matcher" and args.use_fused_matcher_features)

                    if not has_dedicated_pass:
                        matcher_calculator.arm_inference_capture(args.attn_layers)

                    # Execution of the main pipeline: evaluates the query image using the support prompt.
                    # Variables returned:
                    # - prediction: the final binary segmentation mask [H, W] predicted for the query image
                    # - norm_pts_sampled: [num_points, 1, 2] the selected subset of prompt points in [0, 1] normalized coords
                    # - all_norm_pts: [M, 2] all candidate points evaluated before sampling (e.g. from matcher), in [0, 1] coords
                    # - norm_box: [4] [x1, y1, x2, y2] bounding box in [0, 1] normalized coords (generated by the matcher)
                    # - actual_pts_sampled: [num_points, 2] exact pixel coordinates [0, H] if sampled directly from a mask, else None
                    prediction, norm_pts_sampled, all_norm_pts, norm_box, actual_pts_sampled, support_boxes = cross_image_prediction(
                        sam3=sam3,
                        query_frame=query_frame,
                        support_imgs=_sup_imgs,
                        support_masks=_sup_masks,
                        text_prompt=_text,
                        skip_coords=args.skip_coords,
                        num_points=args.num_points_from_mask,
                        matcher_calculator=matcher_calculator,
                        use_fused_matcher_features=args.use_fused_matcher_features,
                        sampling=args.sampling,
                        visualize=args.visualize_embeddings,
                        visual_output_path=os.path.join(vid_box_dir, f"frame_0_tsne.png") if args.visualize_embeddings and args.experiment_mode == "matcher" else None,
                        experiment_mode=args.experiment_mode,
                        attn_layers=args.attn_layers,
                        inject_text_pooling=args.inject_text_pooling,
                        injection_text_pooling_stage=args.injection_text_pooling_stage,
                        injection_text_pooling_in_prompts_sampling=args.injection_text_pooling_in_prompts_sampling,
                        injection_text_pooling_in_prompts_inference=args.injection_text_pooling_in_prompts_inference,
                        sampling_inputs=args.sampling_inputs,
                        attention_aggregate_function=args.attention_aggregate_function,
                        attn_sampling_mode=args.attn_sampling_mode,
                        support_prompt_type=args.support_prompt_type,
                        blob_selection=args.blob_selection,
                        sample_points_from_image=args.sample_points_from_image,
                        disable_text_inference=args.disable_text_inference,
                        fixed_first_pts=first_pts if args.fix_sampled_points else None,
                        has_dedicated_pass=has_dedicated_pass,
                        attention_maps_output_dir=vid_attn_sampling_dir,
                        img_numpy=img_numpy,
                        frame_tag=chosen_frames[frame_idx]
                    )

                    if args.fix_sampled_points and first_pts is None:
                        first_pts = norm_pts_sampled
                        print(f"  [fix_sampled_points] Captured first_pts from lemma '{lemma}' (shape: {first_pts.shape if first_pts is not None else None})")

                    # Save support bbox visualizations (one image per support shot)
                    if support_boxes is not None:
                        for s_idx, box_cxcywh in enumerate(support_boxes):
                            if box_cxcywh is not None:
                                s_img_numpy = (_sup_imgs[s_idx] * 255).permute(1, 2, 0).to(torch.uint8).cpu().numpy()
                                cx, cy, bw, bh = box_cxcywh
                                norm_xyxy = [cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2]
                                H_s, W_s = s_img_numpy.shape[:2]
                                pixel_box = rescale_to_pixel(norm_xyxy, (W_s, H_s))
                                save_image_with_box(s_img_numpy, pixel_box, os.path.join(vid_frames_dir, f"support_{s_idx}_bbox.png"))

                    # (if we had a dedicated pass to compute the points to prompt the query image, we'd overwrite the attn maps with the ones of the inference pass is instead we just did an inference or matcher pass, we need to collect the attention maps from the inference pass)
                    # This save is going to change to be dedicated for the saving of the attention maps of the inference pass (the inference pass is going to be always saved, the optional one becomes the sampling pass). We can assume the sampling pass attention maps have already been saved, so we can collect the inference attention maps right away, regardless of the experiment mode.
                    matcher_calculator.collect_inference_attn(
                        args.attn_layers, matcher_calculator.last_num_text_tokens, args.attention_aggregate_function
                    )

                    img_pil = Image.fromarray(img_numpy)
                    save_image(img_pil, os.path.join(vid_frames_dir, f"frame_{chosen_frames[frame_idx]}_input.png"))
                    save_image(prediction, os.path.join(vid_output_dir, f"frame_{chosen_frames[frame_idx]}.png"))
                    save_image(ground_truth, os.path.join(vid_ground_truth_dir, f"frame_{chosen_frames[frame_idx]}.png"))

                    save_mask_overlay(img_pil, prediction, os.path.join(vid_output_dir, f"frame_{chosen_frames[frame_idx]}_overlay.png"))
                    save_mask_overlay(img_pil, ground_truth, os.path.join(vid_ground_truth_dir, f"frame_{chosen_frames[frame_idx]}_overlay.png"))

                    actual_h, actual_w = img_numpy.shape[:2]
                    pixel_size = (actual_w, actual_h)

                    # Single rescale from [0,1] → pixels, only when we have query-side points
                    # has_query_pts: True if the extracted prompt points reside on the QUERY image coordinate space.
                    # It is False for the "random" baseline where points are simply sampled from the SUPPORT image mask.
                    has_query_pts = args.experiment_mode in (
                        "matcher", "self_matching", "attn_prior", "dense_cross_attn", "self_attn"
                    ) or args.use_query_as_support
                    
                    # rescaled_sampled_pts: the final [num_points, 2] absolute pixel coordinates for the selected prompt points
                    if actual_pts_sampled is not None:
                        # We already have exact pixel coordinates (e.g., from reading a support mask directly)
                        rescaled_sampled_pts = actual_pts_sampled if has_query_pts else None
                    else:
                        # Otherwise we scale the [0, 1] normalized coordinates back to the original image dimensions
                        rescaled_sampled_pts = rescale_to_pixel(norm_pts_sampled, pixel_size) if has_query_pts else None

                    # rescaled_all_pts: the full set of candidate pixel coordinates (before taking top-k or core-set sampling)
                    rescaled_all_pts = rescale_to_pixel(all_norm_pts, pixel_size) if all_norm_pts is not None else None
                    
                    # rescaled_box: the predicted bounding box scaled to pixel coordinates [x1, y1, x2, y2]
                    rescaled_box = rescale_to_pixel(norm_box, pixel_size) if norm_box is not None else None

                    if args.experiment_mode == "matcher":
                        if rescaled_all_pts is not None and rescaled_sampled_pts is not None:
                            save_image_with_all_and_sampled_points(img_numpy, rescaled_all_pts, rescaled_sampled_pts, os.path.join(vid_box_dir, f"frame_{chosen_frames[frame_idx]}_matcher_points.png"))

                        if rescaled_box is not None:
                            save_image_with_box(img_numpy, rescaled_box, os.path.join(vid_box_dir, f"frame_{chosen_frames[frame_idx]}_matcher_box.png"))

                    # Save all attention maps for this frame
                    frame_tag = chosen_frames[frame_idx]
                    _sfx = args.attn_layers
                    save_attention_maps(matcher_calculator, vid_attn_inference_dir, _sfx, frame_tag, img_numpy)
                    # for _map_name, _attn_map in [
                    #     ("cross_total",  matcher_calculator.last_cross_attn_map),
                    #     ("cross_text",   matcher_calculator.last_cross_attn_text_map),
                    #     ("cross_points", matcher_calculator.last_cross_attn_points_map),
                    #     ("self",         matcher_calculator.last_self_attn_map),
                    # ]:
                    #     if _attn_map is not None:
                    #         save_attn_heatmap(
                    #             img_numpy, _attn_map,
                    #             os.path.join(vid_attn_dir, f"frame_{frame_tag}_{_map_name}_{_sfx}.png")
                    #         )

                    # for _map_name, _attn_map_list in [
                    #     ("self", matcher_calculator.all_self_attn_maps),
                    #     ("cross", matcher_calculator.all_cross_attn_maps),
                    #     ("cross_text", matcher_calculator.all_cross_attn_text_maps),
                    #     ("cross_points", matcher_calculator.all_cross_attn_points_maps),
                    # ]:
                    #     if _attn_map_list is not None:
                    #         for num_layer, _attn_map in enumerate(_attn_map_list):
                    #             if _attn_map is not None:
                    #                 save_attn_heatmap(
                    #                     img_numpy, _attn_map,
                    #                     os.path.join(vid_attn_dir, f"frame_{frame_tag}_layer_{num_layer}_{_map_name}.png")
                    #                 )

                    # Points overlay on the visual/points-prior map
                    if rescaled_sampled_pts is not None and matcher_calculator.last_cross_attn_points_map is not None:
                        save_image_with_all_and_sampled_points(
                            img_numpy, rescaled_all_pts, rescaled_sampled_pts,
                            os.path.join(vid_attn_dir, f"frame_{frame_tag}_cross_points_{_sfx}_sampled.png")
                        )

                    # Point score evaluation
                    if rescaled_sampled_pts is not None:
                        point_scores_list.append(rescaled_sampled_pts)
                    else:
                        point_scores_list.append(None)
                    
                    if args.experiment_mode in ("matcher", "attn_prior") and rescaled_all_pts is not None:
                        all_point_scores_list.append(rescaled_all_pts)
                    else:
                        all_point_scores_list.append(None)
                    
                    # Record normalized box for CSV output
                    if norm_box is not None:
                        box_coordinates.append(convert_norm_box_to_sam3_format(norm_box))

                    # ── Point-feature logging (inline, same coordinate frame) ────────────
                    # `ground_truth`        : original-image H×W GT mask (from query_masks)
                    # `rescaled_sampled_pts`: positive prompt points scaled to the same
                    #                         original-image pixel frame (actual_w × actual_h)
                    # Both variables are already in the original-image pixel coordinate
                    # system at this point in the loop, so no further reprojection needed.
                    _gt_for_feat = ground_truth.cpu().numpy() if hasattr(ground_truth, 'cpu') else np.array(ground_truth)
                    _pts_for_feat = rescaled_sampled_pts  # [N,2] or None
                    _feat = compute_point_features(_gt_for_feat, _pts_for_feat)
                    _frame_feat_list.append(_feat)

                    predictions.append(prediction)
                    ground_truths.append(ground_truth)

                evaluator.update_evl(eval_id, ground_truths, predictions, sample_id=f"{dir_name}_{eval_id}_{idx}", points_list=point_scores_list, all_points_list=all_point_scores_list)
                print(f"Updated evaluation metrics for '{lemma}'")

                # ── Aggregate per-frame geometry features → one row per (sample, lemma) ──
                # j_score is now available in evaluator.sample_details for this eval_id.
                _class_internal_id = evaluator.class_indexes.index(eval_id)
                _last_detail = evaluator.sample_details[_class_internal_id][-1]
                _sample_j_score = float(_last_detail["j_score"])

                if _frame_feat_list:
                    # Average numeric geometry features across frames; keep
                    # points_xy from the first frame (representative).
                    _avg_feat = {}
                    _numeric_keys = [
                        "n_points", "n_neg_points", "frac_offmask",
                        "object_radius_px",
                        "coverage_gap_mean", "coverage_gap_p95",
                        "dt_depth_mean", "dt_depth_min",
                        "dispersion_norm", "centroid_offset_norm",
                    ]
                    for _k in _numeric_keys:
                        _vals = [f[_k] for f in _frame_feat_list
                                 if not (isinstance(f[_k], float) and np.isnan(f[_k]))]
                        _avg_feat[_k] = float(np.mean(_vals)) if _vals else float("nan")
                    _avg_feat["points_xy"] = _frame_feat_list[0]["points_xy"]

                    _orig_cid = virtual_to_original.get(eval_id, eval_id) if virtual_to_original else eval_id
                    _orig_cname = original_class_dic.get(_orig_cid, "") if original_class_dic else ""

                    _row = {
                        "class_id":            eval_id,
                        "class_name":          lemma,
                        "sample_id":           f"{dir_name}_{eval_id}_{idx}",
                        "original_class_idx":  _orig_cid,
                        "original_class_name": _orig_cname,
                        "fold":                args.fold,
                        "run":                 args.run_n,
                        "j_score":             _sample_j_score,
                        "points_xy":           _avg_feat["points_xy"],
                        "n_points":            int(_avg_feat["n_points"]),
                        "n_neg_points":        int(_avg_feat["n_neg_points"]),
                        "frac_offmask":        _avg_feat["frac_offmask"],
                        "object_radius_px":    _avg_feat["object_radius_px"],
                        "coverage_gap_mean":   _avg_feat["coverage_gap_mean"],
                        "coverage_gap_p95":    _avg_feat["coverage_gap_p95"],
                        "dt_depth_mean":       _avg_feat["dt_depth_mean"],
                        "dt_depth_min":        _avg_feat["dt_depth_min"],
                        "dispersion_norm":     _avg_feat["dispersion_norm"],
                        "centroid_offset_norm": _avg_feat["centroid_offset_norm"],
                    }
                    point_feat_rows.append(_row)


            if (idx + 1) % 10 == 0:
                current_time = time.perf_counter()
                elapsed_so_far = current_time - start_time
                avg_time_per_img = elapsed_so_far / (idx + 1)
                print(f">>> Processate {idx + 1} immagini in {elapsed_so_far:.2f} sec (Media: {avg_time_per_img:.2f} sec/img)")
    print("-" * 50)

    end_time = time.perf_counter()
    total_time = end_time - start_time
    
    print("-" * 50)
    print(f"SEGMENTAZIONE COMPLETATA IN: {total_time:.2f} secondi")
    print(f"Tempo medio finale: {total_time / (idx + 1):.2f} secondi per immagine")
    
    mean_iou = np.mean(evaluator.iou_list)
    str_mean_iou = 'J: %.8f ' % (mean_iou)

    # Generate dictionary with class id as key and f_score, j_score, point_accuracy as values
    scoring_class_list = virtual_class_list if args.all_lemmas else class_list
    score_dict = {
        cid: {
            "iou_score": float(iou),
            "point_accuracy_micro": float(evaluator.pt_accuracy[ic]),
            "all_point_accuracy_micro": float(evaluator.all_point_accuracy[ic]),
            "point_accuracy_macro": float(evaluator.pt_accuracy_macro[ic]),
            "all_point_accuracy_macro": float(evaluator.all_point_accuracy_macro[ic])
        }
        for ic, (cid, iou) in enumerate(zip(scoring_class_list, evaluator.iou_list))
    }

    clean_score_dict = {k: v.tolist() if hasattr(v, 'tolist') else v for k, v in score_dict.items()}

    save_results(class_dic, evaluator, args, virtual_to_original=virtual_to_original, original_class_dic=original_class_dic, box_coordinates=box_coordinates, grond_truth_class_dic=grond_truth_class_dic)
    save_point_features_csv(point_feat_rows, args)

    # PRINT RESULTS
    print("\n\n")
    print("-" * 50)
    
    # Calculate mean point scores excluding classes where no points were sampled
    valid_pt_micro = [acc for ic, acc in enumerate(evaluator.pt_accuracy) if evaluator.pt_total_list[ic] > 0]
    mean_pt_micro = np.mean(valid_pt_micro) if valid_pt_micro else -1.0
    
    valid_pt_macro = [acc for ic, acc in enumerate(evaluator.pt_accuracy_macro) if evaluator.pt_ratios_count[ic] > 0]
    mean_pt_macro = np.mean(valid_pt_macro) if valid_pt_macro else -1.0
    
    valid_all_pt_micro = [acc for ic, acc in enumerate(evaluator.all_point_accuracy) if evaluator.all_pt_total_list[ic] > 0]
    mean_all_pt_micro = np.mean(valid_all_pt_micro) if valid_all_pt_micro else -1.0

    valid_all_pt_macro = [acc for ic, acc in enumerate(evaluator.all_point_accuracy_macro) if evaluator.all_pt_ratios_count[ic] > 0]
    mean_all_pt_macro = np.mean(valid_all_pt_macro) if valid_all_pt_macro else -1.0

    print(f"Fold {args.fold} - Mean IoU: {mean_iou}")
    if mean_pt_micro != -1.0:
        print(f"Mean Point Accuracy (Sampled - Micro per class): {mean_pt_micro:.4f}")
        print(f"Mean Point Accuracy (Sampled - Macro per class): {mean_pt_macro:.4f}")
    if mean_all_pt_micro != -1.0:
        print(f"Mean Point Accuracy (Total - Micro per class): {mean_all_pt_micro:.4f}")
        print(f"Mean Point Accuracy (Total - Macro per class): {mean_all_pt_macro:.4f}")
    print(f"Detailed Scores: {json.dumps(clean_score_dict, indent=4)}")


if __name__ == '__main__':
    main()