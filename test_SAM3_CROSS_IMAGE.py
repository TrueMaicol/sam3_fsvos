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

    if em == "self_attn":
        if args.sampling_inputs == "support_only" and not args.nshot > 0:
            raise Exception(
                "--experiment_mode=self_attn with --sampling_inputs=support_only requires --nshot > 0"
            )

    if args.sampling_inputs != "both" and em not in ("attn_prior", "self_attn"):
        raise Exception("--sampling_inputs only applies to --experiment_mode=attn_prior or self_attn")


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
        parser.add_argument("--synset_mapping_folder_path", type=str, default="/leonardo_work/IscrC_MARSv2/datasets/synset_mappings")
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
                "'support_only': support visual prompts only (no text tokens in cross-attention)."
            ))
        parser.add_argument("--run_n", type=int, default=0)
        parser.add_argument("--skip_coords", action="store_true", default=False, help="Skip coordinate-based embeddings when generating prompt tokens from support images")
        parser.add_argument("--use_fused_matcher_features", action="store_true", default=False, help="Use fused features from the fusion encoder instead of native PE backbone features for matcher (only relevant for experiment_mode=matcher).")
        parser.add_argument("--attn_layers", type=str, default="last", choices=["last", "all"], help="Which fusion encoder layers to aggregate attention from for both point selection and saving")
        parser.add_argument("--attention_aggregate_function", type=agg_function_arg, default="sum", help="Aggregation function to be used to aggregate attention matrices on the key dimension")
        parser.add_argument("--attn_sampling_mode", type=str, default="top-k", choices=["top-k", "bottom-k"], help="Point sampling method for attention priors")
        
        # Exp 6: Dense cross-attention sub-options
        parser.add_argument("--dense_cross_attn_skip_text_injection", action="store_true", default=False, help="Exp 6: Skip the pooled-text→image-patch injection before the Fusion Encoder, so query features are purely visual during the dense cross-attention pass")
        
        #
        parser.add_argument("--num_points_from_mask", type=int, default=20)
        parser.add_argument("--use_query_as_support", action="store_true", default=False, help="Use the query image as support image (only for 1-shot)")
        parser.add_argument("--disable_text", action="store_true", default=False, help="Disable text prompts")
        parser.add_argument("--sampling", type=str, default="random", choices=["random", "top-k", "patch-core", "k-means-embeddings", "k-means-points", "k-medoids-embeddings", "k-medoids-points"], help="Sampling strategy for Matcher points")
        parser.add_argument("--visualize_embeddings", action="store_true", default=False, help="Generate t-SNE plots of the embeddings")
        # Random state management
        parser.add_argument('--seed', type=int, default=0)

        # Loggin arguments
        parser.add_argument('--log_dir', type=str, default='/leonardo_work/IscrC_MARSv2/SAM3_FSVOS/JOB_OUTPUT/logs')
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

def save_results(class_dic, evaluator, args, virtual_to_original=None, original_class_dic=None, box_coordinates=None, point_scores=None):
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

    size_data = []
    for label, scores in [('SMALL', evaluator.iou_small_score), ('MEDIUM', evaluator.iou_medium_score), ('LARGE', evaluator.iou_large_score)]:
        for cid, cname, s in zip(class_list_idx, class_list_names, scores):
            row = {'size': label, 'class_id': cid, 'class_name': cname, 'score': float(s)}
            orig_cid, orig_cname = _orig(cid)
            if orig_cid is not None:
                row['original_class_idx'] = orig_cid
                row['original_class_name'] = orig_cname
            size_data.append(row)
    size_df = pd.DataFrame(size_data)
    size_csv_path = os.path.join(results_dir, f"{args.benchmark}_size_scores.csv")
    size_df.to_csv(size_csv_path, index=False, sep=';')
    
    sample_data = []
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
            orig_cid, orig_cname = _orig(class_idx)
            if orig_cid is not None:
                row['original_class_idx'] = orig_cid
                row['original_class_name'] = orig_cname
            sample_data.append(row)
    sample_df = pd.DataFrame(sample_data)
    sample_csv_path = os.path.join(results_dir, f"{args.benchmark}_sample_scores.csv")
    sample_df.to_csv(sample_csv_path, index=False, sep=';')

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


def get_points_from_matcher(support_imgs=None, support_masks=None, query_img=None, num_points=20, matcher_calculator=None, text_prompt="visual", use_fused_matcher_features=False, skip_coords=False, sampling="random", visualize=False, visual_output_path=None, reference_visual_prompt=None, reference_visual_mask=None, attn_layers="last"):
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
        reference_visual_mask=aggregated_visual_mask
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
                           visual_prompt=None, visual_prompt_mask=None, attention_aggregate_function="sum", attn_sampling_mode="top-k"):
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
    )
    # Point selection uses attention to visual/point tokens (not text)
    pts_map = matcher_calculator.last_cross_attn_points_map  # [72, 72] numpy
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
                                  include_text_in_prompt=True, attention_aggregate_function="sum", attn_sampling_mode="bottom-k"):
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
                                 skip_text_injection=False,
                                 attention_aggregate_function="sum",
                                 attn_sampling_mode="top-k"):
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
    agg_support_prompt, agg_support_mask = encode_support_visual_tokens(
        processor, support_imgs, support_masks, num_points, skip_coords, tag="DenseCA"
    )

    # Run the dense cross-attention pass and extract the heatmap
    heatmap_2d = matcher_calculator.get_dense_cross_attn_map(
        query_img=query_img,
        support_imgs=support_imgs,
        support_masks=support_masks,
        text_prompt=text_prompt,
        skip_coords=skip_coords,
        attn_layers=attn_layers,
        visual_prompt=agg_support_prompt,
        visual_prompt_mask=agg_support_mask,
        skip_text_injection=skip_text_injection,
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


def get_prompt_tokens_from_support(processor=None, support_imgs=None, support_masks=None, query_img=None, skip_coords=False, num_points=20, matcher_calculator=None, text_prompt="visual", use_fused_matcher_features=False, sampling="random", visualize=False, visual_output_path=None, experiment_mode="random", attn_layers="last", dense_cross_attn_skip_text_injection=False, sampling_inputs="both", attention_aggregate_function="sum", attn_sampling_mode="top-k"):
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
    if experiment_mode in ("attn_prior", "self_attn"):
        # if the sampling needs support point embeddings, run the geometric encoder
        if sampling_inputs in ("both", "support_only"):
            _sampling_vp, _sampling_vm = encode_support_visual_tokens(
                processor, support_imgs, support_masks, num_points, skip_coords, tag="Sampling"
            )
        _include_text = (sampling_inputs != "support_only")

    # All point/box data returned is in normalized [0,1] coordinates
    norm_pts_sampled = None
    all_norm_pts = None
    norm_box = None
    actual_pts_sampled = None

    if experiment_mode == "dense_cross_attn":
        # Exp 6: dense cross-attention heatmap → sample prompt points
        result = get_dense_cross_attn_points(
            processor=processor,
            support_imgs=support_imgs,
            support_masks=support_masks,
            query_img=query_img,
            num_points=num_points,
            matcher_calculator=matcher_calculator,
            text_prompt=text_prompt,
            skip_coords=skip_coords,
            attn_layers=attn_layers,
            skip_text_injection=dense_cross_attn_skip_text_injection,
            sampling=sampling,
            visualize=visualize,
            visual_output_path=visual_output_path,
            attention_aggregate_function=attention_aggregate_function,
            attn_sampling_mode=attn_sampling_mode,
        )
        pts_norm = result
        if pts_norm is not None:
            state = encode_pts_prompts(processor, query_img, pts_norm, skip_coords)
            all_visual_tokens.append(state["prompt"])
            all_visual_masks.append(state["prompt_mask"])
            norm_pts_sampled = pts_norm
            all_norm_pts = pts_norm
        
    elif experiment_mode == "self_attn":
        # Exp 7: bottom-k points from fusion encoder self-attention map
        pts_norm = get_self_attn_points(
            query_img, text_prompt, num_points, matcher_calculator, attn_layers,
            visual_prompt=_sampling_vp, visual_prompt_mask=_sampling_vm,
            include_text_in_prompt=_include_text,
            attention_aggregate_function=attention_aggregate_function,
            attn_sampling_mode=attn_sampling_mode,
        )
        if pts_norm is not None:
            state = encode_pts_prompts(processor, query_img, pts_norm, skip_coords)
            all_visual_tokens.append(state["prompt"])
            all_visual_masks.append(state["prompt_mask"])
            norm_pts_sampled = pts_norm
            all_norm_pts = pts_norm
    elif experiment_mode == "attn_prior":
        # Exp 5: top-k points from fusion encoder cross-attention map
        pts_norm = get_attn_prior_points(
            query_img, text_prompt, num_points, matcher_calculator, attn_layers,
            visual_prompt=_sampling_vp, visual_prompt_mask=_sampling_vm,
            attention_aggregate_function=attention_aggregate_function,
            attn_sampling_mode=attn_sampling_mode,
        )
        if pts_norm is not None:
            state = encode_pts_prompts(processor, query_img, pts_norm, skip_coords)
            all_visual_tokens.append(state["prompt"])
            all_visual_masks.append(state["prompt_mask"])
            norm_pts_sampled = pts_norm
            all_norm_pts = pts_norm
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
            state = encode_pts_prompts(processor, query_img, pts_norm, skip_coords)
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
        )
        if pts_norm is not None:
            state = encode_pts_prompts(processor, query_img, pts_norm, skip_coords)
            all_visual_tokens.append(state["prompt"])
            all_visual_masks.append(state["prompt_mask"])
            norm_pts_sampled = pts_norm
            all_norm_pts = all_pts_norm
            norm_box = box_norm
    else:  # "random"
        for idx in range(support_imgs.shape[0]):
            support_img = support_imgs[idx]
            support_mask = support_masks[idx]

            pts_norm, pts_actual = get_random_points_from_mask(mask=support_mask, num_points=num_points)
            if pts_norm is None:
                raise Exception(f"Mask for support image {idx} is empty. No points have been returned.")
            print(f"[PromptTokens] support {idx+1}/{support_imgs.shape[0]} - encoding {len(pts_norm)} random points from mask")
            state = encode_pts_prompts(processor, support_img, pts_norm, skip_coords)
            all_visual_tokens.append(state["prompt"])
            all_visual_masks.append(state["prompt_mask"])
            # pts_norm is already [0,1] keep it for caller if needed
            norm_pts_sampled = pts_norm
            actual_pts_sampled = pts_actual
        
    if not all_visual_tokens:
        raise ValueError("No valid support shots found.")

    # Aggregate visual knowledge
    aggregated_visual_prompt = torch.cat(all_visual_tokens, dim=0)
    aggregated_visual_mask = torch.cat(all_visual_masks, dim=1)
    
    return aggregated_visual_prompt, aggregated_visual_mask, norm_pts_sampled, all_norm_pts, norm_box, actual_pts_sampled

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

def cross_image_prediction(sam3=None, query_frame=None, support_imgs=None, support_masks=None, text_prompt="visual", skip_coords=False, num_points=20, matcher_calculator=None, use_fused_matcher_features=False, sampling="random", visualize=False, visual_output_path=None, experiment_mode="random", attn_layers="last", dense_cross_attn_skip_text_injection=False, sampling_inputs="both", attention_aggregate_function="sum", attn_sampling_mode="top-k"):
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

    visual_prompt, visual_mask, norm_pts_sampled, all_norm_pts, norm_box, actual_pts_sampled = get_prompt_tokens_from_support(
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
        dense_cross_attn_skip_text_injection=dense_cross_attn_skip_text_injection,
        sampling_inputs=sampling_inputs,
        attention_aggregate_function=attention_aggregate_function,
        attn_sampling_mode=attn_sampling_mode,
    )
    final_prompt, final_mask, text_outputs = aggregate_prompt_with_text_tokens(
        processor=sam3.processor,
        text_prompt=text_prompt,
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
    state = sam3.processor._forward_with_encoded_prompt(state)
    print(f"Found {len(state['scores'])} objects")
    merged_mask = np.any(np.array(state['masks'].cpu()), axis=0).squeeze(0)
    return merged_mask, norm_pts_sampled, all_norm_pts, norm_box, actual_pts_sampled


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
        
    class_list = dataset.get_class_ids()
    class_dic = dataset.idx_to_classname
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

            for eval_id, lemma in lemma_entries:
                if args.all_lemmas:
                    print(f"  Prompting with lemma: '{lemma}' (eval_id: {eval_id})")

                vid_output_dir = os.path.join(args.output_dir, f"{dir_name}_{eval_id}_{idx}", "output")
                vid_ground_truth_dir = os.path.join(args.output_dir, f"{dir_name}_{eval_id}_{idx}", "ground_truth")
                vid_frames_dir = os.path.join(args.output_dir, f"{dir_name}_{eval_id}_{idx}", "frames")
                vid_attn_dir = os.path.join(args.output_dir, f"{dir_name}_{eval_id}_{idx}", "attention_maps")
                if args.experiment_mode == "matcher":
                    vid_box_dir = os.path.join(args.output_dir, f"{dir_name}_{eval_id}_{idx}", "bounding_box")

                os.makedirs(vid_output_dir, exist_ok=True)
                os.makedirs(vid_ground_truth_dir, exist_ok=True)
                os.makedirs(vid_frames_dir, exist_ok=True)
                os.makedirs(vid_attn_dir, exist_ok=True)
                if args.experiment_mode == "matcher":
                    os.makedirs(vid_box_dir, exist_ok=True)

                predictions = []
                ground_truths = []
                point_scores_list = []
                all_point_scores_list = []

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
                    
                    _text = lemma if not args.disable_text else "visual"
                    if args.use_query_as_support:
                        _sup_imgs  = query_frame.unsqueeze(0)
                        _sup_masks = ground_truth.unsqueeze(0)
                    else:
                        _sup_imgs  = support_imgs
                        _sup_masks = support_masks

                    has_dedicated_pass = args.experiment_mode in (
                        "attn_prior", "dense_cross_attn", "self_attn"
                    )
                    if not has_dedicated_pass:
                        matcher_calculator.arm_inference_capture(args.attn_layers)

                    # Execution of the main pipeline: evaluates the query image using the support prompt.
                    # Variables returned:
                    # - prediction: the final binary segmentation mask [H, W] predicted for the query image
                    # - norm_pts_sampled: [num_points, 1, 2] the selected subset of prompt points in [0, 1] normalized coords
                    # - all_norm_pts: [M, 2] all candidate points evaluated before sampling (e.g. from matcher), in [0, 1] coords
                    # - norm_box: [4] [x1, y1, x2, y2] bounding box in [0, 1] normalized coords (generated by the matcher)
                    # - actual_pts_sampled: [num_points, 2] exact pixel coordinates [0, H] if sampled directly from a mask, else None
                    prediction, norm_pts_sampled, all_norm_pts, norm_box, actual_pts_sampled = cross_image_prediction(
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
                        visual_output_path=os.path.join(vid_box_dir, f"frame_{chosen_frames[frame_idx]}_tsne.png") if args.visualize_embeddings and args.experiment_mode == "matcher" else None,
                        experiment_mode=args.experiment_mode,
                        attn_layers=args.attn_layers,
                        dense_cross_attn_skip_text_injection=args.dense_cross_attn_skip_text_injection,
                        sampling_inputs=args.sampling_inputs,
                        attention_aggregate_function=args.attention_aggregate_function,
                        attn_sampling_mode=args.attn_sampling_mode,
                    )

                    # if we had a dedicated pass to compute the points to prompt the query image, we'd overwrite the attn maps with the ones of the inference pass
                    # is instead we just did an inference or matcher pass, we need to collect the attention maps from the inference pass
                    if not has_dedicated_pass:
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

                    predictions.append(prediction)
                    ground_truths.append(ground_truth)

                evaluator.update_evl(eval_id, ground_truths, predictions, sample_id=f"{dir_name}_{eval_id}_{idx}", points_list=point_scores_list, all_points_list=all_point_scores_list)
                print(f"Updated evaluation metrics for '{lemma}'")

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

    save_results(class_dic, evaluator, args, virtual_to_original=virtual_to_original, original_class_dic=original_class_dic, box_coordinates=box_coordinates)

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