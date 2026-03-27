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
    # Series of rules to match for the args
    if args.matcher_box and (not args.nshot > 0):
        raise Exception("To use --matcher_box --n_shot must be > 0, otherwise we don't have any reference image")
    elif args.frame_num <= 0:
        raise Exception("--frame_num must be > 0")


def get_arguments():
        parser = argparse.ArgumentParser(description='FSVOS')
        parser.add_argument("--checkpoint", type=str, default=None)
        parser.add_argument("--benchmark", type=str, default="youtube-fsvos", choices=["youtube_fsvos", "minivspw", "coco", "lvis", "ade20k", "pascal"])
        parser.add_argument("--session_name", type=str, default=None)
        parser.add_argument("--dataset_path", type=str, default=None)
        parser.add_argument("--data_list_path", type=str, default=None)
        parser.add_argument("--output_dir", type=str, default="./output")
        parser.add_argument("--fold", type=int, default=1)
        parser.add_argument("--frame_num", type=int, default=2)
        parser.add_argument("--nshot", type=int, default=0)
        parser.add_argument("--use_synset_names", action="store_true", default=False)
        parser.add_argument("--synset_mapping_folder_path", type=str, default="/leonardo_work/IscrC_MARSv2/datasets/synset_mappings")
        parser.add_argument("--use_grouping_ade20k", action="store_true", default=False, help="Enable grouping of classes using JSON [ONLY ON ADE20K].")
        parser.add_argument("--all_lemmas", action="store_true", default=False, help="Iterate over all lemmas, instead of just the one selected inside the mapping")
        parser.add_argument("--matcher_box", action="store_true", default=False, help="Use bipartite matching from matcher to get a bounding box on the target image. Requires n_shot > 0")

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

def save_image_with_points(image, points, output_path):
    """
    Draws red points on the image and saves it.
    Points are expected to be a list of (x, y) tuples.
    """
    res_image = Image.fromarray(image.copy())
    draw = ImageDraw.Draw(res_image)
    
    for (x, y) in points:
        draw.ellipse(
            [x-5, y-5, x+5, y+5], 
            fill="red", 
            outline="red"
        )
    res_image.save(output_path)

def convert_box_to_sam3_format(box=None, image_size=518):
    # SAM3 source code says
    # "The box is assumed to be in [center_x, center_y, width, height] format and normalized in [0, 1] range."
    
    assert isinstance(image_size, (int, tuple, list))
    if isinstance(image_size, int):
        out_resolution = (image_size, image_size)
    else:
        out_resolution = image_size

    x1, y1, x2, y2 = box
    (W, H) = out_resolution
    
    center_X = (x1 + x2) / 2
    center_Y = (y1 + y2) / 2
    abs_width = x2 - x1
    abs_height = y2 - y1

    norm_center_X = center_X / W
    norm_center_Y = center_Y / H
    norm_width = abs_width / W
    norm_height = abs_height / H

    return [norm_center_X, norm_center_Y, norm_width, norm_height]

def convert_points_to_sam3_format(points=None, image_size=518):
    # SAM3 requires point list to be in shape [N_POINTS, bs, 2] with batch_size being 1 as default in non batching cases
    return (points[:, :2]/image_size)[:, None, :]
    

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

def save_results(class_dic, evaluator, args, virtual_to_original=None, original_class_dic=None, box_coordinates=None):
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
            row = {
                'size': sample['size_category'],
                'class_id': class_idx,
                'class_name': class_name,
                'j_score': float(sample['j_score']),
                'pixel_ratio': float(sample['pixel_ratio']),
                'sample_id': sample['sample_id'],
            }
            orig_cid, orig_cname = _orig(class_idx)
            if orig_cid is not None:
                row['original_class_idx'] = orig_cid
                row['original_class_name'] = orig_cname
            sample_data.append(row)
    sample_df = pd.DataFrame(sample_data)
    sample_csv_path = os.path.join(results_dir, f"{args.benchmark}_sample_scores.csv")
    sample_df.to_csv(sample_csv_path, index=False, sep=';')

    # Class-level scores CSV
    class_scores_data = []
    for cid, cname, iou in zip(class_list_idx, class_list_names, evaluator.iou_list):
        row = {'class_id': cid, 'class_name': cname, 'iou_score': float(iou)}
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


def main():
    args = get_arguments()
    
    validate_args(args)

    # Validate arguments
    print('Running parameters:\n')
    print(json.dumps(vars(args), indent=4, separators=(',', ':')))

    sam3 = SAM3_IMAGE_PREDICTOR(checkpoint=args.checkpoint)

    random.seed(args.seed)  
    fix_randseed(args.seed)
   
    if args.matcher_box:
        matcher_box_calculator = MatcherBoxCalculator(sam3_model=sam3.model)

    # create the dataset from the builder
    loader = ImageDataset(args.benchmark, args)
    dataloader = DataLoader(
        loader,
        batch_size=None,
        num_workers=1,
        shuffle=False,
        prefetch_factor=0
    )

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
    for idx, data in enumerate(loader):
        
        query_imgs = data['query_imgs']
        query_masks = data['query_masks']
        support_imgs = data['support_imgs']
        support_masks = data['support_masks']
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
            if args.matcher_box:
                vid_box_dir = os.path.join(args.output_dir, f"{dir_name}_{eval_id}_{idx}", "bounding_box")

            os.makedirs(vid_output_dir, exist_ok=True)
            os.makedirs(vid_ground_truth_dir, exist_ok=True)
            os.makedirs(vid_frames_dir, exist_ok=True)
            if args.matcher_box:
                os.makedirs(vid_box_dir, exist_ok=True)

            predictions = []
            ground_truths = []

            for frame_idx in range(len(query_imgs)):
                print(f"Processing frame {chosen_frames[frame_idx]}")

                # img_pil = Image.fromarray((query_imgs[frame_idx] * 255).astype(np.uint8))
                query_frame = query_imgs[frame_idx]
                img_numpy = (query_frame * 255).permute(1, 2, 0).to(torch.uint8).cpu().numpy()
                ground_truth = query_masks[frame_idx].squeeze()
                print(f"query_frame shape: {query_frame.shape}")
                print(f"ground_truth shape: {ground_truth.shape}")
                
                box = None
                if args.matcher_box:
                    box, points = matcher_box_calculator.compute_box(
                        reference_image=support_imgs, 
                        target_image=query_frame, 
                        reference_mask=support_masks
                    )
                    for i in range(support_imgs.shape[0]):
                        save_mask_overlay(Image.fromarray((support_imgs[i]*255).permute(1,2,0).to(torch.uint8).cpu().numpy()), support_masks[i], os.path.join(vid_box_dir, f"support_{i}.png"))
                    box = matcher_box_calculator.convert_box_to_input_resolution(box=box, output_resolution=518)
                    save_image_with_box(image=img_numpy, box=box, output_path=os.path.join(vid_box_dir, f"frame_{chosen_frames[frame_idx]}.png"))
                    save_image_with_points(image=img_numpy, points=points, output_path=os.path.join(vid_box_dir, f"frame_{chosen_frames[frame_idx]}_points.png"))
                    sam3_box = convert_box_to_sam3_format(box=box, image_size=518)
                    sam3_points = convert_points_to_sam3_format(points=points, image_size=518)
                    box_coordinates.append(sam3_box)
                    print(f"Box: {box}")
                    print(f"SAM3 Box: {sam3_box}")

                if args.matcher_box:
                    # prediction = sam3.prompt_text_with_box(image=query_frame, text_prompt=lemma, box=sam3_box)
                    prediction = sam3.prompt_text_with_points(image=query_frame, text_prompt=lemma, points=sam3_points)
                else:
                    prediction = sam3.prompt_text(image=query_frame, text_prompt=lemma)
                img_pil = Image.fromarray(img_numpy)
                save_image(img_pil, os.path.join(vid_frames_dir, f"frame_{chosen_frames[frame_idx]}_input.png"))
                save_image(prediction, os.path.join(vid_output_dir, f"frame_{chosen_frames[frame_idx]}.png"))
                save_image(ground_truth, os.path.join(vid_ground_truth_dir, f"frame_{chosen_frames[frame_idx]}.png"))

                save_mask_overlay(img_pil, prediction, os.path.join(vid_output_dir, f"frame_{chosen_frames[frame_idx]}_overlay.png"))
                save_mask_overlay(img_pil, ground_truth, os.path.join(vid_ground_truth_dir, f"frame_{chosen_frames[frame_idx]}_overlay.png"))

                predictions.append(prediction)
                ground_truths.append(ground_truth)

            evaluator.update_evl(eval_id, ground_truths, predictions, sample_id=f"{dir_name}_{eval_id}_{idx}")
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

    # Generate dictionary with class id as key and f_score, j_score as values
    scoring_class_list = virtual_class_list if args.all_lemmas else class_list
    score_dict = {
        cid: {"iou_score": float(iou)}
        for cid, iou in zip(scoring_class_list, evaluator.iou_list)
    }

    clean_score_dict = {k: v.tolist() if hasattr(v, 'tolist') else v for k, v in score_dict.items()}

    save_results(class_dic, evaluator, args, virtual_to_original=virtual_to_original, original_class_dic=original_class_dic, box_coordinates=box_coordinates)

    # PRINT RESULTS
    print("\n\n")
    print("-" * 50)
    print(f"Fold {args.fold} - Mean IoU: {mean_iou}")
    print(f"Detailed Scores: {json.dumps(clean_score_dict, indent=4)}")


if __name__ == '__main__':
    main()