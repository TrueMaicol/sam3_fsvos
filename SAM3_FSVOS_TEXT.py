import torch 
import numpy as np
import os
from PIL import Image
import cv2
import random 
from utils.Evaluator import Evaluator
import time
from datetime import datetime
from datasets.YoutubeFSVOS.YoutubeFSVOS import YTVOSDataset
from datasets.YoutubeFSVOS.transform import TestTransform
from sam3.model_builder import build_sam3_video_predictor

from VLM_label_gen import LabelGenerator
from datasets.MiniVSPW.nminivspw_dataset import NMiniVSPWEpisodicData
from RandomStateManager import RandomStateManager

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

class SAM3_FSVOS_TEXT:
    def __init__(self, checkpoint, session_name, dataset_path, output_dir, verbose, test_query_frame_num, args):
        if checkpoint is None:
            print("No checkpoint path provided. Exiting")
            return

        self.args = args
        self.checkpoint = checkpoint
        self.session_name = session_name
        self.dataset_path = dataset_path
        self.output_dir = output_dir
        self.verbose = verbose
        self.test_query_frame_num = test_query_frame_num     
        self.benchmark = args.benchmark
        # MiniVSPW data
        self.data_list_path = args.data_list_path
        self.random_state_path = args.random_state_path
        self.run_number = args.run_number

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        gpus_to_use = range(torch.cuda.device_count())
        self.video_predictor = build_sam3_video_predictor(checkpoint_path=self.checkpoint, gpus_to_use=gpus_to_use)  
        print("Successfully loaded SAM3 model")
                
    def save_evaluation_results(self, output_directory, mean_f, mean_j, score_dict, elapsed_time):
        results_path = os.path.join(output_directory, "evaluation_results.txt")
        with open(results_path, 'w') as f:
            f.write(f"Mean F: {mean_f:.8f}\n")
            f.write(f"Mean J: {mean_j:.8f}\n\n")
            f.write("Detailed Scores:\n")
            for class_id, scores in score_dict.items():
                f.write(f"Class {class_id} - F: {scores['f_score']:.8f}, J: {scores['j_score']:.8f}\n")
            f.write(f"\nElapsed time in minutes: {elapsed_time:.4f}")
        print(f"Saved evaluation results to {results_path}")

    def save_mask_overlay(self, image, mask, output_path):
        """Save image with mask overlay"""
        # Ensure image is numpy array
        if isinstance(image, Image.Image):
            image = np.array(image)
        
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
        colored_mask[mask] = [0, 255, 0]  # Green overlay
        
        # Blend with original image
        overlay = cv2.addWeighted(image.astype(np.uint8), 0.7, colored_mask.astype(np.uint8), 0.3, 0)
        cv2.imwrite(output_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    def save_image(self, image, path):
        """Save a numpy array or PIL image to the specified path."""
        if isinstance(image, np.ndarray):
            # Handle float32 images from transform (values in range 0-1)
            if image.dtype == np.float32:
                image = (image * 255).astype(np.uint8)
            image = Image.fromarray(image)
        image.save(path)

    def get_bounding_box(self, mask):
        # get the bounding box of the object normalized to the frame size (x,y,w,h)
        # Ensure mask is 2D (squeeze out channel dimensions if present)
        if mask.ndim > 2:
            mask = mask.squeeze()
        if mask.ndim > 2:  # If still 3D, take first channel
            mask = mask[:, :, 0]
        h, w = mask.shape[:2]
        y, x = np.where(mask)
        x_min, x_max = np.min(x), np.max(x)
        y_min, y_max = np.min(y), np.max(y)
        # Normalize to frame size (0-1 range)
        return [x_min / w, y_min / h, (x_max - x_min) / w, (y_max - y_min) / h]

    def create_support_frame(self, support_set, frames_dir):
        # FIRST VERSION [TO BE USED ON 1-SHOT ONLY]: just the support frame and the bounding box of the object
        support_frame = support_set[0][0]
        support_mask = support_set[0][1]

        self.save_image(support_frame, os.path.join(frames_dir, f"{0:05d}.jpg")) 

        # get the bounding box of the object normalized to the frame size (x,y,w,h)
        support_bounding_box = self.get_bounding_box(support_mask)

        return support_bounding_box

    def save_bbox_overlay(self, image, bbox, output_path):
        """Save image with bounding box overlay
        
        Args:
            image: numpy array or PIL Image
            bbox: bounding box in normalized format [x, y, w, h] where values are in range 0-1
            output_path: path to save the output image
        """
        # Ensure image is numpy array
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        # Handle float32 images from transform (values in range 0-1)
        if image.dtype == np.float32:
            image = (image * 255).astype(np.uint8)
        
        # Create a copy to draw on
        overlay = image.copy()
        
        # Get image dimensions
        h, w = image.shape[:2]
        
        # Convert normalized bbox to pixel coordinates
        x_norm, y_norm, w_norm, h_norm = bbox
        x1 = int(x_norm * w)
        y1 = int(y_norm * h)
        x2 = int((x_norm + w_norm) * w)
        y2 = int((y_norm + h_norm) * h)
        
        # Draw bounding box (green color, thickness 2)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Save the image
        cv2.imwrite(output_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    def copy_paste_support_to_query(self, support_set, query_img):
        """
        Crop support objects defined by masks from the support set
        and paste them into a blank frame based on query_img dimensions.
        Objects are placed in a 3x3 grid cross pattern.
        
        Args:
            support_set: list of (img, mask) tuples where:
                        - img: numpy array of shape (H, W, 3) - the support image
                        - mask: numpy array of shape (H, W) - binary mask defining the object
            query_img: numpy array of shape (H, W, 3) - the query image (used for dimensions)
            
        Returns:
            numpy array: blank frame with support objects pasted in cross pattern
            combined_bbox: single bounding box [x, y, w, h] encompassing all pasted objects (normalized coordinates)
        """
        # Ensure query_img is numpy array for dimensions
        if isinstance(query_img, Image.Image):
            query_img = np.array(query_img)
        
        # Create blank result image
        result_img = np.zeros_like(query_img)
        query_h, query_w = result_img.shape[:2]
        
        num_supports = len(support_set)
        assert num_supports <= 5, f"Maximum 5 support frames supported, got {num_supports}"
        
        # Single horizontal row layout (vertically centered):
        # Objects placed in a row for tighter vertical bounding box
        # [0] [1] [2] [3] [4]
        # This gives bbox height = ~1/3 frame instead of full frame
        
        # Calculate cell dimensions (1 row x 5 columns)
        cell_w = query_w // 5
        cell_h = query_h // 3  # Only use middle third vertically
        
        # Vertical offset to center the row
        row_offset_y = query_h // 3
        
        # Padding within each cell
        padding = 5
        
        for idx, (support_img, support_mask) in enumerate(support_set):
            # Ensure inputs are numpy arrays
            if isinstance(support_img, Image.Image):
                support_img = np.array(support_img)
            if isinstance(support_mask, Image.Image):
                support_mask = np.array(support_mask)
            
            # Ensure mask is 2D and binary
            support_mask = support_mask.squeeze()
            if support_mask.ndim > 2:
                support_mask = support_mask[:, :, 0] if support_mask.shape[2] == 1 else support_mask.max(axis=2)
            binary_mask = support_mask > 0
            
            # Find bounding box of the masked region
            rows = np.any(binary_mask, axis=1)
            cols = np.any(binary_mask, axis=0)
            
            if not np.any(rows) or not np.any(cols):
                continue
            
            row_min, row_max = np.where(rows)[0][[0, -1]]
            col_min, col_max = np.where(cols)[0][[0, -1]]
            
            # Crop the support image and mask to the bounding box
            cropped_support = support_img[row_min:row_max+1, col_min:col_max+1].copy()
            cropped_mask = binary_mask[row_min:row_max+1, col_min:col_max+1]
            
            # Scale to fit within cell (with padding)
            crop_h, crop_w = cropped_support.shape[:2]
            target_w = cell_w - 2 * padding
            target_h = cell_h - 2 * padding
            
            scale = min(target_w / crop_w, target_h / crop_h)
            new_w = int(crop_w * scale)
            new_h = int(crop_h * scale)
            
            cropped_support = cv2.resize(cropped_support, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            cropped_mask = cv2.resize(cropped_mask.astype(np.uint8), (new_w, new_h), interpolation=cv2.INTER_NEAREST).astype(bool)
            
            # Column positions for center-first fill order:
            # Layout: [4] [2] [0] [1] [3]
            # 1-shot: center, then alternating outward
            column_positions = [2, 3, 1, 4, 0]
            col = column_positions[idx]
            
            # Calculate cell position
            cell_x = col * cell_w
            cell_y = row_offset_y  # All objects in the same row (middle third)
            
            # Center the object within the cell
            paste_x = cell_x + (cell_w - new_w) // 2
            paste_y = cell_y + (cell_h - new_h) // 2
            
            # Paste the support object into the result image using the mask
            for c in range(3):
                result_img[paste_y:paste_y+new_h, paste_x:paste_x+new_w, c] = np.where(
                    cropped_mask,
                    cropped_support[:, :, c],
                    result_img[paste_y:paste_y+new_h, paste_x:paste_x+new_w, c]
                )
        
        # Create a mask from non-zero pixels in the result image and use existing get_bounding_box
        combined_mask = np.any(result_img > 0, axis=2)
        if not np.any(combined_mask):
            return result_img, None
        
        combined_bbox = self.get_bounding_box(combined_mask)
        
        return result_img, combined_bbox


    def create_dirs(self, base_dir, support_set, video_query_set, use_support_visuals=False, save_support_gt=False):
        
        frames_dir = os.path.join(base_dir, "frames")
        output_dir = os.path.join(base_dir, "output")
        ground_truth_dir = os.path.join(base_dir, "ground_truth")
        os.makedirs(frames_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(ground_truth_dir, exist_ok=True)

        support_bounding_box = None

        if save_support_gt:
            for i, (img, mask) in enumerate(support_set):
                self.save_mask_overlay(img, mask, os.path.join(ground_truth_dir, f"support_{i:05d}.png"))

        if use_support_visuals:
            # THIS IS TO COPY THE SUPPORT OBJECT TO A BLANK FRAME AND RETURN THE BOUNDING BOX OF THE SUPPORT OBJECT
            query_img, _ = video_query_set[0]
            result_img, support_bounding_box = self.copy_paste_support_to_query(support_set, query_img)
            self.save_image(result_img, os.path.join(frames_dir, f"{0:05d}.jpg"))
            self.save_bbox_overlay(result_img, support_bounding_box, os.path.join(ground_truth_dir, f"support_bbox.jpg"))

            # THIS IS TO USE THE NATIVE FRAME FROM THE SUPPORT SET AND RETURN THE BOUNDING BOX OF THE SUPPORT OBJECT
            # support_bounding_box = self.create_support_frame(support_set, frames_dir)
            # self.save_bbox_overlay(support_set[0][0], support_bounding_box, os.path.join(ground_truth_dir, f"support_bbox.jpg"))

        for i, (img, _) in enumerate(video_query_set):
            idx = i + 1 if use_support_visuals else i
            self.save_image(img, os.path.join(frames_dir, f"{idx:05d}.jpg"))

        return frames_dir, output_dir, ground_truth_dir, support_bounding_box

    def propagate_in_video(self, session_id, use_support_visuals):
        # we will just propagate from frame 0 to the end of the video
        outputs_per_frame = {}
        for response in self.video_predictor.handle_stream_request(
            request=dict(
                type="propagate_in_video",
                session_id=session_id,
            )
        ):
            outputs_per_frame[response["frame_index"]] = response["outputs"]

        return outputs_per_frame

    @torch.inference_mode()
    @torch.autocast(device_type="cuda", dtype=torch.float32)
    def process_video_sam2(self, support_set, video_query_set, class_id, dir_name, evaluator, device, data_dir="./output", class_name=None, use_support_visuals=False, gen_labels=False):
        video_predictor = self.video_predictor
        print(f"Processing video: {dir_name}")
        base_dir = f"{data_dir}/{dir_name}"
        print("CLASSNAME: ", class_name)
        frames_dir, prediction_dir, ground_truth_dir, support_bounding_box = self.create_dirs(base_dir, support_set, video_query_set, save_support_gt=gen_labels or use_support_visuals, use_support_visuals=use_support_visuals)
        
        print("Processing query video...")
        # Initialize inference state with all frames directory
        # inference_state = video_predictor.init_state(video_path=frames_dir)
        response = self.video_predictor.handle_request(
            request=dict(
                type="start_session",
                resource_path=frames_dir,
            ))
        session_id = response["session_id"]
                    
        response = self.video_predictor.handle_request(
            request=dict(
                type="add_prompt",
                session_id=session_id,
                frame_index=0,
                text=class_name, # Commented this to use the semantic prompt without text and only image exemplar
                # text=None,
                bounding_boxes=[support_bounding_box] if use_support_visuals else None,
                bounding_box_labels=[1] if use_support_visuals else None,
            ))

        outputs_per_frame = self.propagate_in_video(session_id, use_support_visuals)

        self.video_predictor.close_session(session_id)

        # Build video_segments dict from outputs_per_frame
        # The new API returns:
        #   out_obj_ids: array of object ids (0-indexed in output)
        #   out_binary_masks: array of boolean masks [num_objs, H, W]
        # We merge all object masks into a single mask per frame
        video_segments = {}
        for frame_idx, out in outputs_per_frame.items():
            out_obj_ids = out['out_obj_ids']  # numpy array, e.g. [0, 1, 2, ...]
            out_binary_masks = out['out_binary_masks']  # numpy array [num_objs, H, W]
            
            # Merge all object masks into a single mask using logical OR
            if len(out_obj_ids) > 0 and out_binary_masks.shape[0] > 0:
                # Combine all masks: result is True where any object mask is True
                merged_mask = np.any(out_binary_masks, axis=0)  # Shape: [H, W]
                video_segments[frame_idx] = merged_mask
            else:
                # No objects detected, store None to indicate empty
                video_segments[frame_idx] = None

        # Load query frames in sorted order
        segmented_masks = []
        query_masks_gt = [mask for _, mask in video_query_set]
        
        for i, (query_img, query_mask) in enumerate(video_query_set):
            query_frame = np.array(query_img)
            # Frame indices in outputs_per_frame are 0-indexed from the frames directory
            query_frame_idx = i + 1 if use_support_visuals else i  # Frames saved are 0-indexed
            
            # Extract merged mask for current query frame
            self.save_mask_overlay(query_frame, query_mask, f"{ground_truth_dir}/query_{i:04d}.png")
            if query_frame_idx in video_segments and video_segments[query_frame_idx] is not None:
                mask = video_segments[query_frame_idx]
                segmented_masks.append(mask)
                # Save visualization
                self.save_mask_overlay(query_frame, mask, f"{prediction_dir}/out_{i:04d}.png")
                self.save_image(mask.astype(np.uint8)*255, f"{prediction_dir}/out_mask_{i:04d}.png")
            else:
                # No mask found, append empty mask
                empty_mask = np.zeros((query_frame.shape[0], query_frame.shape[1]), dtype=bool)
                segmented_masks.append(empty_mask)
                self.save_image(empty_mask.astype(np.uint8)*255, f"{prediction_dir}/out_mask_{i:04d}.png")
                print(f"  WARNING: No mask found for query frame {i} (frame_idx {query_frame_idx})")
               
        print(f"Segmentation complete! Generated {len(segmented_masks)} masks")
        print("Updating evaluation metrics")
        evaluator.update_evl(class_id, query_masks_gt, segmented_masks)
        
        return segmented_masks

    def reset_reproducibility(self, seed: int = 42, verbose: bool = True):
        """
        Reset Python, NumPy, and Torch seeds for reproducibility.
        Does not force deterministic kernels.
        """
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)

        if verbose:
            print(f"[Seeds reset to {seed}]")

    def load_test_data(self, n_support_frames=5):
        test_dataset = []
        for i, dir in enumerate(os.listdir(self.dataset_path)):
            if os.path.isdir(os.path.join(self.dataset_path, dir)):
                temp = {}
                support_set = []
                video_query_set = []
                for j in range(len(os.listdir(os.path.join(self.dataset_path, dir, "frames")))):
                    if j < n_support_frames:
                        img = Image.open(os.path.join(self.dataset_path, dir, "frames", f"{j:05d}.jpg"))
                        img = np.array(img)
                        mask = np.array(Image.open(os.path.join(self.dataset_path, dir, "support", f"{j:05d}.png")).convert("L"))
                        mask = np.array(mask)
                        support_set.append((img, mask))
                    else:
                        img = Image.open(os.path.join(self.dataset_path, dir, "frames", f"{j:05d}.jpg"))
                        img = np.array(img)
                        mask = np.array(Image.open(os.path.join(self.dataset_path, dir, "ground_truth", f"{j:05d}.png")).convert("L"))
                        mask = np.array(mask)
                        video_query_set.append((img, mask))
                dir_name_split = dir.split("_")
                temp["dir_name"] = dir_name_split[0]
                temp["class_id"] = int(dir_name_split[1])
                temp["support_set"] = support_set
                temp["video_query_set"] = video_query_set
                test_dataset.append(temp)
        return test_dataset

    def load_video_data(self, video_dir_path, n_support_frames=5):
        data = {}
        if os.path.isdir(video_dir_path):
            support_set = []
            video_query_set = []
            for j in range(len(os.listdir(os.path.join(video_dir_path, "frames")))):
                if j < n_support_frames:
                    img = Image.open(os.path.join(video_dir_path, "frames", f"{j:05d}.jpg"))
                    img = np.array(img)
                    mask = np.array(Image.open(os.path.join(video_dir_path, "support", f"{j:05d}.png")).convert("L"))
                    mask = np.array(mask)
                    support_set.append((img, mask))
                else:
                    img = Image.open(os.path.join(video_dir_path, "frames", f"{j:05d}.jpg"))
                    img = np.array(img)
                    mask = np.array(Image.open(os.path.join(video_dir_path, "ground_truth", f"{j:05d}.png")).convert("L"))
                    mask = np.array(mask)
                    video_query_set.append((img, mask))
            dir_name_split = os.path.basename(video_dir_path).split("_")
            data["dir_name"] = os.path.basename(video_dir_path)
            data["class_id"] = int(dir_name_split[1])
            data["support_set"] = support_set
            data["video_query_set"] = video_query_set
        return data

    def get_video_names(self):
        return os.listdir(self.dataset_path)

    def test(self, fold=1, seed=42, nshot=5):

        device = self.device

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_directory = f"{self.output_dir}/{self.session_name}/fold_{fold}_{timestamp}"
        if self.output_dir is None:
            output_directory = f"./output/{self.session_name}/fold_{fold}_{timestamp}"
        
        os.makedirs(output_directory, exist_ok=True)

        if self.benchmark == "youtube-fsvos":
            test_dataset = YTVOSDataset(train=False, set_index=fold, data_dir=self.dataset_path, test_query_frame_num=self.test_query_frame_num, seed=seed, support_frame=nshot, transforms=TestTransform(size=518))
            random.seed(seed)
            fix_randseed(seed)
        elif self.benchmark == "minivspw":
            test_dataset = NMiniVSPWEpisodicData(
                fold=fold-1,
                sprtset_as_frames=False,
                split_type='test',
                shot=nshot,
                data_root=self.dataset_path,
                data_list_path=self.data_list_path,
                transform=TestTransform(size=518) ,
            )
            state_manager = RandomStateManager(save_dir=self.random_state_path)

            if self.run_number == 1:
                state_manager.initialize_seed(seed=seed)
            else:
                state_manager.load_state(fold=fold, run_number=self.run_number-1)
        test_list = test_dataset.get_class_ids() 

        print('test_group:',fold, '  test_num:', len(test_dataset), '  class_list:', test_list, ' dataset_path:', self.dataset_path)
        test_evaluations = Evaluator(class_list=test_list, verbose=self.verbose)
        support_set = []
        start_time = time.perf_counter()

        if self.args.gen_labels:
            label_generator = LabelGenerator(self.args)

        for index, data in enumerate(test_dataset):

            if self.benchmark == "youtube-fsvos":
                video_query_img, video_query_mask, new_support_img, new_support_mask, class_id, dir_name, begin_new = data

                if begin_new:
                    support_set = [(img, mask) for img, mask in zip(new_support_img, new_support_mask)]
                    class_name = test_dataset.idx_to_classname[class_id]
                    
            elif self.benchmark == "minivspw":
                video_query_img, video_query_mask, new_support_img, new_support_mask, class_id, dir_name, _ = data
                support_set = [(img, mask) for img, mask in zip(new_support_img, new_support_mask)]
                class_id = class_id[0]
                class_name = test_dataset.idx_to_classname[class_id]

            if self.args.gen_labels:
                if (self.benchmark == "youtube-fsvos" and begin_new) or (self.benchmark == "minivspw"):
                    # new_support_img is a list of numpy arrays in (H, W, C) format
                    # new_support_mask is a list of numpy arrays in (H, W, 1) format
                    # Stack them and convert to tensors with proper format for VLM:
                    # support_imgs: (bs=1, ns, c, h, w)
                    # support_masks: (bs=1, ns, h, w)
                    support_imgs = torch.tensor(np.stack(new_support_img, axis=0)).permute(0, 3, 1, 2).unsqueeze(0)
                    support_masks = torch.tensor(np.stack(new_support_mask, axis=0)).squeeze(-1).unsqueeze(0)
                    
                    print("support set shape: ", support_imgs.shape, support_masks.shape)

                    class_name, _ = label_generator.fetch_semantic_info(
                        support_imgs,
                        support_masks,
                        self.args,
                        use_descriptions=False,
                        # save_dir=os.path.join(output_directory, f"class_{class_id}_support_visuals")
                        save_dir=None
                        )
                    print("VLM Predicted class: ", class_name)
            print(f"Support set for class {class_id}, {class_name} -  initialized with {len(support_set)} images.")          

            dir_name = f"{dir_name}_{class_id}_{index}"
            video_query_set = [(img, mask) for img, mask in zip(video_query_img, video_query_mask)]
            self.process_video_sam2(
                support_set,
                video_query_set, 
                class_id, 
                dir_name, 
                test_evaluations, 
                device,
                data_dir=output_directory,
                class_name=class_name,
                use_support_visuals=self.args.use_support_visuals,
                gen_labels=self.args.gen_labels)
            
            print(f"F-score list: {test_evaluations.f_score}")
            print(f"J-score list: {test_evaluations.j_score}")
        elapsed_time = time.perf_counter() - start_time
        elapsed_minutes = elapsed_time / 60.0
        print(f"Total processing time: {elapsed_minutes:.4f} minutes")

        mean_f = np.mean(test_evaluations.f_score)
        str_mean_f = 'F: %.8f ' % (mean_f)
        mean_j = np.mean(test_evaluations.j_score)
        str_mean_j = 'J: %.8f ' % (mean_j)

        f_list = ['%.8f' % n for n in test_evaluations.f_score]
        str_f_list = ' '.join(f_list)
        j_list = ['%.8f' % n for n in test_evaluations.j_score]
        str_j_list = ' '.join(j_list)
        # Generate dictionary with class id as key and f_score, j_score as values
        score_dict = {
            class_id: {"f_score": f, "j_score": j}
            for class_id, f, j in zip(test_list, test_evaluations.f_score, test_evaluations.j_score)
        }

        print(str_mean_f, str_f_list + '\n')
        print(str_mean_j, str_j_list + '\n')

        # Save evaluation results
        self.save_evaluation_results(output_directory, mean_f, mean_j, score_dict, elapsed_minutes)
        
        # if self.benchmark == 'minivspw':
        #     state_manager.save_state(fold=fold, run_number=self.run_number)

        return mean_f, mean_j, score_dict
    

    def reprod_test(self, group=1):
        device = self.device
        n_support_frames = 1

        output_directory = f"{self.output_dir}/{self.session_name}"
        if self.output_dir is None:
            output_directory = f"./output/{self.session_name}"

        test_list = [i * 4 + group for i in range(10)]
        print(f"Testing on classes: {test_list}")
        os.makedirs(output_directory, exist_ok=True)

        test_evaluations = Evaluator(class_list=test_list, verbose=self.verbose)
        # test_dataset = self.load_test_data(n_support_frames=n_support_frames)
        test_video_names = self.get_video_names()

        # self.reset_reproducibility(seed=42, verbose=True)
        start_time = time.perf_counter()

        support_set = []
        for index, vid in enumerate(test_video_names):
            print(f"Processing video {index + 1}/{len(test_video_names)}: {vid}")
            data = self.load_video_data(os.path.join(self.dataset_path, vid), n_support_frames=n_support_frames)
            support_set = data["support_set"]
            video_query_set = data["video_query_set"]
            dir_name = data["dir_name"]
            class_id = data["class_id"]
            
            self.process_video_sam2(
                support_set,
                video_query_set, 
                class_id, 
                dir_name, 
                test_evaluations, 
                device,
                data_dir=output_directory)

            f_list = ['%.8f' % n for n in test_evaluations.f_score]
            str_f_list = ' '.join(f_list)
            j_list = ['%.8f' % n for n in test_evaluations.j_score]
            str_j_list = ' '.join(j_list)
            score_dict = {
                class_id: {"f_score": f, "j_score": j}
                for class_id, f, j in zip(test_list, test_evaluations.f_score, test_evaluations.j_score)
            }
            print(f"Intermediate Results after video {vid}:")
            print(score_dict)

        mean_f = np.mean(test_evaluations.f_score)
        str_mean_f = 'F: %.8f ' % (mean_f)
        mean_j = np.mean(test_evaluations.j_score)
        str_mean_j = 'J: %.8f ' % (mean_j)

        elapsed_time = time.perf_counter() - start_time
        elapsed_minutes = elapsed_time / 60.0
        print(f"Total processing time: {elapsed_minutes:.4f} minutes")

        f_list = ['%.8f' % n for n in test_evaluations.f_score]
        str_f_list = ' '.join(f_list)
        j_list = ['%.8f' % n for n in test_evaluations.j_score]
        str_j_list = ' '.join(j_list)
        # Generate dictionary with class id as key and f_score, j_score as values
        score_dict = {
            class_id: {"f_score": f, "j_score": j}
            for class_id, f, j in zip(test_list, test_evaluations.f_score, test_evaluations.j_score)
        }

        print(str_mean_f, str_f_list + '\n')
        print(str_mean_j, str_j_list + '\n')

        # Save evaluation results
        self.save_evaluation_results(output_directory, mean_f, mean_j, score_dict, elapsed_minutes)

