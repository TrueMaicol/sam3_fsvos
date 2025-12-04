import torch 
import numpy as np
import os
from PIL import Image
import cv2
import random 
from utils.Evaluator import Evaluator
import time
from datetime import datetime
from YoutubeFSVOS import YTVOSDataset
from sam3.model_builder import build_sam3_video_model

class SAM3_FSVOS:
    def __init__(self, checkpoint, config, session_name, dataset_path, output_dir, verbose, test_query_frame_num):
        # self.args = args
        if checkpoint is None:
            print("No checkpoint path provided. Exiting")
            return

        self.checkpoint = checkpoint
        self.session_name = session_name
        self.dataset_path = dataset_path
        self.output_dir = output_dir
        self.verbose = verbose
        self.test_query_frame_num = test_query_frame_num     

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        # self.video_predictor = build_sam2_video_predictor(self.model_cfg, self.checkpoint, device=self.device, apply_postprocessing=False)
        self.sam3_model = build_sam3_video_model(checkpoint_path=self.checkpoint, device=self.device)  
        self.video_predictor = self.sam3_model.tracker
        self.video_predictor.backbone = self.sam3_model.detector.backbone
        print("Successfully loaded SAM2 model")
                
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
            image = Image.fromarray(image)
        image.save(path)

    def copy_paste_support_to_query(self, support_img, support_mask, query_img):
        """
        Crop the support object defined by the support_mask from support_img
        and paste it into the bottom left corner of query_img.
        
        Args:
            support_img: numpy array of shape (H, W, 3) - the support image
            support_mask: numpy array of shape (H, W) - binary mask defining the object
            query_img: numpy array of shape (H, W, 3) - the query image
            
        Returns:
            numpy array: query_img with the support object pasted in the bottom left corner
        """
        # Ensure inputs are numpy arrays
        if isinstance(support_img, Image.Image):
            support_img = np.array(support_img)
        if isinstance(query_img, Image.Image):
            query_img = np.array(query_img)
        if isinstance(support_mask, Image.Image):
            support_mask = np.array(support_mask)
        
        # Make a copy of query_img to avoid modifying the original
        result_img = query_img.copy()
        
        # Ensure mask is 2D and binary
        support_mask = support_mask.squeeze()
        if support_mask.ndim > 2:
            support_mask = support_mask[:, :, 0] if support_mask.shape[2] == 1 else support_mask.max(axis=2)
        binary_mask = support_mask > 0
        
        # Find bounding box of the masked region
        rows = np.any(binary_mask, axis=1)
        cols = np.any(binary_mask, axis=0)
        
        if not np.any(rows) or not np.any(cols):
            # No object found in mask, return original query image
            return result_img
        
        row_min, row_max = np.where(rows)[0][[0, -1]]
        col_min, col_max = np.where(cols)[0][[0, -1]]
        
        # Crop the support image and mask to the bounding box
        cropped_support = support_img[row_min:row_max+1, col_min:col_max+1].copy()
        cropped_mask = binary_mask[row_min:row_max+1, col_min:col_max+1]

        # Get dimensions
        crop_h, crop_w = cropped_support.shape[:2]
        query_h, query_w = result_img.shape[:2]
                
        # Calculate scale based on desired width (1/3 of query_w)
        target_max_width = query_w / 4
        target_max_height = query_h / 2

        scale_w = target_max_width / crop_w
        scale_h = target_max_height / crop_h
        
        scale = min(scale_w, scale_h)

        new_h = int(crop_h * scale)
        new_w = int(crop_w * scale)

        # Resize the cropped support image
        cropped_support = cv2.resize(cropped_support, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # Resize the mask (use INTER_NEAREST to preserve binary values)
        cropped_mask = cv2.resize(cropped_mask.astype(np.uint8), (new_w, new_h), interpolation=cv2.INTER_NEAREST).astype(bool)
        
        # Update dimensions
        crop_h, crop_w = new_h, new_w
        
        # Calculate paste position (bottom left corner)
        corner = random.choice(['top_left', 'top_right', 'bottom_left', 'bottom_right'])

        if corner == 'top_left':
            paste_y = 0
            paste_x = 0
        elif corner == 'top_right':
            paste_y = 0
            paste_x = query_w - crop_w
        elif corner == 'bottom_left':
            paste_y = query_h - crop_h
            paste_x = 0
        else:  # bottom_right
            paste_y = query_h - crop_h
            paste_x = query_w - crop_w
        
        # Paste the support object into the query image using the mask
        # Only paste pixels where the mask is True
        for c in range(3):  # For each color channel
            result_img[paste_y:paste_y+crop_h, paste_x:paste_x+crop_w, c] = np.where(
                cropped_mask,
                cropped_support[:, :, c],
                result_img[paste_y:paste_y+crop_h, paste_x:paste_x+crop_w, c]
            )
        
        return result_img

    def create_dirs(self, base_dir, video_query_set, support_set, crop_paste_support_to_query=False):
        
        frames_dir = os.path.join(base_dir, "frames")
        output_dir = os.path.join(base_dir, "output")
        ground_truth_dir = os.path.join(base_dir, "ground_truth")
        
        os.makedirs(frames_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(ground_truth_dir, exist_ok=True)

        if not crop_paste_support_to_query:
            for i, (img, _) in enumerate(support_set):
                self.save_image(img, os.path.join(frames_dir, f"{i:04d}.jpg"))
        else:
            (query_img, _) = video_query_set[0]
            for i, (img, mask) in enumerate(support_set):
                img = self.copy_paste_support_to_query(img, mask, query_img)
                self.save_image(img, os.path.join(frames_dir, f"{i:04d}.jpg"))

        for i, (img, _) in enumerate(video_query_set):
            self.save_image(img, os.path.join(frames_dir, f"{i + len(support_set):04d}.jpg"))

        return frames_dir, output_dir, ground_truth_dir

    @torch.inference_mode()
    @torch.autocast(device_type="cuda", dtype=torch.float32)
    def process_video_sam2(self, video_predictor, support_set, video_query_set, class_id, dir_name, evaluator, device, data_dir="./output", crop_paste_support_to_query=False):

        print(f"Processing video: {dir_name}")
        base_dir = f"{data_dir}/{dir_name}"

        frames_dir, prediction_dir, ground_truth_dir = self.create_dirs(base_dir, video_query_set, support_set, crop_paste_support_to_query=crop_paste_support_to_query)
        
        # Initialize inference state with all frames directory
        inference_state = video_predictor.init_state(video_path=frames_dir)
        if self.verbose:
            # Debug: Print the exact order SAM2 sees the frames
            print("=== SAM2 Frame Loading Debug ===")
            
            # Show what files we created in the frames directory
            frame_files = [f for f in os.listdir(frames_dir) if f.endswith('.jpg')]
            frame_files.sort(key=lambda p: int(os.path.splitext(p)[0]))
            print(f"Files in frames directory: {frame_files}")
            
            # Show what SAM2 actually loaded (from inference state)
            print(f"SAM2 loaded {inference_state['num_frames']} frames")
            print(f"Video dimensions: {inference_state['video_height']}x{inference_state['video_width']}")
            
            # Show the mapping between frame indices and file names
            print("Frame index to filename mapping:")
            for idx, filename in enumerate(frame_files):
                frame_type = "SUPPORT" if idx < len(support_set) else "QUERY"
                local_idx = idx if idx < len(support_set) else idx - len(support_set)
                print(f"  Frame {idx}: {filename} -> {frame_type} {local_idx}")
            
            print(f"Support frames: 0-{len(support_set)-1}")
            print(f"Query frames: {len(support_set)}-{len(support_set) + len(video_query_set) - 1}")
            
            # Show the exact frame paths SAM2 will use (mimicking SAM2's internal logic)
            sam2_frame_order = []
            for p in os.listdir(frames_dir):
                if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]:
                    sam2_frame_order.append(p)
            sam2_frame_order.sort(key=lambda p: int(os.path.splitext(p)[0]))
            print(f"SAM2 internal frame order: {sam2_frame_order}")
            
            # Verification: Check if our ordering matches SAM2's ordering
            our_order_matches = frame_files == sam2_frame_order
            print(f"Our frame order matches SAM2's internal order: {our_order_matches}")
            if not our_order_matches:
                print("WARNING: Frame ordering mismatch detected!")
                print(f"Our order: {frame_files}")
                print(f"SAM2 order: {sam2_frame_order}")
            print("=== End Debug ===\n")

        print("Loading support frames and their masks into SAM2")
        
        obj_id = 1  # Use same object ID for all support frames and query frames

        # Add support frame masks to the predictor
        for i, (img, mask) in enumerate(support_set):
            mask_tensor = torch.tensor(mask > 0, dtype=torch.bool, device=device)
            # print(f"Support frame {i}: mask tensor shape {mask_tensor.shape}, dtype {mask_tensor.dtype}, device {mask_tensor.device}")
            video_predictor.add_new_mask(
                inference_state=inference_state,
                frame_idx=i,
                obj_id=obj_id,
                mask=mask_tensor
            )
            self.save_mask_overlay(img, mask_tensor.cpu().numpy(), os.path.join(ground_truth_dir, f"support_{i:04d}.png"))
            # print(f"Added support frame {i} (corresponds to frame file {i:05d}.jpg)")
            # print(f"  Mask has {torch.sum(mask_tensor).item()} non-zero pixels")

        print("Processing query video...")

        # Load query frames in sorted order
        segmented_masks = []
        # Propagate masks to this frame
        video_segments = {}
        for out_frame_idx, out_obj_ids, low_res_masks, video_res_masks, obj_scores in video_predictor.propagate_in_video(inference_state, start_frame_idx=0, max_frame_num_to_track=len(video_query_set)+10, reverse=False, propagate_preflight=True):
            video_segments[out_frame_idx] = {
                out_obj_id: (video_res_masks[j] > 0.0).cpu().numpy()
                for j, out_obj_id in enumerate(out_obj_ids)
            }

        query_masks_gt = [mask for _, mask in video_query_set]
        for i, (query_img, query_mask) in enumerate(video_query_set):
            # print(f"Processing query frame {i}")
            query_frame = np.array(query_img)
            query_frame_idx = len(support_set) + i
            
            # Extract mask for current query frame
            if query_frame_idx in video_segments and obj_id in video_segments[query_frame_idx]:
                mask = video_segments[query_frame_idx][obj_id]
                segmented_masks.append(mask)
                # print(f"  Found mask with {np.sum(mask)} non-zero elements")
                # Save visualization
                self.save_mask_overlay(query_frame, mask, f"{prediction_dir}/out_{i:04d}.png")
                self.save_mask_overlay(query_frame, query_mask, f"{ground_truth_dir}/query_{i:04d}.png")
                # print(f"  Successfully processed query frame {i}")
            else:
                # No mask found, append empty mask
                empty_mask = np.zeros((query_frame.shape[0], query_frame.shape[1]), dtype=bool)
                segmented_masks.append(empty_mask)
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

    def test(self, group=1, seed=42, crop_paste_support_to_query=False):
        device = self.device
        video_predictor = self.video_predictor

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_directory = f"{self.output_dir}/{self.session_name}/fold_{group}_{timestamp}"
        if self.output_dir is None:
            output_directory = f"./output/{self.session_name}/fold_{group}_{timestamp}"
        
        os.makedirs(output_directory, exist_ok=True)

        test_dataset = YTVOSDataset(train=False, set_index=group, data_dir=self.dataset_path, test_query_frame_num=self.test_query_frame_num, seed=seed)
        test_list = test_dataset.get_class_ids()

        print('test_group:',group, '  test_num:', len(test_dataset), '  class_list:', test_list, ' dataset_path:', self.dataset_path)
        test_evaluations = Evaluator(class_list=test_list, verbose=self.verbose)
        support_set = []
        start_time = time.perf_counter()
        for index, data in enumerate(test_dataset):
            
            video_query_img, video_query_mask, new_support_img, new_support_mask, class_id, dir_name, begin_new = data
            if begin_new:
                support_set = [(img, mask) for img, mask in zip(new_support_img, new_support_mask)]
                print(f"Support set for class {class_id} initialized with {len(support_set)} images.")

            video_query_set = [(img, mask) for img, mask in zip(video_query_img, video_query_mask)]
            self.process_video_sam2(
                video_predictor,
                support_set,
                video_query_set, 
                class_id, 
                dir_name, 
                test_evaluations, 
                device,
                data_dir=output_directory,
                crop_paste_support_to_query=crop_paste_support_to_query)
            
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

        return mean_f, mean_j, score_dict
    

    def reprod_test(self, group=1):
        device = self.device
        video_predictor = self.video_predictor
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
                video_predictor,
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

