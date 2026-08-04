import numpy as np
from utils.davis_JF import db_eval_boundary, db_eval_iou
from utils.blob_evaluation import blob_analysis
import torch

def measure(y_in, pred_in):
    thresh = .5
    y = y_in > thresh
    pred = pred_in > thresh
    tp = np.logical_and(y, pred).sum()
    tn = np.logical_and(np.logical_not(y), np.logical_not(pred)).sum()
    fp = np.logical_and(np.logical_not(y), pred).sum()
    fn = np.logical_and(y, np.logical_not(pred)).sum()
    return tp, tn, fp, fn

class Evaluator():
    '''eval training output'''

    def __init__(self, class_list=None, verbose=False, input_image_size=518):
        assert class_list is not None
        self.class_indexes = class_list
        self.num_classes = len(class_list)
        self.verbose = verbose
        if input_image_size is not None:
            if isinstance(input_image_size, tuple):
                self.input_image_size = input_image_size[0] * input_image_size[1]
            else:
                self.input_image_size = input_image_size**2
        else:
            print("Using default input image size: 518x518")
            self.input_image_size = 518**2
        # self.threshold = {
        #     "SMALL": 32**2 / self.input_image_size,
        #     "MEDIUM": 96**2 / self.input_image_size,
        # }
        self.threshold = {
            "SMALL": 0.05,
            "MEDIUM": 0.20,
        }
        self.setup()

    def setup(self):
        self.tp_list = [0] * self.num_classes
        self.total_list = [0] * self.num_classes
        self.f_list = [0] * self.num_classes
        self.j_list = [0] * self.num_classes
        self.n_list = [0] * self.num_classes
        
        self.tp_small_list = [0] * self.num_classes
        self.tp_medium_list = [0] * self.num_classes
        self.tp_large_list = [0] * self.num_classes
        self.total_small_list = [0] * self.num_classes
        self.total_medium_list = [0] * self.num_classes
        self.total_large_list = [0] * self.num_classes

        self.sample_details = [[] for _ in range(self.num_classes)] 

        self.f_score = [0] * self.num_classes
        self.j_score = [0] * self.num_classes
        self.iou_list = [0] * self.num_classes
        self.iou_small_list = [0] * self.num_classes
        self.iou_medium_list = [0] * self.num_classes
        self.iou_large_list = [0] * self.num_classes

        self.pt_hit_list = [0] * self.num_classes
        self.pt_total_list = [0] * self.num_classes
        self.all_pt_hit_list = [0] * self.num_classes
        self.all_pt_total_list = [0] * self.num_classes

        self.pt_accuracy = [0] * self.num_classes
        self.all_point_accuracy = [0] * self.num_classes

        self.pt_sum_ratios = [0] * self.num_classes
        self.pt_ratios_count = [0] * self.num_classes
        self.all_pt_sum_ratios = [0] * self.num_classes
        self.all_pt_ratios_count = [0] * self.num_classes

        self.pt_accuracy_macro = [0] * self.num_classes
        self.all_point_accuracy_macro = [0] * self.num_classes

        self.blob_results = [[] for _ in range(self.num_classes)]
        
    def evaluate_points_overlap(self, rescaled_pts, gt_mask):
        """
        rescaled_pts: numpy array of shape [N, 2] or [N, 1, 2] containing points in un-normalized image dimensions
        gt_mask: 2D numpy array or tensor representing the ground truth mask for the query image
        """
        if rescaled_pts is None:
            return 0, 0
            
        if torch.is_tensor(rescaled_pts):
            rescaled_pts = rescaled_pts.cpu().numpy()
            
        if len(rescaled_pts.shape) == 3:
            rescaled_pts = rescaled_pts[:, 0, :]
            
        gt_arr = np.array(gt_mask)
        if gt_arr.ndim > 2:
            gt_arr = gt_arr.squeeze()
            
        gt_arr = gt_arr.astype(bool)
        h, w = gt_arr.shape
        
        hits = 0
        total = len(rescaled_pts)
        if total == 0:
            return 0, 0
            
        for x, y in rescaled_pts:
            px = int(np.round(np.clip(x, 0, w - 1)))
            py = int(np.round(np.clip(y, 0, h - 1)))
            if gt_arr[py, px]:
                hits += 1
                
        return hits, total

    def update_evl(self, class_idx, ground_truth_masks, pred_masks, sample_id=None, points_list=None, all_points_list=None):
        """
        Update evaluation metrics for a single sample
        
        Args:
            class_idx: class index for this sample
            ground_truth_masks: ground truth masks for N frames
            pred_masks: predicted masks for N frames
        """
        assert sample_id is not None, "Sample ID is required for stratified evaluation"
        
        # Convert class ID to internal index position
        id = self.class_indexes.index(class_idx)
        
        # Process each frame for F-score and J-score
        num_frames = len(ground_truth_masks)
        for j in range(num_frames):

            # Compute IoU metrics using the existing test_in_train method
            pred_mask = [pred_masks[j]]
            ground_truth_mask = [ground_truth_masks[j]]

            tp, total = self.test_in_train(ground_truth_mask, pred_mask)

            thresh = 0.5
            # Binarize ground truth and predictions
            y = ground_truth_masks[j] > thresh
            predict = pred_masks[j] > thresh
            
            # Convert to numpy and ensure proper shape for DAVIS evaluation functions
            y_np = np.array(y, dtype=np.uint8)
            predict_np = np.array(predict, dtype=np.uint8)
            
            area = y_np.sum()
            ratio = area / (y_np.shape[0] * y_np.shape[1])

            # Remove any extra dimensions and ensure 2D first for the DAVIS evaluation functions
            if y_np.ndim > 2:
                y_np = y_np.squeeze()
            if predict_np.ndim > 2:
                predict_np = predict_np.squeeze()

            # Accumulate boundary F-score and IoU J-score
            x = db_eval_boundary(predict_np, y_np)
            y = db_eval_iou(y_np, predict_np)
            
            if self.verbose:
                print(f"Frame {j}: Boundary F-score = {x}, IoU J-score = {y}")
                print(f"Accumulated F-score before update: {self.f_list[id]}, J-score before update: {self.j_list[id]}, Count: {self.n_list[id]}")
            
            self.f_list[id] += x
            self.j_list[id] += y
            self.n_list[id] += 1

            if ratio < self.threshold["SMALL"]:
                self.tp_small_list[id] += tp
                self.total_small_list[id] += total
            elif ratio < self.threshold["MEDIUM"]:
                self.tp_medium_list[id] += tp
                self.total_medium_list[id] += total
            else: # everything else is large
                self.tp_large_list[id] += tp
                self.total_large_list[id] += total


            pt_hits, pt_total = 0, 0
            if points_list is not None and j < len(points_list):
                pt_hits, pt_total = self.evaluate_points_overlap(points_list[j], ground_truth_masks[j])
            
            all_pt_hits, all_pt_total = 0, 0
            if all_points_list is not None and j < len(all_points_list):
                all_pt_hits, all_pt_total = self.evaluate_points_overlap(all_points_list[j], ground_truth_masks[j])

            pt_score = float(pt_hits) / pt_total if pt_total > 0 else -1.0
            all_pt_score = float(all_pt_hits) / all_pt_total if all_pt_total > 0 else -1.0

            self.pt_hit_list[id] += pt_hits
            self.pt_total_list[id] += pt_total
            self.all_pt_hit_list[id] += all_pt_hits
            self.all_pt_total_list[id] += all_pt_total

            if pt_total > 0:
                self.pt_sum_ratios[id] += pt_score
                self.pt_ratios_count[id] += 1
            if all_pt_total > 0:
                self.all_pt_sum_ratios[id] += all_pt_score
                self.all_pt_ratios_count[id] += 1

            pts_for_blob = points_list[j] if (points_list is not None and j < len(points_list)) else None
            blob_results = blob_analysis(ground_truth_masks[j], pts_for_blob) if pts_for_blob is not None else []

            # sample_details contains the IoU for each individual image, and the relative pixel ratio w.r.t. the gt. We cannot use the tp/total because that's an incremental approach, and we need an atomic value for each image
            self.sample_details[id].append({
                "sample_id": sample_id,
                "j_score": y,
                "f_score": x,
                "pixel_ratio": ratio,
                "point_score": pt_score,
                "all_point_score": all_pt_score,
                "num_total_points": len(all_points_list[j]) if (all_points_list is not None and j < len(all_points_list) and all_points_list[j] is not None) else 0, # record the total number of points computed by Matcher upon this sample
                "size_category": "SMALL" if ratio < self.threshold["SMALL"] else "MEDIUM" if ratio < self.threshold["MEDIUM"] else "LARGE",
                "blob_results": blob_results
            })

            if self.verbose:
                print(f"Accumulated F-score after update: {self.f_list[id]}, J-score after update: {self.j_list[id]}, Count: {self.n_list[id]}")

            # Accumulate IoU metrics
            self.tp_list[id] += tp
            self.total_list[id] += total
        
        # Recompute final scores (same logic as original)
        self.iou_list = [self.tp_list[ic] /
                        float(max(self.total_list[ic], 1))
                        for ic in range(self.num_classes)]
        self.iou_small_score = [self.tp_small_list[ic] /
                        float(max(self.total_small_list[ic], 1))
                        for ic in range(self.num_classes)]
        self.iou_medium_score = [self.tp_medium_list[ic] /
                        float(max(self.total_medium_list[ic], 1))
                        for ic in range(self.num_classes)]
        self.iou_large_score = [self.tp_large_list[ic] /
                        float(max(self.total_large_list[ic], 1))
                        for ic in range(self.num_classes)]
        self.f_score = [self.f_list[ic] /
                        float(max(self.n_list[ic], 1))
                        for ic in range(self.num_classes)]
        self.j_score = [self.j_list[ic] /
                        float(max(self.n_list[ic], 1))
                        for ic in range(self.num_classes)]
        self.pt_accuracy = [self.pt_hit_list[ic] /
                           float(max(self.pt_total_list[ic], 1))
                           for ic in range(self.num_classes)]
        self.all_point_accuracy = [self.all_pt_hit_list[ic] /
                                  float(max(self.all_pt_total_list[ic], 1))
                                  for ic in range(self.num_classes)]
        self.pt_accuracy_macro = [self.pt_sum_ratios[ic] /
                                 float(max(self.pt_ratios_count[ic], 1))
                                 for ic in range(self.num_classes)]
        self.all_point_accuracy_macro = [self.all_pt_sum_ratios[ic] /
                                        float(max(self.all_pt_ratios_count[ic], 1))
                                        for ic in range(self.num_classes)]

    def test_in_train(self, query_label, pred):
        assert len(query_label) == len(pred)
        total_tp = 0
        total_total = 0
        
        # Process each frame individually
        num_frames = len(query_label)
        for frame_idx in range(num_frames):
            # Extract single frame
            frame_gt = query_label[frame_idx]     # H*W
            frame_pred = pred[frame_idx]          # H*W
            
            # Compute metrics for this frame
            tp, tn, fp, fn = measure(frame_gt, frame_pred)
            total_tp += tp
            total_total += (tp + fp + fn)
        
        return total_tp, total_total

