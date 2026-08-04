import cv2 
import numpy as np

def compute_blobs(gt_mask):
    """ Compute the connected components of the gt_mask and return the set of blobs and the label map """
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(gt_mask, connectivity=8)
    # labels[y, x] == 0 is background; == k means blob k (1-indexed)
    # labels is a 2D array with shape (H, W), values are 0, 1, 2, ..., n_labels-1
    # the pixel value of the labels map identifies the blob
    
    return stats[1:], labels

def compute_blob_ratio(gt_mask, blob):
    """ Compute the blob's pixel ratio w.r.t. the gt_mask dimensions """
    total_pixels = gt_mask.shape[0] * gt_mask.shape[1]
    blob_pixels = blob[cv2.CC_STAT_AREA]
    return blob_pixels / total_pixels

def compute_n_points_in_blob(blob_label_id, labels, points):
    """ 
        Compute the number of points that fall exactly within the blob pixels.
        Args:
            blob_label_id: 1-indexed label value of this blob in the label map
            labels: 2D label map from cv2.connectedComponentsWithStats (H x W, dtype int32)
            points: numpy array of shape [N, 2] with (x, y) pixel coordinates (x=col, y=row)
        Returns:
            number of points whose pixel belongs to this blob
    """
    h, w = labels.shape
    # points[:, 0] are the columns (x)
    # points[:, 1] are the rows (y)
    # clip is used for security due to floating point operations before rounding, for the very rare cases where points are slightly outside image boundaries
    # if the points are computed by rounding and clipping, then this operation is not strictly necessary, but it's a good practice to keep it for robustness
    cols = np.round(np.clip(points[:, 0], 0, w - 1)).astype(np.intp)
    rows = np.round(np.clip(points[:, 1], 0, h - 1)).astype(np.intp)
    return int((labels[rows, cols] == blob_label_id).sum())

def blob_analysis(gt_mask=None, points=None):
    """ 
        Compute the number of points that fall within the blob and the blob ratio
        Args:
            gt_mask: ground truth mask
            points: list of points (x, y) not normalized
        Returns:
            list of dicts with the number of points that fall within each blob and the blob ratio
    """
    if gt_mask is None:
        raise ValueError("gt_mask is required")
    if points is None:
        raise ValueError("points is required")

    if hasattr(gt_mask, "cpu"):
        gt_mask = gt_mask.cpu()
    if gt_mask.ndim > 2:
        gt_mask = gt_mask.squeeze()
    gt_mask = gt_mask.numpy().astype(np.uint8)

    # Normalise points shape to [N, 2]
    pts = np.array(points)
    if pts.ndim == 3:          # [N, 1, 2] → [N, 2]
        pts = pts[:, 0, :]

    blobs, labels = compute_blobs(gt_mask)
    # blob_label_id is 1-indexed (label 0 = background)
    points_in_blobs = [
        compute_n_points_in_blob(blob_idx + 1, labels, pts)
        for blob_idx in range(len(blobs))
    ]
    blob_ratios = [compute_blob_ratio(gt_mask, blob) for blob in blobs]
    
    results = [
        {
            "BlobID" : blob_idx,
            "n_points_in_blob" : n_points,
            "blob_dim_ratio" : blob_ratio
        }
        for blob_idx, (n_points, blob_ratio) in enumerate(zip(points_in_blobs, blob_ratios))
    ]

    return results


    