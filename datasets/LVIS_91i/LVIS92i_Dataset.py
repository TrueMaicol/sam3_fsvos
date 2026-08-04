r""" LVIS-92i few-shot semantic segmentation dataset """
import os
import pickle
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import PIL.Image as Image
import numpy as np
import pandas as pd
from lvis import LVIS  # Used for category metadata

# If detectron2 is not installed, we can use a fallback for polygon to bitmask
try:
    from detectron2.structures.masks import polygons_to_bitmask
except ImportError:
    # Minimal fallback using cv2
    import cv2
    def polygons_to_bitmask(polygons, height, width):
        mask = np.zeros((height, width), dtype=np.uint8)
        for poly in polygons:
            p = np.array(poly).reshape(-1, 2).astype(np.int32)
            cv2.fillPoly(mask, [p], 1)
        return mask

import pycocotools.mask as mask_util

class LVIS92i_Dataset(Dataset):
    def __init__(self, data_dir, fold, transform, split, shot, use_original_imgsize, use_synset_names=False, synset_mapping_csv_path=None):
        self.split = 'val' if split in ['val', 'test'] else 'trn'
        self.fold = fold
        self.nfolds = 10
        self.benchmark = 'lvis-92i'
        self.shot = shot
        self.data_dir = data_dir
        # Pickles are directly in the LVIS/ directory
        self.anno_path = data_dir
        # Images are in the coco/ subdirectory
        self.base_path = os.path.join(data_dir, "coco")
        self.transform = transform
        self.use_original_imgsize = use_original_imgsize
        self.use_synset_names = use_synset_names

        # Load annotation metadata
        self.nclass, self.class_ids_ori, self.img_metadata_classwise = self.build_img_metadata_classwise()
        self.class_ids_c = {cid: i for i, cid in enumerate(self.class_ids_ori)}
        self.class_ids = sorted(list(self.class_ids_c.values()))

        # Initialize LVIS API just for category names
        # Looking for it in the same directory as pickles
        lvis_json = os.path.join(data_dir, 'lvis_v1_val.json')
        if os.path.exists(lvis_json):
            self.lvis_api = LVIS(lvis_json)
        else:
            self.lvis_api = None

        # Build idx_to_classname (0-N mapping)
        self.idx_to_classname = {}
        self.class_idx_to_all_lemmas = {}
        self.idx_to_ground_truth_label = {}
        
        if self.use_synset_names and synset_mapping_csv_path:
            synset_mapping = pd.read_csv(synset_mapping_csv_path, sep="|")
            for idx, ori_id in enumerate(self.class_ids_ori):
                match = synset_mapping[synset_mapping['idx'] == ori_id]
                if not match.empty:
                    selected_lemma = match['selected_lemma'].values[0]
                    if pd.notna(selected_lemma):
                        self.idx_to_classname[idx] = str(selected_lemma).split(",")[0].replace("_", " ")
                    
                    lemmas_str = match['lemmas'].values[0] if 'lemmas' in synset_mapping.columns else None
                    if pd.notna(lemmas_str):
                        self.class_idx_to_all_lemmas[idx] = [l.replace("_", " ") for l in str(lemmas_str).split(",")]
                else:
                    self.idx_to_classname[idx] = self.lvis_api.load_cats([ori_id])[0]['name'] if self.lvis_api else f"cat_{ori_id}"
                self.idx_to_ground_truth_label[idx] = self.lvis_api.load_cats([ori_id])[0]['name'] if self.lvis_api else f"cat_{ori_id}"
        else:
            for idx, ori_id in enumerate(self.class_ids_ori):
                self.idx_to_classname[idx] = self.lvis_api.load_cats([ori_id])[0]['name'] if self.lvis_api else f"cat_{ori_id}"

        self.img_metadata = self.build_img_metadata()

    def __len__(self):
        return len(self.img_metadata) if self.split == 'trn' else 2300

    def __getitem__(self, idx):
        if idx >= self.__len__():
            raise IndexError

        # Sample an episode
        query_img, query_mask, support_imgs, support_masks, class_idx, query_name = self.load_frame()

        if self.transform is not None:
            query_img, query_mask = self.transform([query_img], [query_mask])
            if self.shot > 0:
                support_imgs, support_masks = self.transform(support_imgs, support_masks)
            else:
                support_imgs = torch.tensor([])
                support_masks = torch.tensor([])
            
        return query_img, query_mask, support_imgs, support_masks, int(class_idx), query_name

    def build_img_metadata_classwise(self):
        with open(os.path.join(self.anno_path, 'lvis_train.pkl'), 'rb') as f:
            train_anno = pickle.load(f)
        with open(os.path.join(self.anno_path, 'lvis_val.pkl'), 'rb') as f:
            val_anno = pickle.load(f)

        train_cat_ids = sorted(list(train_anno.keys()))
        val_cat_ids = sorted([i for i in list(val_anno.keys()) if len(val_anno[i]) > self.shot])

        nclass_val_split = len(val_cat_ids) // self.nfolds
        class_ids_val = [val_cat_ids[self.fold + self.nfolds * v] for v in range(nclass_val_split)]
        class_ids_trn = [x for x in train_cat_ids if x not in class_ids_val]

        class_ids_ori = class_ids_trn if self.split == 'trn' else class_ids_val
        nclass = len(class_ids_ori)
        img_metadata_classwise = train_anno if self.split == 'trn' else val_anno

        return nclass, class_ids_ori, img_metadata_classwise

    def build_img_metadata(self):
        img_metadata = []
        for k in self.img_metadata_classwise.keys():
            img_metadata.extend(list(self.img_metadata_classwise[k].keys()))
        return sorted(list(set(img_metadata)))

    def get_mask(self, segm, image_size):
        if isinstance(segm, list):
            mask = polygons_to_bitmask(segm, image_size[1], image_size[0])
        elif isinstance(segm, dict):
            mask = mask_util.decode(segm)
        else:
            mask = np.array(segm)
        return mask.astype(np.uint8)

    def load_frame(self):
        class_idx = np.random.randint(0, len(self.class_ids_ori))
        class_sample = self.class_ids_ori[class_idx]

        query_name = np.random.choice(list(self.img_metadata_classwise[class_sample].keys()), 1, replace=False)[0]
        query_info = self.img_metadata_classwise[class_sample][query_name]
        query_img = np.array(Image.open(os.path.join(self.base_path, query_name)).convert('RGB'))
        
        # Aggregate all annotations for the query image of this class
        query_mask = np.zeros(query_img.shape[:2], dtype=np.uint8)
        for anno in query_info['annotations']:
            query_mask = np.maximum(query_mask, self.get_mask(anno['segmentation'], (query_img.shape[1], query_img.shape[0])))

        support_names = []
        while len(support_names) < self.shot:
            s_name = np.random.choice(list(self.img_metadata_classwise[class_sample].keys()), 1, replace=False)[0]
            if s_name != query_name:
                support_names.append(s_name)

        support_imgs, support_masks = [], []
        for s_name in support_names:
            support_img = np.array(Image.open(os.path.join(self.base_path, s_name)).convert('RGB'))
            support_imgs.append(support_img)
            
            s_info = self.img_metadata_classwise[class_sample][s_name]
            s_mask = np.zeros(support_img.shape[:2], dtype=np.uint8)
            for anno in s_info['annotations']:
                s_mask = np.maximum(s_mask, self.get_mask(anno['segmentation'], (support_img.shape[1], support_img.shape[0])))
            support_masks.append(s_mask)

        return query_img, query_mask, support_imgs, support_masks, class_idx, query_name

    def get_class_ids(self):
        return self.class_ids