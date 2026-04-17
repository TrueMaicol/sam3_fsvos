r""" PASCAL-5i few-shot semantic segmentation dataset """
import os
import torch
from torch.utils.data import Dataset
import PIL.Image as Image
import numpy as np
import pandas as pd

class PASCAL5i_Dataset(Dataset):
    def __init__(self, data_dir, fold, transform, split, shot, use_original_imgsize, use_synset_names=False, synset_mapping_csv_path=None):
        self.split = 'val' if split in ['val', 'test'] else 'trn'
        self.fold = fold
        self.nfolds = 4
        self.nclass = 20
        self.benchmark = 'pascal-5i'
        self.shot = shot
        self.data_dir = data_dir
        # Structure: datapath/VOCdevkit/VOC2012/...
        self.base_path = os.path.join(data_dir, 'VOCdevkit', 'VOC2012')
        self.img_path = os.path.join(self.base_path, 'JPEGImages')
        self.ann_path = os.path.join(self.base_path, 'SegmentationClassAug')
        self.transform = transform
        self.use_original_imgsize = use_original_imgsize
        self.use_synset_names = use_synset_names

        # Class IDs for this fold
        self.class_ids = self.build_class_ids()
        self.all_cats = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", 
                         "diningtable", "dog", "horse", "motorbike", "person", "potted plant", "sheep", "sofa", "train", "tv/monitor"]
        
        # Build idx_to_classname (Standardized 0-19 mapping)
        self.idx_to_classname = {}
        self.class_idx_to_all_lemmas = {}
        
        if self.use_synset_names and synset_mapping_csv_path:
            synset_mapping = pd.read_csv(synset_mapping_csv_path, sep="|")
            for idx, name in enumerate(self.all_cats):
                match = synset_mapping[synset_mapping['idx'] == idx]
                if not match.empty:
                    selected_lemma = match['selected_lemma'].values[0]
                    if pd.notna(selected_lemma):
                        self.idx_to_classname[idx] = str(selected_lemma).split(",")[0].replace("_", " ")
                    
                    lemmas_str = match['lemmas'].values[0] if 'lemmas' in synset_mapping.columns else None
                    if pd.notna(lemmas_str):
                        self.class_idx_to_all_lemmas[idx] = [l.replace("_", " ") for l in str(lemmas_str).split(",")]
                else:
                    self.idx_to_classname[idx] = name
        else:
            for idx, name in enumerate(self.all_cats):
                self.idx_to_classname[idx] = name

        self.img_metadata = self.build_img_metadata()
        self.img_metadata_classwise = self.build_img_metadata_classwise()

    def __len__(self):
        return len(self.img_metadata) if self.split == 'trn' else 1000

    def __getitem__(self, idx):
        if idx >= self.__len__():
            raise IndexError

        query_img, query_mask, support_imgs, support_masks, class_idx, query_name = self.load_frame()

        if self.transform is not None:
            query_img, query_mask = self.transform([query_img], [query_mask])
            support_imgs, support_masks = self.transform(support_imgs, support_masks)
            
        return query_img, query_mask, support_imgs, support_masks, int(class_idx), query_name

    def build_class_ids(self):
        nclass_trn = self.nclass // self.nfolds
        class_ids_val = [self.fold * nclass_trn + i for i in range(nclass_trn)]
        class_ids_trn = [x for x in range(self.nclass) if x not in class_ids_val]
        return class_ids_trn if self.split == 'trn' else class_ids_val

    def build_img_metadata(self):
        def read_metadata(split, fold_id):
            # Assumes splits folder is in the data_dir
            fold_n_metadata = os.path.join(self.data_dir, 'splits', split, f'fold{fold_id}.txt')
            if not os.path.exists(fold_n_metadata):
                # Fallback to absolute workspace path if provided path fails
                fold_n_metadata = f'/leonardo_work/IscrC_MARSv2/SAM3_FSVOS/src/datasets/pascal_5i/splits/{split}/fold{fold_id}.txt'
            
            with open(fold_n_metadata, 'r') as f:
                lines = f.read().splitlines()
            return [[l.split('__')[0], int(l.split('__')[1]) - 1] for l in lines]

        img_metadata = []
        if self.split == 'trn':
            for fold_id in range(self.nfolds):
                if fold_id != self.fold:
                    img_metadata += read_metadata('trn', fold_id)
        else:
            img_metadata = read_metadata('val', self.fold)
        return img_metadata

    def build_img_metadata_classwise(self):
        img_metadata_classwise = {cid: [] for cid in range(self.nclass)}
        for img_name, img_class in self.img_metadata:
            img_metadata_classwise[img_class].append(img_name)
        return img_metadata_classwise

    def read_mask(self, name, class_idx):
        mask_path = os.path.join(self.ann_path, name + '.png')
        mask = np.array(Image.open(mask_path))
        binary_mask = np.zeros(mask.shape[:2], dtype=np.uint8)
        binary_mask[mask == class_idx + 1] = 1
        return binary_mask

    def load_frame(self):
        class_idx = np.random.choice(self.class_ids)
        query_name = np.random.choice(self.img_metadata_classwise[class_idx])
        query_img = np.array(Image.open(os.path.join(self.img_path, query_name + '.jpg')).convert('RGB'))
        query_mask = self.read_mask(query_name, class_idx)

        support_names = []
        while len(support_names) < self.shot:
            s_name = np.random.choice(self.img_metadata_classwise[class_idx])
            if s_name != query_name:
                support_names.append(s_name)

        support_imgs, support_masks = [], []
        for s_name in support_names:
            support_imgs.append(np.array(Image.open(os.path.join(self.img_path, s_name + '.jpg')).convert('RGB')))
            support_masks.append(self.read_mask(s_name, class_idx))

        return query_img, query_mask, support_imgs, support_masks, class_idx, query_name

    def get_class_ids(self):
        return self.class_ids