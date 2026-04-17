r""" COCO-20i few-shot semantic segmentation dataset """
import os
import pickle
import pandas as pd
from torch.utils.data import Dataset
import torch.nn.functional as F
import torch
import PIL.Image as Image
import numpy as np
from pycocotools.coco import COCO


COCO_ORIGINAL_IDS = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 
    24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 
    48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 67, 70, 
    72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90
]

# Standard COCO 0-79 mapping for few-shot benchmarks
COCO_ID_LABELS_MAPPING = {
    0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 
    9: 'traffic light', 10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat', 
    16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 
    24: 'backpack', 25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis', 
    31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove', 36: 'skateboard', 
    37: 'surfboard', 38: 'tennis racket', 39: 'bottle', 40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 
    44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli', 51: 'carrot', 
    52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed', 
    60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard', 
    67: 'cell phone', 68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book', 
    74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'
}

class COCO20i_Dataset(Dataset):
    def __init__(self, data_dir, fold, transform, split, shot, use_original_imgsize, use_synset_names=False, synset_mapping_csv_path=None):
        self.split = 'val' if split in ['val', 'test'] else 'trn'
        self.fold = fold
        self.nfolds = 4
        self.nclass = 80
        self.benchmark = 'coco-20i'
        self.shot = shot
        self.split_coco = split if split == 'val2014' else 'train2014'
        self.data_dir = data_dir
        self.transform = transform
        self.use_original_imgsize = use_original_imgsize
        self.use_synset_names = use_synset_names

        if synset_mapping_csv_path is None:
            self.synset_mapping_csv_path = os.path.join(data_dir, "synset_mapping.csv")
        else:
            self.synset_mapping_csv_path = synset_mapping_csv_path

        # Initialize COCO API for generating masks on the fly
        json_split = 'train2014' if self.split == 'trn' else 'val2014'
        ann_file = os.path.join(data_dir, 'annotations', f'instances_{json_split}.json')
        print(f"Loading COCO annotations from {ann_file}...")
        self.coco = COCO(ann_file)
        self.coco_id_to_idx = {id: i for i, id in enumerate(COCO_ORIGINAL_IDS)}

        self.class_ids = self.build_class_ids()

        self.class_idx_to_all_lemmas = {}
        if self.use_synset_names:
            synset_mapping = pd.read_csv(self.synset_mapping_csv_path, sep="|")
            self.idx_to_classname = {}
            
            for idx in self.class_ids:
                # Map the 0-79 index to the original COCO category ID (1-90)
                original_coco_id = COCO_ORIGINAL_IDS[idx]
                
                # Search the CSV using the original_coco_id
                match = synset_mapping[synset_mapping['idx'] == original_coco_id]
                
                if not match.empty:
                    selected_lemma = match['selected_lemma'].values[0]
                    if pd.notna(selected_lemma):
                        self.idx_to_classname[idx] = str(selected_lemma).split(",")[0].replace("_", " ")
                    
                    lemmas_str = match['lemmas'].values[0] if 'lemmas' in synset_mapping.columns else None
                    if pd.notna(lemmas_str):
                        self.class_idx_to_all_lemmas[idx] = [l.replace("_", " ") for l in str(lemmas_str).split(",")]
                else:
                    # Fallback to standard mapping if CSV lookup fails
                    print("No match found for COCO ID {}".format(original_coco_id))
                    self.idx_to_classname[idx] = COCO_ID_LABELS_MAPPING[idx]
        else:
            self.idx_to_classname = {idx: COCO_ID_LABELS_MAPPING[idx] for idx in self.class_ids}

        self.img_metadata_classwise = self.build_img_metadata_classwise()
        self.img_metadata = self.build_img_metadata()

    def __len__(self):
        return len(self.img_metadata) if self.split == 'trn' else 1000

    def __getitem__(self, idx):
        # IndexError is handled natively by Python, is equivalent of saying "the array is finished exit the iterator loop"
        if idx >= self.__len__():
            raise IndexError
        # ignores idx during training & testing and perform uniform sampling over object classes to form an episode
        # (due to the large size of the COCO dataset)
        query_img, query_mask, support_imgs, support_masks, query_name, support_names, class_sample, org_qry_imsize = self.load_frame()

        if self.transform is not None:
            query_img, query_mask = self.transform([query_img], [query_mask])
            support_imgs, support_masks = self.transform(support_imgs, support_masks)
            

        # query_mask = query_mask.float()
        # if not self.use_original_imgsize:
        #     query_mask = F.interpolate(query_mask.unsqueeze(0).unsqueeze(0).float(), query_img.size()[-2:], mode='nearest').squeeze()

        # support_imgs = torch.stack([self.transform(support_img) for support_img in support_imgs])
        # for midx, smask in enumerate(support_masks):
        #     support_masks[midx] = F.interpolate(smask.unsqueeze(0).unsqueeze(0).float(), support_imgs.size()[-2:], mode='nearest').squeeze()
        # support_masks = torch.stack(support_masks)

        # batch = {'query_img': query_img,
        #          'query_mask': query_mask,
        #          'query_name': query_name,

        #          'org_query_imsize': org_qry_imsize,

        #          'support_imgs': support_imgs,
        #          'support_masks': support_masks,
        #          'support_names': support_names,
        #          'class_id': torch.tensor(class_sample)}

        # class_sample -> class_idx inside the [0,79] range
        # query_name -> directory name of the query image
        return query_img, query_mask, support_imgs, support_masks, int(class_sample), query_name

    def get_class_ids(self):
        return self.class_ids

    def build_class_ids(self):
        nclass_trn = self.nclass // self.nfolds
        class_ids_val = [self.fold + self.nfolds * v for v in range(nclass_trn)]
        class_ids_trn = [x for x in range(self.nclass) if x not in class_ids_val]
        class_ids = class_ids_trn if self.split == 'trn' else class_ids_val

        return class_ids

    def build_img_metadata_classwise(self):
        with open(f'{self.data_dir}/splits/{self.split}/fold{self.fold}.pkl', 'rb') as f:
            img_metadata_classwise = pickle.load(f)
        return img_metadata_classwise

    def build_img_metadata(self):
        img_metadata = []
        for k in self.img_metadata_classwise.keys():
            img_metadata += self.img_metadata_classwise[k]
        return sorted(list(set(img_metadata)))

    def read_mask(self, name, class_idx):
        """Build a binary mask for the given class_idx (0-79) only.
        
        Only loads COCO annotations for the target category, avoiding
        the overwriting problem where larger overlapping annotations
        from other classes erase smaller target-class regions.
        """
        filename = os.path.basename(name)
        img_id = int(filename.split('_')[-1].split('.')[0])
        
        img_info = self.coco.loadImgs(img_id)[0]
        h, w = img_info['height'], img_info['width']
        
        target_cat_id = COCO_ORIGINAL_IDS[class_idx]
        ann_ids = self.coco.getAnnIds(imgIds=img_id, catIds=[target_cat_id])
        anns = self.coco.loadAnns(ann_ids)
        
        mask = np.zeros((h, w), dtype=np.uint8)
        for ann in anns:
            mask_fragment = self.coco.annToMask(ann)
            mask[mask_fragment > 0] = 1
                
        return mask

    def load_frame(self):
        class_sample = np.random.choice(self.class_ids, 1, replace=False)[0]
        query_name = np.random.choice(self.img_metadata_classwise[class_sample], 1, replace=False)[0]
        query_img = np.array(Image.open(os.path.join(self.data_dir, query_name)).convert('RGB'))
        query_mask = self.read_mask(query_name, class_sample)

        org_qry_imsize = query_img.size

        support_names = []
        while True:  # keep sampling support set if query == support
            support_name = np.random.choice(self.img_metadata_classwise[class_sample], 1, replace=False)[0]
            if query_name != support_name: support_names.append(support_name)
            if len(support_names) == self.shot: break

        support_imgs = []
        support_masks = []
        for support_name in support_names:
            support_imgs.append(np.array(Image.open(os.path.join(self.data_dir, support_name)).convert('RGB')))
            support_mask = self.read_mask(support_name, class_sample)
            support_masks.append(support_mask)

        return query_img, query_mask, support_imgs, support_masks, query_name, support_names, class_sample, org_qry_imsize