from lvis import LVIS
import os 
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import pandas as pd

class LVIS_Dataset(Dataset): 
    def __init__(self, data_dir=None, split='val', transform=None, seed=None, use_synset_names=False, synset_mapping_csv_path=None):
        self.data_dir = data_dir
        self.split = split
        self.transform = transform
        self.seed = seed
        self.benchmark = "lvis"
        self.use_synset_names = use_synset_names

        if synset_mapping_csv_path is None:
            self.synset_mapping_csv_path = os.path.join(data_dir, "synset_mapping.csv")
        else:
            self.synset_mapping_csv_path = synset_mapping_csv_path

        self.lvis = LVIS(os.path.join(data_dir, 'lvis_v1_{}.json'.format(split)))
        self.class_ids = [cat_id for cat_id in self.lvis.get_cat_ids() if len(self.lvis.cat_img_map[cat_id]) > 0]

        self.class_idx_to_all_lemmas = {}
        if self.use_synset_names:
            synset_mapping = pd.read_csv(self.synset_mapping_csv_path, sep="|")
            self.idx_to_classname = {}
            for idx in self.class_ids:
                match = synset_mapping[synset_mapping['idx'] == idx]
                selected_lemma = match['selected_lemma']
                if len(selected_lemma) > 0 and pd.notna(selected_lemma.values[0]):
                    self.idx_to_classname[idx] = str(selected_lemma.values[0]).split(",")[0].replace("_", " ")
                else:
                    print("No match found for {}".format(idx))

                lemmas_str = match['lemmas'].values[0] if 'lemmas' in synset_mapping.columns else None
                if lemmas_str is not None and pd.notna(lemmas_str):
                    print(f"lemmas_str: {lemmas_str}")
                    self.class_idx_to_all_lemmas[idx] = [l.replace("_", " ") for l in str(lemmas_str).split(",")]
                else:
                    print(f"No lemmas found for {idx}")
        else:
            self.idx_to_classname = {idx: self.lvis.load_cats([idx])[0]['name'] for idx in self.class_ids}

        # Build list of (class_idx, img_id) pairs
        # Since the same image can have multiple categories, we create a pair for each
        self.img_ids = []
        for cat_id in self.class_ids:
            # cat_img_map contains image_ids for each category
            img_ids_for_cat = self.lvis.cat_img_map[cat_id]
            # Use set to avoid duplicates within the same category
            for img_id in set(img_ids_for_cat):
                self.img_ids.append((cat_id, img_id))

        self.idx_to_synset = {idx: self.lvis.load_cats([idx])[0]['synset'] for idx in self.lvis.get_cat_ids()}
    
    def get_class_ids(self):
        return self.class_ids
        
    def __len__(self):
        return len(self.img_ids)
        
    def __getitem__(self, idx):
        cat_id, img_id = self.img_ids[idx]
        img_info = self.lvis.load_imgs([img_id])[0]
        
        # LVIS images come from COCO, construct the path
        # Image filename format: 000000XXXXXX.jpg (12 digits)
        split_folder = img_info['coco_url'].split('/')[-2]
        file_name = img_info['coco_url'].split('/')[-1]
        img_path = os.path.join(self.data_dir, split_folder, file_name)
        img = np.array(Image.open(img_path).convert('RGB'))
        
        # Get annotation IDs for this image and category
        ann_ids = self.lvis.get_ann_ids(img_ids=[img_id], cat_ids=[cat_id])
        anns = self.lvis.load_anns(ann_ids)
        
        # Create binary mask for this category
        mask = np.zeros((img_info['height'], img_info['width']), dtype=np.uint8)
        for ann in anns:
            mask = np.maximum(mask, self.lvis.ann_to_mask(ann))
        
        if self.transform is not None:
            img, mask = self.transform([img], [mask])
        
        return img, mask, cat_id, img_id

if __name__ == "__main__":
    dataset = LVIS_Dataset(data_dir="/leonardo_work/IscrC_MARSv2/datasets/LVIS", split="val", seed=1000)
    print(len(dataset))
    
    print("ID-LABEL-SYNSET")
    print("------------------------------------------")
    for idx in dataset.get_class_ids():
        print(f"{idx}|{dataset.idx_to_classname[idx]}")