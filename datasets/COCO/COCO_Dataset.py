from pycocotools.coco import COCO
import os 
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import pandas as pd

import nltk
from nltk.corpus import wordnet as wn

class COCO_Dataset(Dataset): 
    def __init__(self, data_dir=None, split='val', transform=None, seed=None, use_synset_names=False, synset_mapping_csv_path=None):
        self.data_dir = data_dir
        self.split = split
        self.transform = transform
        self.seed = seed
        self.benchmark = "coco"
        self.use_synset_names = use_synset_names

        if synset_mapping_csv_path is None:
            self.synset_mapping_csv_path = os.path.join(data_dir, "synset_mapping.csv")
        else:
            self.synset_mapping_csv_path = synset_mapping_csv_path

        self.coco = COCO(os.path.join(data_dir, 'annotations', 'instances_{}2017.json'.format(split)))
        self.class_ids = self.coco.getCatIds()

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
                    self.class_idx_to_all_lemmas[idx] = [l.replace("_", " ") for l in str(lemmas_str).split(",")]
                
        else:
            self.idx_to_classname = {idx: self.coco.loadCats(idx)[0]['name'] for idx in self.class_ids}

        self.img_per_cat = {}
        self.img_ids = []
        for cat_id in self.class_ids:
            for img_id in set(self.coco.getImgIds(catIds=cat_id)):
                self.img_ids.append((cat_id, img_id))
                if cat_id not in self.img_per_cat:
                    self.img_per_cat[cat_id] = 0
                self.img_per_cat[cat_id] += 1
    
    def get_class_ids(self):
        return self.class_ids
        
    def __len__(self):
        return len(self.img_ids)
        
    def __getitem__(self, idx):
        cat_id, img_id = self.img_ids[idx]
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.data_dir, f'{self.split}2017', img_info['file_name'])
        img = np.array(Image.open(img_path).convert('RGB'))
        
        ann_ids = self.coco.getAnnIds(imgIds=img_id, catIds=cat_id, iscrowd=None)
        anns = self.coco.loadAnns(ann_ids)
        mask = np.zeros((img_info['height'], img_info['width']), dtype=np.uint8)
        for ann in anns:
            instance_mask = self.coco.annToMask(ann)
            mask = np.maximum(mask, instance_mask)
            
        if self.transform is not None:
            img, mask = self.transform([img], [mask])
        
        return img, mask, cat_id, img_id


if __name__ == "__main__":
    dataset = COCO_Dataset(data_dir="/leonardo_work/IscrC_MARSv2/datasets/COCO", split="val", seed=1000)
    nltk.data.path.append('/leonardo_work/IscrC_MARSv2/datasets/NLTK_WORDNET')

    print("ID-LABEL-SYNSET")
    print("------------------------------------------")
    for idx in dataset.coco.getCatIds():
        label = "_".join(dataset.idx_to_classname[idx].split())
        syn = wn.synsets(label, pos=wn.NOUN)
        if syn is None or len(syn) == 0:
                syn = "None"
        else:
            syn = syn[0].name()
        print(f"{idx}-{dataset.idx_to_classname[idx]}-{syn}")    