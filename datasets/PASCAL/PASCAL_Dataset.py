from torch.utils.data import Dataset
import os
import cv2
import numpy as np
from PIL import Image
import json
import pandas as pd


class PASCAL_Dataset(Dataset):

    def __init__(self, data_dir=None, split='val', transform=None, seed=None, use_synset_names=False, synset_mapping_csv_path=None):
        self.data_dir = data_dir
        self.split = split
        self.transform = transform
        self.seed = seed
        self.benchmark = "pascal"
        self.use_synset_names = use_synset_names
        
        if synset_mapping_csv_path is None:
            self.synset_mapping_csv_path = os.path.join(data_dir, "synset_mapping.csv")
        else:
            self.synset_mapping_csv_path = synset_mapping_csv_path
        self.img_ids = self.construct_dataset()


    def construct_dataset(self):
        with open(os.path.join(self.data_dir, 'class_mapping.json'), 'r') as f:
            self.class_mapping = json.load(f)
        self.class_ids = [v['id'] for v in self.class_mapping]

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
            self.idx_to_classname = {v['id']: v['label'] for v in self.class_mapping}
        img_ids = []
        split_file = os.path.join(self.data_dir, 'ImageSets', 'Segmentation', f'{self.split}.txt')
        with open(split_file, 'r') as f:
            self.file_names = [x.strip() for x in f.readlines()]
        
        self.masks = {}
        self.img_per_cat = {}

        for file_name in self.file_names:
            mask_file = os.path.join(self.data_dir, 'SegmentationClass', f'{file_name}.png')
            mask = np.array(Image.open(mask_file))
            self.masks[file_name] = mask
            present_classes = np.unique(mask)
            present_classes = present_classes[(present_classes != 255) & (present_classes != 0)]  # Exclude ignore index and background
            for class_id in present_classes:
                if class_id in self.class_ids:
                    img_ids.append((class_id, file_name))
                    if class_id not in self.img_per_cat:
                        self.img_per_cat[class_id] = 0
                    self.img_per_cat[class_id] += 1

        return img_ids

    def __len__(self):
        return len(self.img_ids)

    def get_class_ids(self):
        return self.class_ids

    def __getitem__(self, idx):
        class_id, file_name = self.img_ids[idx]
        img_file = os.path.join(self.data_dir, 'JPEGImages', f'{file_name}.jpg')

        image = np.array(Image.open(img_file).convert('RGB'))
        img_mask = self.masks[file_name]
        mask = np.zeros_like(img_mask).astype(np.uint8)
        mask[img_mask == class_id] = 255

        if self.transform is not None:
            image, mask = self.transform([image], [mask])

        return image, mask, class_id, file_name
            
    
if __name__ == "__main__":
    dataset = PASCAL_Dataset(data_dir='/leonardo_work/IscrC_MARSv2/datasets/PASCAL/VOC2012_train_val/VOC2012_train_val', split='val')
    print("dataset lenght: ", len(dataset))
    count = 0
    print("ID*LABEL*COUNT")
    for cat_id in dataset.get_class_ids():
        print(f"{cat_id}*{dataset.idx_to_classname[cat_id]}*{dataset.img_per_cat[cat_id]}")


