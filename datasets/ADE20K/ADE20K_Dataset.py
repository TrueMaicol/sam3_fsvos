from torch.utils.data import Dataset
import os
import cv2
import numpy as np
from PIL import Image
from .utils_ade20k import loadAde20K
import pickle
import json
import sys
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from YoutubeFSVOS.transform import TestTransform

class ADE20K_Dataset(Dataset): 
    def __init__(self, data_dir=None, split='val', transform=None, seed=None, use_synset_names=False, synset_mapping_csv_path=None, all_lemmas=False, use_grouping=False):
        self.data_dir = data_dir
        self.split = split
        self.transform = transform
        self.seed = seed
        self.benchmark = "ade20k" # Consistent with COCO dataset
        self.index_file = "index_ade20k.pkl" 
        self.use_synset_names = use_synset_names
        self.all_lemmas = all_lemmas
        
        self.use_grouping = use_grouping

        if synset_mapping_csv_path is None:
            self.synset_mapping_csv_path = os.path.join(data_dir, "synset_mapping.csv")
        else:
            self.synset_mapping_csv_path = synset_mapping_csv_path

        if self.use_grouping:
            grouping_json_path = os.path.join(os.path.dirname(self.synset_mapping_csv_path), "ADE20K_grouping.json")
            with open(grouping_json_path, 'r') as f:
                self.grouping_data = json.load(f)
            self.new_to_old_ids = {} # maps new_grouped_id to a list of original masks
            self.old_to_new_id = {}  # maps original pixel mask to new_grouped_id
            for new_id_str, metadata in self.grouping_data.items():
                new_id = int(new_id_str) # JSON is now 1-based, like ADE20K mask pixel values
                old_ids = metadata['id_list']
                self.new_to_old_ids[new_id] = old_ids
                for old_id in old_ids:
                    self.old_to_new_id[old_id] = new_id
        self.index = self.load_index()
        if self.index is None:
            raise FileNotFoundError(f"Index file {self.index_file} not found in {data_dir}")
        else:
            print("Index loaded correctly")
            # print(f"index dict: {self.index.keys()}")
            
        self.img_ids = [] # List of (class_idx, img_idx)
        self.construct_dataset()

        # cat_id is the same as the pixel value contained in the objects.txt file and the pixel value for the masks
        self.class_ids = list(set([int(cat_id) for cat_id, _ in self.img_ids])) # the list conversion is required by the Evaluator class

        if self.use_grouping:
            self.idx_to_classname = {}
            self.class_idx_to_all_lemmas = {}
            for new_id in self.class_ids:
                if str(new_id) in self.grouping_data:
                    metadata = self.grouping_data[str(new_id)]
                    if self.use_synset_names:
                        self.idx_to_classname[new_id] = metadata['selected_lemma'].replace("_", " ")
                        self.class_idx_to_all_lemmas[new_id] = [l.strip().replace("_", " ") for l in metadata['lemmas'].split(",")]
                    else:
                        # Use native labels from JSON instead of synset lemmas
                        self.idx_to_classname[new_id] = metadata['labels'][0].replace("_", " ")
                        self.class_idx_to_all_lemmas[new_id] = [l.replace("_", " ") for l in metadata['labels']]
                else:
                    self.idx_to_classname[new_id] = f"unknown_{new_id}"
                    self.class_idx_to_all_lemmas[new_id] = [f"unknown_{new_id}"]

        elif self.use_synset_names:
            synset_mapping = pd.read_csv(self.synset_mapping_csv_path, sep="|")
            self.idx_to_classname = {}
            self.class_idx_to_all_lemmas = {}
            for idx in self.class_ids:
                match = synset_mapping[synset_mapping['idx'] == idx]
                selected_lemma = match['selected_lemma']
                
                if len(selected_lemma) > 0 and pd.notna(selected_lemma.values[0]):
                    self.idx_to_classname[idx] = str(selected_lemma.values[0]).split(",")[0].replace("_", " ")
                else:
                    raise ValueError("No match found for {}".format(idx))
                
                lemmas_str = match['lemmas'].values[0]
                if pd.notna(lemmas_str):
                    self.class_idx_to_all_lemmas[idx] = [l.replace("_", " ") for l in lemmas_str.split(",")] 
        else:
            self.idx_to_classname = {idx: self.index['objectnames'][idx-1].split(",")[0] for idx in self.class_ids} # idx-1 to go back to 0-index
        
        # if we are using all the lemmas we shall have a number of classes that equal to the total number of lemmas
        if all_lemmas:
            self.class_ids = list(set([int(lemma["global_idx"]) for class_idx, lemma in class_idx_to_lemmas.items()]))

        
    def load_index(self):
        print("Loading index...")
        index_path = os.path.join(self.data_dir, self.index_file)
        if os.path.isfile(index_path):
            with open(index_path, 'rb') as f:
                return pickle.load(f)
        else:
            return None
            
    def get_class_ids(self):
        # In ADE20K index, objectnames list defines the classes implicitly by index
        return self.class_ids

    def construct_dataset(self):
        # Filter images by split and collect (class, img) pairs
        folders = self.index['folder']
        object_presence = self.index['objectPresence'] # Shape [C, N]
        
        num_images = len(self.index['filename'])
        
        # Determine split keyword
        # ADE20K folders typically contain 'training' or 'validation'
        split_key = 'validation'
        self.img_per_cat = {} # counter of images per class
        for img_idx in range(num_images):
            # Check folder for split
            if split_key not in folders[img_idx]:
                continue
                
            # Find classes present in this image
            # object_presence is [C, N]. Slice for this image.
            # We want indices where count > 0
            # Since these are indexes (starting at 0), the cat_id starts at 1, therefore they are scaled -1 w.r.t. mask pixel values
            cat_ids = np.where(object_presence[:, img_idx] > 0)[0]
            # print("\n")
            # print(f"All classes ids: {cat_ids}")
            cat_ids = [cat_id for cat_id in cat_ids if self.index['objectIsPart'][cat_id, img_idx] == 0]
            # print(f"cat_ids for image {self.index['filename'][img_idx]}")
            # print(f"Only objects ids: {cat_ids}")
            
            if self.use_grouping:
                final_cat_ids = set()
                for cat_id in cat_ids:
                    old_id = cat_id + 1
                    if old_id in self.old_to_new_id: # Only retain objects explicitly present in group mapping
                        final_cat_ids.add(self.old_to_new_id[old_id])
                final_cat_ids = list(final_cat_ids)
            else:
                final_cat_ids = [cat_id + 1 for cat_id in cat_ids]

            for cat_id in final_cat_ids:
                self.img_ids.append((cat_id, img_idx)) # cat_id mapped safely if using JSON
                if cat_id not in self.img_per_cat:
                    self.img_per_cat[cat_id] = 0
                self.img_per_cat[cat_id] += 1

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        cat_id, img_idx = self.img_ids[idx]
        
        # Retrieve file path details
        filename = self.index['filename'][img_idx]
        folder = "/".join(self.index['folder'][img_idx].split("/")[1:])
        # Construct full path
        # Assuming data_dir is the base where the ADE20K root folder resides, or is the root.
        # We try to join.
        img_path = os.path.join(self.data_dir, folder, filename)
        
        # Load image
        try:
            img = np.array(Image.open(img_path).convert("RGB"))
        except Exception as e:
            # Fallback or informative error
            raise FileNotFoundError(f"Could not load image at {img_path}. Error: {e}")

        # Load mask
        # loadAde20K expects the image path and finds _seg.png relative to it
        try:
            ade_info = loadAde20K(img_path)
        except Exception as e:
             raise FileNotFoundError(f"Could not load annotation for {mask_path}. Error: {e}")
        
        # Get class mask matrix (values are class IDs)
        class_mask_full = ade_info['class_mask']
        # print(f"unique values of mask: {np.unique(class_mask_full)}")
        
        # Create binary mask for the specific cat_id
        # COCO dataset returns a mask where the object is 1, others 0.
        mask = np.zeros_like(class_mask_full).astype(np.uint8)
        
        if getattr(self, 'use_grouping', False) and cat_id in self.new_to_old_ids:
            old_ids = self.new_to_old_ids[cat_id]
            mask[np.isin(class_mask_full, old_ids)] = 255
        else:
            mask[class_mask_full == cat_id] = 255
        
        if self.transform is not None:
            # Passing as lists to comport with transform signature seen in COCO
            img, mask = self.transform([img], [mask])
        
        return img, mask, cat_id, img_idx

if __name__ == "__main__":
    # Test block
    dataset = ADE20K_Dataset(
        data_dir="/leonardo_work/IscrC_MARSv2/datasets/ADE20K/ADE20K_2021_17_01", 
        split="val", 
        transform=TestTransform(size=518), 
        use_synset_names=True, 
        synset_mapping_csv_path="/leonardo_work/IscrC_MARSv2/datasets/synset_mappings/leaf/ADE20K.csv",
        use_grouping=True
    )
    
    print("ID|LABEL|NUM_IMAGES")
    print("------------------------------------")
    for idx in dataset.get_class_ids():
        print(f"{idx}|{dataset.idx_to_classname[idx]}|{dataset.img_per_cat[idx]}")
        