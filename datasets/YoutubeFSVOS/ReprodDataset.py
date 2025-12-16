from torch.utils.data import Dataset
from torchvision import transforms
import os
import json
from PIL import Image
from tqdm import tqdm
import numpy as np

class ReprodDataset(Dataset):
    def __init__(self, dataset_path, fold=1, transform=None):
        """
        Args:
            data_list (list): List of data samples.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """

        self.transform = transform
        self.benchmark = "youtube-fsvos"
        self.fold = fold
        categories_path = os.path.join(os.path.dirname(__file__), 'categories.json')
        with open(categories_path, 'r') as f:
            categories = json.load(f)
        
        self.idx_to_classname = {cat['id']: cat['name'] for cat in categories}
        
        self.class_ids = [n + 1 for n in range(40) if n % 4 == (fold - 1)]
        self.data_list = self.load_video(dataset_path)

    def load_video(self, dataset_path, n_support_frames=5):
        test_dataset = []
        
        video_dirs = sorted([d for d in os.listdir(dataset_path) 
                            if os.path.isdir(os.path.join(dataset_path, d))])
        
        for dir in tqdm(video_dirs, desc="Loading videos"):
            class_id = int(dir.split("_")[1])
            temp = {}
            support_imgs = []
            support_masks = []
            query_imgs = []
            query_masks = []
            
            frames_dir = os.path.join(dataset_path, dir, "frames")
            n_frames = len(os.listdir(frames_dir))
            
            # Load all images first
            for j in range(n_frames):
                if j < n_support_frames:
                    img_path = os.path.join(dataset_path, dir, "frames", f"support_{j:04d}.jpg")
                    mask_path = os.path.join(dataset_path, dir, "ground_truth", f"support_{j:04d}.png")
                    support_imgs.append(np.array(Image.open(img_path)))
                    support_masks.append(np.array(Image.open(mask_path)))
                else:
                    img_path = os.path.join(dataset_path, dir, "frames", f"query_{j:04d}.jpg")
                    mask_path = os.path.join(dataset_path, dir, "ground_truth", f"query_{j:04d}.png")
                    query_imgs.append(np.array(Image.open(img_path)))
                    query_masks.append(np.array(Image.open(mask_path)))
            
            # Apply transform to entire sequences
            if self.transform is not None:
                support_imgs, support_masks = self.transform(support_imgs, support_masks)
                query_imgs, query_masks = self.transform(query_imgs, query_masks)
            
            temp["dir_name"] = dir
            temp["class_id"] = class_id
            temp["support_imgs"] = support_imgs
            temp["support_masks"] = support_masks
            temp["query_imgs"] = query_imgs
            temp["query_masks"] = query_masks
            test_dataset.append(temp)
        
        return test_dataset

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        sample = self.data_list[idx]
        return sample