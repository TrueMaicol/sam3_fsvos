from pycocotools.ytvos import YTVOS
from torch.utils.data import Dataset
import os
import numpy as np
import random
from PIL import Image
from torchvision import transforms
import json
import pandas as pd

class YTVOSDataset_Image(Dataset):
    def __init__(self, data_dir=None, train=True, valid=False,
                 set_index=1, finetune_idx=None,
                 support_frame=5, query_frame=1, sample_per_class=10,
                 transforms=None, another_transform=None, frame_num=None, seed=None,
                 use_synset_names=False, synset_mapping_csv_path=None):
        self.train = train
        self.valid = valid
        self.set_index = set_index
        self.support_frame = support_frame
        self.query_frame = query_frame
        self.sample_per_class = sample_per_class
        self.transforms = transforms
        self.another_transform = another_transform
        if frame_num <= 0:
            raise Exception("frame_num must be greater than 0")
        self.frame_num = frame_num
        self.benchmark = "youtube-fsvos"
        self.seed = seed
        self.shortenend_videos = 0
        self.use_synset_names = use_synset_names

        # Load class names from categories.json
        categories_path = os.path.join(os.path.dirname(__file__), 'categories.json')
        with open(categories_path, 'r') as f:
            categories = json.load(f)
        
        if data_dir is None:
            data_dir = "./datasets/Youtube-FSVOS/train"
        
        if synset_mapping_csv_path is None:
            self.synset_mapping_csv_path = os.path.join(data_dir, "synset_mapping.csv")
        else:
            self.synset_mapping_csv_path = synset_mapping_csv_path


        self.img_dir = os.path.join(data_dir, 'JPEGImages')
        self.ann_file = os.path.join(data_dir, 'instances.json')

        self.load_annotations()

        self.train_list = [n + 1 for n in range(40) if n % 4 != (set_index - 1)]
        self.valid_list = [n + 1 for n in range(40) if n % 4 == (set_index - 1)]

        if train and not valid:
            self.class_ids = self.train_list
        else:
            self.class_ids = self.valid_list
        if finetune_idx is not None:
            self.class_ids = [self.class_ids[finetune_idx]]

        self.class_idx_to_all_lemmas = {}
        if self.use_synset_names:
            synset_mapping = pd.read_csv(self.synset_mapping_csv_path, sep="|")
            self.idx_to_classname = {}
            for cat in categories:
                idx = cat['id']
                # don't use all the classes, just use the ones inside the validation list
                if idx not in self.class_ids:
                    continue
                match = synset_mapping[synset_mapping['idx'] == idx]
                selected_lemma = match['selected_lemma']
                
                if len(selected_lemma) > 0 and pd.notna(selected_lemma.values[0]):
                    self.idx_to_classname[idx] = str(selected_lemma.values[0]).split(",")[0].replace("_", " ")
                else:
                    print("No match found for {}".format(idx))
                    # Fallback to categories.json name

                lemmas_str = match['lemmas'].values[0] if 'lemmas' in synset_mapping.columns else None
                if lemmas_str is not None and pd.notna(lemmas_str):
                    self.class_idx_to_all_lemmas[idx] = [l.replace("_", " ") for l in str(lemmas_str).split(",")]
        else:
            self.idx_to_classname = {cat['id']: cat['name'] for cat in categories}
            
        assert len(self.class_ids) == len(self.idx_to_classname.keys()), "The number of classes in the dataset does not match the number of classes in the label mapping."

        self.video_ids = []
        for class_id in self.class_ids:
            tmp_list = self.ytvos.getVidIds(catIds=class_id)
            tmp_list.sort()
            self.video_ids.append(tmp_list)  # list[list[video_id]] # video_ids is a list of length (num_classes) and each entry has a list of the video ids for that specific class
        if not self.train:
            self.test_video_classes = []
            for i in range(len(self.class_ids)): # for each class
                for j in range(len(self.video_ids[i]) - support_frame):  # remove the support set and append the class idx for each video
                    self.test_video_classes.append(i)

        if self.train:
            self.length = len(self.class_ids) * sample_per_class
        else:
            self.length = len(self.test_video_classes)  # test

    def load_annotations(self):
        self.ytvos = YTVOS(self.ann_file)
        self.vid_ids = self.ytvos.getVidIds()  # list[2238] begin : 1
        self.vid_infos = self.ytvos.vids  # vids
        for vid, vid_info in self.vid_infos.items():  # for each vid
            vid_name = vid_info['file_names'][0].split('/')[0]  # '0043f083b5'
            vid_info['dir'] = vid_name
            frame_len = vid_info['length']  # int
            frame_object, frame_class = [], []
            for i in range(frame_len): frame_object.append([])
            for i in range(frame_len): frame_class.append([])
            category_set = set()
            annos = self.ytvos.vidToAnns[vid]  # list[]
            for anno in annos:  # instance_level anns
                assert len(anno['segmentations']) == frame_len, (
                vid_name, len(anno['segmentations']), vid_info['length'])
                for frame_idx in range(frame_len):
                    anno_segmentation = anno['segmentations'][frame_idx]
                    if anno_segmentation is not None:
                        frame_object[frame_idx].append(anno['id'])  # add instance to vid_frame
                        frame_class[frame_idx].append(anno['category_id'])  # add instance class to vid_frame
                        category_set = category_set.union({anno['category_id']})
            vid_info['objects'] = frame_object
            vid_info['classes'] = frame_class
            class_frame_id = dict()
            for class_id in category_set:  # frames index for each class
                class_frame_id[class_id] = [i for i in range(frame_len) if class_id in frame_class[i]]
            vid_info['class_frames'] = class_frame_id

    def get_GT_byclass(self, vid, class_id, frame_num=None, test=False):
        if frame_num is None:
            raise Exception('frame_num must be specified')
        vid_info = self.vid_infos[vid]
        frame_list = vid_info['class_frames'][class_id]
        
        full_masks = {}
        for frame_id in frame_list:
            object_ids = vid_info['objects'][frame_id]
            mask = None
            for object_id in object_ids:
                ann = self.ytvos.loadAnns(object_id)[0]
                if ann['category_id'] not in self.class_ids:
                    continue
                track_id = 1
                if ann['category_id'] != class_id:
                    track_id = 0
                temp_mask = self.ytvos.annToMask(ann, frame_id)
                if mask is None:
                    mask = temp_mask * track_id
                else:
                    mask += temp_mask * track_id

            assert mask is not None
            mask[mask > 0] = 1
            if np.sum(mask) > 0:
                full_masks[frame_id] = mask
        
        
        available_frames = list(full_masks.keys())
        print(available_frames)
        if frame_num >= len(available_frames):
            # raise Exception(f'Asking for {frame_num} frames but the video has {len(available_frames)} non-empty frames')
            print(f"The video {vid_info['dir']} has {len(available_frames)} frame with class {class_id} present. Requested {frame_num} frames. Taking the entire video...")
            chosen_frames = available_frames
            self.shortenend_videos += 1
        else:
            chosen_frames = random.sample(available_frames, frame_num)
        
        print(f"Selected frames idx: {chosen_frames} from video {vid_info['dir']}")

        frames = [np.array(Image.open(os.path.join(self.img_dir, vid_info['file_names'][frame_idx]))) for frame_idx in chosen_frames]
        masks=[full_masks[frame_idx] for frame_idx in chosen_frames]

        return frames, masks, chosen_frames

    def __gettrainitem__(self, idx):
        list_id = idx // self.sample_per_class
        vid_set = self.video_ids[list_id]

        query_vid = random.sample(vid_set, 1)
        support_vid = random.sample(vid_set, self.support_frame)

        query_frames, query_masks = self.get_GT_byclass(query_vid[0], self.class_ids[list_id], self.query_frame)

        support_frames, support_masks = [], []
        for i in range(self.support_frame):
            one_frame, one_mask = self.get_GT_byclass(support_vid[i], self.class_ids[list_id], 1)
            support_frames += one_frame
            support_masks += one_mask

        if self.transforms is not None:
            query_frames, query_masks = self.transforms(query_frames, query_masks)
            support_frames, support_masks = self.transforms(support_frames, support_masks)
        return query_frames, query_masks, support_frames, support_masks, self.class_ids[list_id]

    def __gettestitem__(self, idx):
        
        begin_new = False
        if idx == 0:
            begin_new = True
        else:
            if self.test_video_classes[idx] != self.test_video_classes[idx - 1]:
                begin_new = True
        list_id = self.test_video_classes[idx] # get the class_idx specific for the requested video
        vid_set = self.video_ids[list_id] # get the list of videos for the chosen class
        # print(f"Testing class {self.class_ids[list_id]} with {len(vid_set)} videos")
        support_frames, support_masks = [], []
        if begin_new:
            if self.seed is not None:
                random.seed(self.seed + self.class_ids[list_id])  # Deterministic per class
                print(f"Setting random seed to {self.seed + self.class_ids[list_id]} for class {self.class_ids[list_id]}")
            support_vid = random.sample(vid_set, self.support_frame)
            query_vids = []
            for id in vid_set:
                if not id in support_vid:
                    query_vids.append(id)
            self.query_ids = query_vids
            self.query_idx = -1
            for i in range(self.support_frame):
                one_frame, one_mask, support_chosen_frames = self.get_GT_byclass(support_vid[i], self.class_ids[list_id], 1)
                print("Support video ID: ", support_vid[i])
                support_frames += one_frame
                support_masks += one_mask

        self.query_idx += 1
        query_vid = self.query_ids[self.query_idx]
        query_frames, query_masks, chosen_frames = self.get_GT_byclass(query_vid, self.class_ids[list_id], frame_num=self.frame_num, test=True)

        if self.transforms is not None:
            query_frames, query_masks = self.transforms(query_frames, query_masks)
            if begin_new and support_frames:
                if self.another_transform is not None:
                    support_frames, support_masks = self.another_transform(support_frames, support_masks)
                else:
                    support_frames, support_masks = self.transforms(support_frames, support_masks)
        vid_info = self.vid_infos[query_vid]
        vid_name = vid_info['dir']
        return query_frames, query_masks, support_frames, support_masks, self.class_ids[list_id], vid_name, begin_new, chosen_frames

    def __getitem__(self, idx):
        if self.train:
            return self.__gettrainitem__(idx)
        else:
            return self.__gettestitem__(idx)

    def __len__(self):
        return self.length

    def get_class_ids(self):
        return self.class_ids


class YouTubeTransform:
    """Custom transform for YouTube-VOS that handles both frames and masks."""
    def __init__(self, img_size: int):
        self.img_size = img_size
        self.resize = transforms.Resize(size=(img_size, img_size))
    
    def __call__(self, frames, masks):
        """
        Apply transforms to frames and masks.
        
        Args:
            frames: List of numpy arrays or PIL Images
            masks: List of numpy arrays or PIL Images (masks)
        
        Returns:
            Transformed frames as list of PIL Images and masks as list of tensors
        """
        transformed_frames = []
        transformed_masks = []
        
        for frame in frames:
            # Convert numpy array to PIL Image if needed
            if isinstance(frame, np.ndarray):
                # Handle both RGB (H, W, 3) and grayscale (H, W) arrays
                if frame.ndim == 2:
                    pil_frame = Image.fromarray(frame, mode='L').convert('RGB')
                else:
                    pil_frame = Image.fromarray(frame, mode='RGB')
            else:
                pil_frame = frame
            
            # Resize and keep as PIL Image
            resized_frame = pil_frame.resize((self.img_size, self.img_size), Image.BILINEAR)
            img_tensor = transforms.ToTensor()(resized_frame)
            transformed_frames.append(img_tensor)

        for mask in masks:
            # Convert numpy array to PIL Image if needed
            if isinstance(mask, np.ndarray):
                # Squeeze if necessary and ensure it's 2D
                if mask.ndim == 3:
                    mask = np.squeeze(mask)
                pil_mask = Image.fromarray(mask.astype(np.uint8), mode='L')
            else:
                pil_mask = mask
            
            # Resize with NEAREST interpolation for masks
            resized_mask = pil_mask.resize((self.img_size, self.img_size), Image.NEAREST)
            
            # Convert to tensor and make binary
            mask_tensor = transforms.ToTensor()(resized_mask)
            mask_tensor = (mask_tensor > 0).float().squeeze(0)
            
            transformed_masks.append(mask_tensor)
        
        return transformed_frames, transformed_masks

if __name__ == "__main__":
    ytvos = YTVOSDataset(train=True, query_frame=5, support_frame=5)
    video_query_img, video_query_mask, new_support_img, new_support_mask, idx, *_ = ytvos[0]