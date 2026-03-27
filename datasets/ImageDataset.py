from .YoutubeFSVOS.YoutubeFSVOS_IMAGE import YTVOSDataset_Image
from .MiniVSPW.nminivspw_dataset_IMAGE import NMiniVSPWEpisodicData_IMAGE
from .COCO.COCO_Dataset import COCO_Dataset
from .LVIS.LVIS_Dataset import LVIS_Dataset
from .ADE20K.ADE20K_Dataset import ADE20K_Dataset
from .PASCAL.PASCAL_Dataset import PASCAL_Dataset

import os

from .YoutubeFSVOS.transform import TestTransform

from torch.utils.data import Dataset

class ImageDataset(Dataset):

    def __init__(self, benchmark, args):
        self.benchmark = benchmark
        self.benchmark_type = benchmark.split('-')[-1] # 'image' or 'video'
        self.use_synset_names = args.use_synset_names
        self.synset_mapping_folder_path = os.path.join(args.synset_mapping_folder_path, f"{self.benchmark}.csv")
        
        self.use_grouping_ade20k = args.use_grouping_ade20k
            
        self.dataset = self.build_dataset(args)
        self.args = args
    
    def build_dataset(self, args):
        if self.benchmark == 'minivspw':
            return NMiniVSPWEpisodicData_IMAGE(
                data_root=args.dataset_path,
                data_list_path=args.data_list_path,
                split_type='test',
                sprtset_as_frames=False,
                n_frames=args.frame_num,
                fold=args.fold-1,
                shot=args.nshot,
                seed_offset=42,
                transform=TestTransform(size=518),
                use_synset_names=self.use_synset_names,
                synset_mapping_csv_path=self.synset_mapping_folder_path,
            )
        elif self.benchmark == 'youtube_fsvos':
            return YTVOSDataset_Image(
                train=False, 
                set_index=args.fold, 
                data_dir=args.dataset_path, 
                frame_num=args.frame_num, 
                seed=args.seed, 
                support_frame=args.nshot, 
                transforms=TestTransform(size=518),
                use_synset_names=self.use_synset_names,
                synset_mapping_csv_path=self.synset_mapping_folder_path,
            )
        elif self.benchmark == 'coco':
            return COCO_Dataset(
                data_dir=args.dataset_path,
                split='val',
                transform=TestTransform(size=518),
                seed=args.seed,
                use_synset_names=self.use_synset_names,
                synset_mapping_csv_path=self.synset_mapping_folder_path,
            )
        elif self.benchmark == 'lvis':
            return LVIS_Dataset(
                data_dir=args.dataset_path,
                split='val',
                transform=TestTransform(size=518),
                seed=args.seed,
                use_synset_names=self.use_synset_names,
                synset_mapping_csv_path=self.synset_mapping_folder_path,
            )
        elif self.benchmark == 'ade20k':
            return ADE20K_Dataset(
                data_dir=args.dataset_path,
                split='val',
                transform=TestTransform(size=518),
                seed=args.seed,
                use_synset_names=self.use_synset_names,
                synset_mapping_csv_path=self.synset_mapping_folder_path,
                use_grouping=self.use_grouping_ade20k,
            )
        elif self.benchmark == 'pascal':
            return PASCAL_Dataset(
                data_dir=args.dataset_path,
                split='val',
                transform=TestTransform(size=518),
                seed=args.seed,
                use_synset_names=self.use_synset_names,
                synset_mapping_csv_path=self.synset_mapping_folder_path,
            )
        else:
            raise ValueError(f'Unknown benchmark: {self.benchmark}')

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        if self.benchmark == 'minivspw':
            query_imgs, query_masks, support_imgs, support_masks, class_id, dir_name, chosen_frames = self.dataset[idx]
            class_id = class_id[0]
            self.support_imgs = support_imgs
            self.support_masks = support_masks
            
        elif self.benchmark == 'youtube_fsvos':
            query_imgs, query_masks, new_support_imgs, new_support_masks, class_id, dir_name, begin_new, chosen_frames = self.dataset[idx]
            if begin_new:
                self.support_imgs = new_support_imgs
                self.support_masks = new_support_masks

        elif self.benchmark in ['coco', 'lvis', 'ade20k', 'pascal']:
            self.support_imgs = None
            self.support_masks = None
            query_imgs, query_masks, class_id, dir_name = self.dataset[idx]
            chosen_frames = [0]

        return {
            'query_imgs': query_imgs,
            'query_masks': query_masks,
            'support_imgs': self.support_imgs,
            'support_masks': self.support_masks,
            'class_id': class_id,
            'class_name': self.dataset.idx_to_classname[class_id],
            'dir_name': dir_name,
            'chosen_frames': chosen_frames,
        }

        


    