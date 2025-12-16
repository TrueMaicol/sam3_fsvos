import torch
from torch.utils.data import Dataset
import os
import argparse
import numpy as np
from PIL import Image
from itertools import cycle, islice
from typing import List
import random

from . import transform

class VSPWAuxiliaryData(Dataset):
    def __init__(self,
                 transform: transform.Compose,
                 args: argparse.Namespace):
        # Sequences with diff definition for Pascal Class, E.g. Bottle vs Bottle_or_Cup
        # Or Objects Completely Out of View
        self.ignore_seqs = ["1146_lps8_r-6J00", "118_fgmRMeHommU", "1278_C4zCNFn1xIs",
                            "1275_ARcg-EyKWrA", "1302_OCLlE02BHGk", "131_eUiWFntut00",
                            "1339_J73l0QCY8YM", "1353_CszoyQ3HMcM", "1476_75AL-XN84cI",
                            "1887_tUt0N6eGtGY", "198_b8euyKNT2wY", "1026_kJ_8F7YIEg4",
                            "1265_PBn1W-aOFUA", "1035_248bbw7mpdw", "1107_YXyd44eY_VY",
                            "1154_B4zEa_7Ejtk", "1282_cSmDnZFwqIM", "1477_GkuOGCQiUlk",
                            "2051_e0EI-QqHPIA"]
        self.transform = transform
        self.img_dir = args.aux_data_root

        self.every_frame = args.aux_every_frame
        self.temporal_window = args.aux_temporal_window
        self.n_frames = args.aux_n_frames
        self.imgs_list_by_vid = self._create_list(args)
        self.indices = list(range(len(self.imgs_list_by_vid)))
        random.shuffle(self.indices)

    def reset_indices(self):
        random.shuffle(self.indices)

    def _create_list(self, args):
        imgs_list = {} # per vid_id
        with open('lists/vspw/all.txt') as f:
            for line in f:
                seq = line.strip().split('/')[-3]
                if seq not in self.ignore_seqs:
                    if seq not in imgs_list:
                        imgs_list[seq] = []

                    imgs_list[seq].append(line.strip().replace('mask', 'origin').replace('png', 'jpg'))

        return imgs_list

    def load_frames(self, frames_paths):
        frames = []
        labels = []
        for frame_path in sorted(frames_paths):
            frames.append(np.array(Image.open(os.path.join(self.img_dir, frame_path))) )
            # Loader does not provide labels, only used for self supervised loss
            labels.append(np.zeros(frames[-1].shape[:2], dtype=np.int32))
        return frames, labels

    def __getitem__(self, index):
        index = self.indices[index%len(self.indices)]
        vid_id = list(self.imgs_list_by_vid.keys())[index]
        vid_frames = self.imgs_list_by_vid[vid_id]
        frames, labels = self.load_frames(vid_frames)
        frames = frames[::self.every_frame]
        labels = labels[::self.every_frame]

        # Temporal Sliding Window
        while len(frames) < self.temporal_window:
            frames.append(random.choice(frames))
            labels.append(labels[-1]) # Its all zeros

        # Apply different random augmentations to every frame
        transformed_frames = []
        transformed_labels = []
        for i in range(len(frames)):
            frame, label = self.transform(frames[i], labels[i])
            transformed_frames.append(frame)
            transformed_labels.append(label)

        frames = torch.stack(transformed_frames)
        labels = torch.stack(transformed_labels)

        # Temporal Sliding Window
        frames = frames.unfold(0,self.temporal_window, 1).permute(0, 4, 1, 2, 3)
        labels = labels.unsqueeze(1) # Channel Dim
        labels = labels.unfold(0,self.temporal_window, 1).permute(0, 4, 1, 2, 3)

        # Randomly select N frames
        frame_indices = torch.randperm(frames.shape[0])
        frames = [f for f in frames[frame_indices]]
        labels = [f for f in labels[frame_indices]]

        # Fill # frames
        frames = [f for f in islice(cycle(frames), self.n_frames)]
        labels = [f for f in islice(cycle(labels), self.n_frames)]

        frames = torch.stack(frames)
        labels = torch.stack(labels)
        assert frames.shape[0] == self.n_frames, "Sampled less frames than %d"%self.n_frames
        assert labels.shape[0] == self.n_frames, "Sampled less frames than %d"%self.n_frames

        return frames, labels, 0

    def __len__(self):
        return len(self.imgs_list_by_vid)

