#!/usr/bin/env python3

"https://github.com/alinlab/BAR/archive/refs/heads/master.zip"

import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from typing import List, Tuple, Generator, Union
import os
import shutil
import sys
from os import path
from PIL import Image
from sklearn.model_selection import train_test_split
import pandas as pd
import zipfile
import requests
import gdown


class BAR(Dataset):
    DOWNLOAD_URL = "https://drive.google.com/file/d/15QbT46k1TynFTQaeTyb5mMwbzL-5-HHa/view?usp=sharing"
    DATASET_NAME = "bar"

    eval_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    classes_str_to_idx = {
        "climbing": 0,
        "diving": 1,
        "fishing": 2,
        "racing": 3,
        "throwing": 4,
        "pole vaulting": 5,
    }

    def __init__(
            self,
            root="./data",
            env="train",
            bias_amount=95,
            target_name="action",
            confounder_names="background",
            return_index=False,
            transform=None,
            external_bias_labels: bool = False,
            **kwargs
    ) -> None:

        self.root = root
        if transform is None:
            self.transform = BAR.train_transform if env == "train" else BAR.eval_transform
        else:
            self.transform = transform

        self.env = env
        self.bias_amount = bias_amount
        self.num_classes = 6
        self.return_index = return_index
        self.target_name = target_name
        self.confounder_names = confounder_names
        self.n_confounders = 1
        self.num_groups = 2

        self.bias_folder_dict = {
            99: "1pct",
            95: "5pct"
        }

        if not os.path.isdir(os.path.join(self.root, "bar")):
            self.__download_dataset()
        else:
            self.root = os.path.join(self.root, "bar")

        if self.env == "train":
            self.filename_array, self.y_array, self.confounder_array = self.load_train_samples()
            if external_bias_labels:
                print("Loading external bias labels for the training set...")
                self.old_garray = self.confounder_array.copy()
                self.confounder_array = pd.read_csv(os.path.join("outputs", "bar_metadata_aug.csv"), header="infer")[
                    "ddb"].to_numpy()
                assert len(self.old_garray) == len(self.confounder_array)
                self.old_garray = None

        if self.env == "val":
            self.filename_array, self.y_array, self.confounder_array = self.load_val_samples()

        if self.env == "test":
            self.filename_array, self.y_array, self.confounder_array = self.load_test_samples()

        self.group_array = (self.y_array * self.num_groups + self.confounder_array).long()

        # LEGACY WHEN WITHOUT BIAS LABELS
        # if self.env == "train":
        #     self.num_classes = 6  # Six classes
        #     self.n_confounders = 1  # Still one confounder
        #     self.num_groups = self.num_classes * 2  # 2 groups per class => 6 * 2 = 12 groups
        #     self.group_array = (self.y_array * 2 + self.bias_labels).astype('int')
        # else:
        #     self.num_classes = 6  # Six classes
        #     self.n_confounders = 0
        #     self.num_groups = self.num_classes
        #     self.group_array = (2 * self.y_array).astype("int")

    def __len__(self):
        return len(self.y_array)

    def get_group_array(self):
        return self.group_array

    def get_label_array(self):
        return self.y_array

    def __getitem__(self, index):
        file_path = self.filename_array[index]
        class_label = self.y_array[index]
        bias_label = self.confounder_array[index]

        image = self.transform(Image.open(file_path).convert("RGB"))

        return image, (class_label, bias_label), index

    def __download_dataset(self) -> None:
        os.makedirs(self.root, exist_ok=True)
        output_path = os.path.join(self.root, "bar.zip")
        print(f"=> Downloading {BAR.DATASET_NAME} from {BAR.DOWNLOAD_URL}")

        try:
            gdown.download(id="15QbT46k1TynFTQaeTyb5mMwbzL-5-HHa", output=output_path)
        except:
            raise RuntimeError(
                "Unable to complete dataset download, check for your internet connection or try changing download link.")

        print(f"=> Extracting bar.zip to directory {self.root}")
        try:
            with zipfile.ZipFile(output_path, mode="r") as unzipper:
                unzipper.extractall(self.root)
        except:
            raise RuntimeError(f"Unable to extract {output_path}, an error occured.")

        self.root = os.path.join(self.root, "bar")
        os.remove(output_path)

    def load_train_samples(self):
        samples_path: List[str] = []
        class_labels: List[int] = []
        bias_labels: List[int] = []

        bias_folder = self.bias_folder_dict[self.bias_amount]

        for c, class_folder in enumerate(sorted(os.listdir(os.path.join(self.root, "train", "align")))):
            for filename in sorted(os.listdir(os.path.join(self.root, "train", "align", class_folder))):
                samples_path.append(os.path.join(self.root, "train", "align", class_folder, filename))
                class_labels.append(c)
                bias_labels.append(0)

        for c, class_folder in enumerate(sorted(os.listdir(os.path.join(self.root, "train", "conflict", bias_folder)))):
            for filename in sorted(os.listdir(os.path.join(self.root, "train", "conflict", bias_folder, class_folder))):
                samples_path.append(os.path.join(self.root, "train", "conflict", bias_folder, class_folder, filename))
                class_labels.append(int(c))
                bias_labels.append(1)

        return (
            np.array(samples_path),
            torch.as_tensor(class_labels),
            torch.as_tensor(bias_labels)
        )

    def load_val_samples(self):
        samples_path: List[str] = []
        class_labels: List[int] = []
        bias_labels: List[int] = []

        for c, class_folder in enumerate(sorted(os.listdir(os.path.join(self.root, "valid")))):
            for filename in sorted(os.listdir(os.path.join(self.root, "valid", class_folder))):
                samples_path.append(os.path.join(self.root, "valid", class_folder, filename))
                class_labels.append(int(c))
                bias_labels.append(0)

        return (
            np.array(samples_path),
            torch.as_tensor(class_labels),
            torch.as_tensor(bias_labels)
        )

    def load_test_samples(self):
        samples_path: List[str] = []
        class_labels: List[int] = []
        bias_labels: List[int] = []

        for c, class_folder in enumerate(sorted(os.listdir(os.path.join(self.root, "test")))):
            for filename in sorted(os.listdir(os.path.join(self.root, "test", class_folder))):
                samples_path.append(os.path.join(self.root, "test", class_folder, filename))
                class_labels.append(int(c))
                bias_labels.append(1)

        return (
            np.array(samples_path),
            torch.as_tensor(class_labels),
            torch.as_tensor(bias_labels)
        )

    # def assign_bias_label(self, filename: str) -> int:
    #     no_extension = filename.split(".")[0]
    #     _, _, z = no_extension.split("_")
    #     return int(z)

    # def assign_class_label(self, filename: str) -> int:
    #     no_extension = filename.split(".")[0]
    #     _, y, _ = no_extension.split("_")
    #     return int(y)

    def perclass_populations(self, return_labels: bool = False) -> Union[
        Tuple[float, float], Tuple[Tuple[float, float], torch.Tensor]]:
        labels: torch.Tensor = torch.zeros(len(self))
        for i in range(len(self)):
            labels[i] = self[i][1][0]

        _, pop_counts = labels.unique(return_counts=True)

        if return_labels:
            return pop_counts.long(), labels.long()

        return pop_counts

    def get_sampling_weights(self, classes_only: bool):
        if classes_only:
            group_counts: torch.Tensor = (
                (torch.arange(self.num_classes).unsqueeze(1) == self.y_array)
                .sum(1)
                .float()
            )
        else:
            group_counts: torch.Tensor = (
                (torch.arange(self.num_groups * self.num_classes).unsqueeze(1) == self.group_array)
                .sum(1)
                .float()
            )

        group_weights = len(self) / group_counts
        weights = group_weights[self.y_array if classes_only else self.group_array]
        return weights

    def get_group_counts(self):
        group_counts = (
            (torch.arange(self.num_groups * self.num_classes).unsqueeze(1) == self.group_array)
            .sum(1).float()
        )
        return group_counts

    def get_labels(self) -> torch.Tensor:
        return torch.as_tensor(list([self[i][1][0] for i in range(len(self))]))

    def get_group_labels(self) -> torch.Tensor:
        return torch.as_tensor(list([self[i][1][1] for i in range(len(self))]))

    def get_group_array(self):
        return self.group_array

    def get_label_array(self):
        return self.y_array

    def __repr__(self) -> str:
        return f"BAR(env={self.env}, bias_amount={self.bias_amount}, n_classes={self.num_classes})"


if __name__ == "__main__":
    a = BAR(env="train", bias_amount=95)
    b = BAR(env="train", bias_amount=99)
    c = BAR(env="val")
    d = BAR(env="test")

    print(a)
    print(b)
    print(c)
    print(d)

    print(len(a))
    print(len(b))
    print(len(c))
    print(len(d))

    print(a.perclass_populations())
    print(b.perclass_populations())
    print(c.perclass_populations())
    print(d.perclass_populations())

    print(a.get_group_labels().unique(return_counts=True))
    print(b.get_group_labels().unique(return_counts=True))
    print(c.get_group_labels().unique(return_counts=True))
    print(d.get_group_labels().unique(return_counts=True))