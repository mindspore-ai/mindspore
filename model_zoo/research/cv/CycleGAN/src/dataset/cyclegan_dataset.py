# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

"""Cycle GAN dataset."""

import os
import random
import multiprocessing
import numpy as np
from PIL import Image
import mindspore.dataset as de
import mindspore.dataset.vision.c_transforms as C
from .distributed_sampler import DistributedSampler

IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.tif', '.tiff']

def is_image_file(filename):
    """Judge whether it is a picture."""
    return any(filename.lower().endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(dir_path, max_dataset_size=float("inf")):
    """Return image list in dir."""
    images = []
    assert os.path.isdir(dir_path), '%s is not a valid directory' % dir_path

    for root, _, fnames in sorted(os.walk(dir_path)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)
    return images[:min(max_dataset_size, len(images))]


class UnalignedDataset:
    """
    This dataset class can load unaligned/unpaired datasets.
    It requires two directories to host training images from domain A '/path/to/data/trainA'
    and from domain B '/path/to/data/trainB' respectively.
    You can train the model with the dataset flag '--dataroot /path/to/data'.
    Similarly, you need to prepare two directories:
    '/path/to/data/testA' and '/path/to/data/testB' during test time.
    Returns:
        Two domain image path list.
    """
    def __init__(self, dataroot, phase, max_dataset_size=float("inf"), use_random=True):
        self.dir_A = os.path.join(dataroot, phase + 'A')
        self.dir_B = os.path.join(dataroot, phase + 'B')

        self.A_paths = sorted(make_dataset(self.dir_A, max_dataset_size))   # load images from '/path/to/data/trainA'
        self.B_paths = sorted(make_dataset(self.dir_B, max_dataset_size))    # load images from '/path/to/data/trainB'
        self.A_size = len(self.A_paths)  # get the size of dataset A
        self.B_size = len(self.B_paths)  # get the size of dataset B
        self.use_random = use_random

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index (int)      -- a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor)       -- an image in the input domain
            B (tensor)       -- its corresponding image in the target domain
            A_paths (str)    -- image paths
            B_paths (str)    -- image paths
        """
        index_B = index % self.B_size
        if index % max(self.A_size, self.B_size) == 0 and self.use_random:
            random.shuffle(self.A_paths)
            index_B = random.randint(0, self.B_size - 1)
        A_path = self.A_paths[index % self.A_size]
        B_path = self.B_paths[index_B]
        A_img = np.array(Image.open(A_path).convert('RGB'))
        B_img = np.array(Image.open(B_path).convert('RGB'))

        return A_img, B_img

    def __len__(self):
        """Return the total number of images in the dataset.
        """
        return max(self.A_size, self.B_size)


class ImageFolderDataset:
    """
    This dataset class can load images from image folder.
    Args:
        dataroot (str): Images root directory.
        max_dataset_size (int): Maximum number of return image paths.
    Returns:
        Image path list.
    """
    def __init__(self, dataroot, max_dataset_size=float("inf")):
        self.dataroot = dataroot
        self.paths = sorted(make_dataset(dataroot, max_dataset_size))
        self.size = len(self.paths)

    def __getitem__(self, index):
        img_path = self.paths[index % self.size]
        img = np.array(Image.open(img_path).convert('RGB'))

        return img, os.path.split(img_path)[1]

    def __len__(self):
        """Return the total number of images in the dataset.
        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return self.size


def create_dataset(args):
    """
    Create dataset
    This dataset class can load images for train or test.
    Args:
        dataroot (str): Images root directory.
    Returns:
        RGB Image list.
    """
    dataroot = args.dataroot
    phase = args.phase
    batch_size = args.batch_size
    device_num = args.device_num
    rank = args.rank
    shuffle = args.use_random
    max_dataset_size = args.max_dataset_size
    cores = multiprocessing.cpu_count()
    num_parallel_workers = min(8, int(cores / device_num))
    image_size = args.image_size
    mean = [0.5 * 255] * 3
    std = [0.5 * 255] * 3
    if phase == "train":
        dataset = UnalignedDataset(dataroot, phase, max_dataset_size=max_dataset_size, use_random=args.use_random)
        distributed_sampler = DistributedSampler(len(dataset), device_num, rank, shuffle=shuffle)
        ds = de.GeneratorDataset(dataset, column_names=["image_A", "image_B"],
                                 sampler=distributed_sampler, num_parallel_workers=num_parallel_workers)
        if args.use_random:
            trans = [
                C.RandomResizedCrop(image_size, scale=(0.5, 1.0), ratio=(0.75, 1.333)),
                C.RandomHorizontalFlip(prob=0.5),
                C.Normalize(mean=mean, std=std),
                C.HWC2CHW()
            ]
        else:
            trans = [
                C.Resize((image_size, image_size)),
                C.Normalize(mean=mean, std=std),
                C.HWC2CHW()
            ]
        ds = ds.map(operations=trans, input_columns=["image_A"], num_parallel_workers=num_parallel_workers)
        ds = ds.map(operations=trans, input_columns=["image_B"], num_parallel_workers=num_parallel_workers)
        ds = ds.batch(batch_size, drop_remainder=True)
        ds = ds.repeat(1)
    else:
        datadir = os.path.join(dataroot, args.data_dir)
        dataset = ImageFolderDataset(datadir, max_dataset_size=max_dataset_size)
        ds = de.GeneratorDataset(dataset, column_names=["image", "image_name"],
                                 num_parallel_workers=num_parallel_workers)
        trans = [
            C.Resize((image_size, image_size)),
            C.Normalize(mean=mean, std=std),
            C.HWC2CHW()
        ]
        ds = ds.map(operations=trans, input_columns=["image"], num_parallel_workers=num_parallel_workers)
        ds = ds.batch(1, drop_remainder=True)
        ds = ds.repeat(1)
    args.dataset_size = len(dataset)
    return ds
    