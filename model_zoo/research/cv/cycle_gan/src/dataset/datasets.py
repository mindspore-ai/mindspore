# Copyright 2020 Huawei Technologies Co., Ltd
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
"""Cycle GAN datasets."""
import os
import random
import numpy as np
from PIL import Image

random.seed(1)
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

    Args:
        dataroot (str): Images root directory.
        phase (str): Train or test. It requires two directories in dataroot, like trainA and trainB to
            host training images from domain A '{dataroot}/trainA' and from domain B '{dataroot}/trainB' respectively.
        max_dataset_size (int): Maximum number of return image paths.

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
        return self.size
