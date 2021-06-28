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
"""loadImg."""
import glob
from collections import OrderedDict
import h5py
import cv2
import numpy as np
from src.utils.transforms import Resize, NormalizeImage, PrepareForNet
from src.config import config


class LoadImagesDepth:
    """LoadImagesDepth."""

    def __init__(self, local_path=None, img_paths=None):
        self.img_files = OrderedDict()
        self.depth_files = OrderedDict()
        self.nF = 0
        for ds, path in img_paths.items():
            self.img_files[ds] = sorted(glob.glob(local_path + path))
            self.depth_files[ds] = [x.replace('Imgs', 'RDs').replace('jpg', 'png') for x in
                                    self.img_files[ds]]
            self.ds = ds
        self.nds = [len(x) for x in self.img_files.values()]
        self.cds = [sum(self.nds[:i]) for i in range(len(self.nds))]
        self.nF = sum(self.nds)
        print(self.nds)
        print(self.cds)
        print(self.nF)
        self.img_input_1 = Resize(config.img_width,
                                  config.img_height,
                                  resize_target=config.resize_target,
                                  keep_aspect_ratio=config.keep_aspect_ratio,
                                  ensure_multiple_of=config.ensure_multiple_of,
                                  resize_method=config.resize_method,
                                  image_interpolation_method=cv2.INTER_CUBIC)
        self.img_input_2 = NormalizeImage(mean=config.nm_img_mean, std=config.nm_img_std)
        self.img_input_3 = PrepareForNet()

    def __getitem__(self, files_index):
        for i, c in enumerate(self.cds):
            if files_index >= c:
                ds = list(self.depth_files.keys())[i]
                start_index = c
        img_path = self.img_files[ds][files_index - start_index]
        depth_path = self.depth_files[ds][files_index - start_index]
        return self.get_data(self.ds, img_path, depth_path)

    def get_data(self, ds, img_path, label_path):
        """get_data."""

        sample = {}
        img = read_image2RGB(img_path)
        if ds == 'Mega':
            depth = read_h5(label_path)
        else:
            depth = read_image2gray(label_path)

        mask = np.ones(depth.shape)
        sample["image"] = img
        sample["mask"] = mask
        sample["depth"] = depth
        sample = self.img_input_1(sample)
        sample = self.img_input_2(sample)
        sample = self.img_input_3(sample)
        return sample["image"], sample["mask"], sample["depth"]

    def __len__(self):
        return self.nF


def read_image2gray(path):
    """Read image and output GRAY image (0-1).
    Args:
        path (str): path to file
    Returns:
        array: GRAY image (0-1)
    """
    imgOri = cv2.imread(path, -1)
    depth = cv2.split(imgOri)[0]
    return depth


def read_image2RGB(path):
    """Read image and output RGB image (0-1).
    Args:
        path (str): path to file
    Returns:
        array: RGB image (0-1)
    """
    img = cv2.imread(path)
    if img is None:
        raise ValueError('File corrupt {}'.format(path))
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0

    return img


def read_h5(path):
    f = h5py.File(path, 'r')
    if f is None:
        raise ValueError('File corrupt {}'.format(path))
    gt = f.get('/depth')
    return np.array(gt)
