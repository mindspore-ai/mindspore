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
"""
Data operations, will be used in train.py and eval.py
"""
import os
import glob
import re
from functools import reduce
import random

from PIL import Image
import numpy as np

import mindspore.dataset as ds


def get_rank_info():
    """
    get rank size and rank id
    """
    from model_utils.moxing_adapter import get_rank_id, get_device_num
    return get_device_num(), get_rank_id()


class FolderImagePair:
    """
    get image pair
    dir_patterns(list): a list of image path patterns. such as ["/LR/*.jpg", "/HR/*.png"...]
                        the file key is matched chars from * and ?
    reader(object/func): a method to read image by path.
    """
    def __init__(self, dir_patterns, reader=None):
        self.dir_patterns = dir_patterns
        self.reader = reader
        self.pair_keys, self.image_pairs = self.scan_pair(self.dir_patterns)

    @staticmethod
    def scan_pair(dir_patterns):
        """
        scan pair
        """
        images = []
        for _dir in dir_patterns:
            imgs = glob.glob(_dir)
            _dir = os.path.basename(_dir)
            pat = _dir.replace("*", "(.*)").replace("?", "(.?)")
            pat = re.compile(pat, re.I | re.M)
            keys = [re.findall(pat, os.path.basename(p))[0] for p in imgs]
            images.append({k: v for k, v in zip(keys, imgs)})
        same_keys = reduce(lambda x, y: set(x) & set(y), images)
        same_keys = sorted(same_keys)
        image_pairs = [[d[k] for d in images] for k in same_keys]
        same_keys = [x if isinstance(x, str) else "_".join(x) for x in same_keys]
        return same_keys, image_pairs

    def get_key(self, idx):
        return self.pair_keys[idx]

    def __getitem__(self, idx):
        if self.reader is None:
            images = [Image.open(p) for p in self.image_pairs[idx]]
            images = [img.convert('RGB') for img in images]
            images = [np.array(img) for img in images]
        else:
            images = [self.reader(p) for p in self.image_pairs[idx]]
        pair_key = self.pair_keys[idx]
        return (pair_key, *images)

    def __len__(self):
        return len(self.pair_keys)


class LrHrImages(FolderImagePair):
    """
    make LrHrImages dataset
    """
    def __init__(self, lr_pattern, hr_pattern, reader=None):
        self.hr_pattern = hr_pattern
        self.lr_pattern = lr_pattern
        self.dir_patterns = []
        if isinstance(self.lr_pattern, str):
            self.is_multi_lr = False
            self.dir_patterns.append(self.lr_pattern)
        elif len(lr_pattern) == 1:
            self.is_multi_lr = False
            self.dir_patterns.append(self.lr_pattern[0])
        else:
            self.is_multi_lr = True
            self.dir_patterns.extend(self.lr_pattern)
        self.dir_patterns.append(self.hr_pattern)
        super(LrHrImages, self).__init__(self.dir_patterns, reader=reader)

    def __getitem__(self, idx):
        _, *images = super(LrHrImages, self).__getitem__(idx)
        return tuple(images)


class _BasePatchCutter:
    """
    cut patch from images
    patch_size(int): patch size, input images should be bigger than patch_size.
    lr_scale(int/list): lr scales for input images. Choice from [1,2,3,4, or their combination]
   """
    def __init__(self, patch_size, lr_scale):
        self.patch_size = patch_size
        self.multi_lr_scale = lr_scale
        if isinstance(lr_scale, int):
            self.multi_lr_scale = [lr_scale]
        else:
            self.multi_lr_scale = [*lr_scale]
        self.max_lr_scale_idx = self.multi_lr_scale.index(max(self.multi_lr_scale))
        self.max_lr_scale = self.multi_lr_scale[self.max_lr_scale_idx]

    def get_tx_ty(self, target_height, target_weight, target_patch_size):
        raise NotImplementedError()

    def __call__(self, *images):
        target_img = images[self.max_lr_scale_idx]

        tp = self.patch_size // self.max_lr_scale
        th, tw, _ = target_img.shape

        tx, ty = self.get_tx_ty(th, tw, tp)

        patch_images = []
        for _, (img, lr_scale) in enumerate(zip(images, self.multi_lr_scale)):
            x = tx * self.max_lr_scale // lr_scale
            y = ty * self.max_lr_scale // lr_scale
            p = tp * self.max_lr_scale // lr_scale
            patch_images.append(img[y:(y + p), x:(x + p), :])
        return tuple(patch_images)


class RandomPatchCutter(_BasePatchCutter):

    def __init__(self, patch_size, lr_scale):
        super(RandomPatchCutter, self).__init__(patch_size=patch_size, lr_scale=lr_scale)

    def get_tx_ty(self, target_height, target_weight, target_patch_size):
        target_x = random.randrange(0, target_weight - target_patch_size + 1)
        target_y = random.randrange(0, target_height - target_patch_size + 1)
        return target_x, target_y


class CentrePatchCutter(_BasePatchCutter):

    def __init__(self, patch_size, lr_scale):
        super(CentrePatchCutter, self).__init__(patch_size=patch_size, lr_scale=lr_scale)

    def get_tx_ty(self, target_height, target_weight, target_patch_size):
        target_x = (target_weight - target_patch_size) // 2
        target_y = (target_height - target_patch_size) // 2
        return target_x, target_y


def hflip(img):
    return img[:, ::-1, :]


def vflip(img):
    return img[::-1, :, :]


def trnsp(img):
    return img.transpose(1, 0, 2)


AUG_LIST = [
    [],
    [trnsp],
    [vflip],
    [vflip, trnsp],
    [hflip],
    [hflip, trnsp],
    [hflip, vflip],
    [hflip, vflip, trnsp],
]


AUG_DICT = {
    "0": [],
    "t": [trnsp],
    "v": [vflip],
    "vt": [vflip, trnsp],
    "h": [hflip],
    "ht": [hflip, trnsp],
    "hv": [hflip, vflip],
    "hvt": [hflip, vflip, trnsp],
}


def flip_and_rotate(*images):
    aug = random.choice(AUG_LIST)
    res = []
    for img in images:
        for a in aug:
            img = a(img)
        res.append(img)
    return tuple(res)


def hwc2chw(*images):
    res = [i.transpose(2, 0, 1) for i in images]
    return tuple(res)


def uint8_to_float32(*images):
    res = [(i.astype(np.float32) if i.dtype == np.uint8 else i) for i in images]
    return tuple(res)


def create_dataset_DIV2K(config, dataset_type="train", num_parallel_workers=10, shuffle=True):
    """
    create a train or eval DIV2K dataset
    Args:
        config(dict):
            dataset_path(string): the path of dataset.
            scale(int/list): lr scale, read data ordered by it, choices=(2,3,4,[2],[3],[4],[2,3],[2,4],[3,4],[2,3,4])
            lr_type(string): lr images type, choices=("bicubic", "unknown"), Default "bicubic"
            batch_size(int): the batch size of dataset. (train prarm), Default 1
            patch_size(int): train data size. (train param), Default -1
            epoch_size(int): times to repeat dataset for dataset_sink_mode, Default None
        dataset_type(string): choices=("train", "valid", "test"), Default "train"
        num_parallel_workers(int): num-workers to read data, Default 10
        shuffle(bool): shuffle dataset. Default: True
    Returns:
        dataset
    """
    dataset_path = config["dataset_path"]
    lr_scale = config["scale"]
    lr_type = config.get("lr_type", "bicubic")
    batch_size = config.get("batch_size", 1)
    patch_size = config.get("patch_size", -1)
    epoch_size = config.get("epoch_size", None)

    # for multi lr scale, such as [2,3,4]
    if isinstance(lr_scale, int):
        multi_lr_scale = [lr_scale]
    else:
        multi_lr_scale = lr_scale

    # get HR_PATH/*.png
    dir_hr = os.path.join(dataset_path, f"DIV2K_{dataset_type}_HR")
    hr_pattern = os.path.join(dir_hr, "*.png")

    # get LR_PATH/X2/*x2.png, LR_PATH/X3/*x3.png, LR_PATH/X4/*x4.png
    column_names = []
    lrs_pattern = []
    for lr_scale in multi_lr_scale:
        dir_lr = os.path.join(dataset_path, f"DIV2K_{dataset_type}_LR_{lr_type}", f"X{lr_scale}")
        lr_pattern = os.path.join(dir_lr, f"*x{lr_scale}.png")
        lrs_pattern.append(lr_pattern)
        column_names.append(f"lrx{lr_scale}")
    column_names.append("hr")  # ["lrx2","lrx3","lrx4",..., "hr"]

    # make dataset
    dataset = LrHrImages(lr_pattern=lrs_pattern, hr_pattern=hr_pattern)

    # make mindspore dataset
    device_num, rank_id = get_rank_info()
    if device_num == 1 or device_num is None:
        generator_dataset = ds.GeneratorDataset(dataset, column_names=column_names,
                                                num_parallel_workers=num_parallel_workers,
                                                shuffle=shuffle and dataset_type == "train")
    elif dataset_type == "train":
        generator_dataset = ds.GeneratorDataset(dataset, column_names=column_names,
                                                num_parallel_workers=num_parallel_workers,
                                                shuffle=shuffle and dataset_type == "train",
                                                num_shards=device_num, shard_id=rank_id)
    else:
        sampler = ds.DistributedSampler(num_shards=device_num, shard_id=rank_id, shuffle=False, offset=0)
        generator_dataset = ds.GeneratorDataset(dataset, column_names=column_names,
                                                num_parallel_workers=num_parallel_workers,
                                                sampler=sampler)

    # define map operations
    if dataset_type == "train":
        transform_img = [
            RandomPatchCutter(patch_size, multi_lr_scale + [1]),
            flip_and_rotate,
            hwc2chw,
            uint8_to_float32,
        ]
    elif patch_size > 0:
        transform_img = [
            CentrePatchCutter(patch_size, multi_lr_scale + [1]),
            hwc2chw,
            uint8_to_float32,
        ]
    else:
        transform_img = [
            hwc2chw,
            uint8_to_float32,
        ]

    # pre-process hr lr
    generator_dataset = generator_dataset.map(input_columns=column_names,
                                              output_columns=column_names,
                                              column_order=column_names,
                                              operations=transform_img)

    # apply batch operations
    generator_dataset = generator_dataset.batch(batch_size, drop_remainder=False)

    # apply repeat operations
    if dataset_type == "train" and epoch_size is not None and epoch_size != 1:
        generator_dataset = generator_dataset.repeat(epoch_size)

    return generator_dataset
