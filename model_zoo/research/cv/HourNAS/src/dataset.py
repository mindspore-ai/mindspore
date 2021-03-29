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
"""Data operations, will be used in train.py and eval.py"""
import math
import os

import numpy as np
import mindspore.dataset.vision.py_transforms as py_vision
import mindspore.dataset.transforms.py_transforms as py_transforms
import mindspore.dataset.transforms.c_transforms as c_transforms
import mindspore.common.dtype as mstype
import mindspore.dataset as ds
from mindspore.communication.management import get_rank, get_group_size
from mindspore.dataset.vision import Inter
import mindspore.dataset.vision.c_transforms as vision

# values that should remain constant
DEFAULT_CROP_PCT = 0.875
IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

# data preprocess configs
SCALE = (0.08, 1.0)
RATIO = (3./4., 4./3.)

ds.config.set_seed(1)


def split_imgs_and_labels(imgs, labels, batchInfo):
    """split data into labels and images"""
    ret_imgs = []
    ret_labels = []

    for i, image in enumerate(imgs):
        ret_imgs.append(image)
        ret_labels.append(labels[i])
    return np.array(ret_imgs), np.array(ret_labels)


def create_dataset(batch_size, train_data_url='', workers=8, distributed=False,
                   input_size=224, color_jitter=0.4):
    """Create ImageNet training dataset"""
    if not os.path.exists(train_data_url):
        raise ValueError('Path not exists')
    decode_op = py_vision.Decode()
    type_cast_op = c_transforms.TypeCast(mstype.int32)

    random_resize_crop_bicubic = py_vision.RandomResizedCrop(size=(input_size, input_size),
                                                             scale=SCALE, ratio=RATIO,
                                                             interpolation=Inter.BICUBIC)
    random_horizontal_flip_op = py_vision.RandomHorizontalFlip(0.5)
    adjust_range = (max(0, 1 - color_jitter), 1 + color_jitter)
    random_color_jitter_op = py_vision.RandomColorAdjust(brightness=adjust_range,
                                                         contrast=adjust_range,
                                                         saturation=adjust_range)
    to_tensor = py_vision.ToTensor()
    normalize_op = py_vision.Normalize(
        IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)

    # assemble all the transforms
    image_ops = py_transforms.Compose([decode_op, random_resize_crop_bicubic,
                                       random_horizontal_flip_op, random_color_jitter_op, to_tensor, normalize_op])

    rank_id = get_rank() if distributed else 0
    rank_size = get_group_size() if distributed else 1

    dataset_train = ds.ImageFolderDataset(train_data_url,
                                          num_parallel_workers=workers,
                                          shuffle=True,
                                          num_shards=rank_size,
                                          shard_id=rank_id)

    dataset_train = dataset_train.map(input_columns=["image"],
                                      operations=image_ops,
                                      num_parallel_workers=workers)

    dataset_train = dataset_train.map(input_columns=["label"],
                                      operations=type_cast_op,
                                      num_parallel_workers=workers)

    # batch dealing
    ds_train = dataset_train.batch(batch_size,
                                   per_batch_map=split_imgs_and_labels,
                                   input_columns=["image", "label"],
                                   num_parallel_workers=2,
                                   drop_remainder=True)

    ds_train = ds_train.repeat(1)
    return ds_train


def create_dataset_val(batch_size=128, val_data_url='', workers=8, distributed=False,
                       input_size=224):
    """Create ImageNet validation dataset"""
    if not os.path.exists(val_data_url):
        raise ValueError('Path not exists')
    rank_id = get_rank() if distributed else 0
    rank_size = get_group_size() if distributed else 1
    dataset = ds.ImageFolderDataset(val_data_url, num_parallel_workers=workers,
                                    num_shards=rank_size, shard_id=rank_id)
    scale_size = None

    if isinstance(input_size, tuple):
        assert len(input_size) == 2
        if input_size[-1] == input_size[-2]:
            scale_size = int(math.floor(input_size[0] / DEFAULT_CROP_PCT))
        else:
            scale_size = tuple([int(x / DEFAULT_CROP_PCT) for x in input_size])
    else:
        scale_size = int(math.floor(input_size / DEFAULT_CROP_PCT))

    type_cast_op = c_transforms.TypeCast(mstype.int32)
    decode_op = py_vision.Decode()
    resize_op = py_vision.Resize(size=scale_size, interpolation=Inter.BICUBIC)
    center_crop = py_vision.CenterCrop(size=input_size)
    to_tensor = py_vision.ToTensor()
    normalize_op = py_vision.Normalize(
        IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)

    image_ops = py_transforms.Compose([decode_op, resize_op, center_crop,
                                       to_tensor, normalize_op])

    dataset = dataset.map(input_columns=["label"], operations=type_cast_op,
                          num_parallel_workers=workers)
    dataset = dataset.map(input_columns=["image"], operations=image_ops,
                          num_parallel_workers=workers)
    dataset = dataset.batch(batch_size, per_batch_map=split_imgs_and_labels,
                            input_columns=["image", "label"],
                            num_parallel_workers=2,
                            drop_remainder=True)
    dataset = dataset.repeat(1)
    return dataset

def _get_rank_info():
    """
    get rank size and rank id
    """
    rank_size = int(os.environ.get("RANK_SIZE", 1))

    if rank_size > 1:
        rank_size = get_group_size()
        rank_id = get_rank()
    else:
        rank_size = rank_id = None

    return rank_size, rank_id

def create_dataset_cifar10(data_home, repeat_num=1, training=True, cifar_cfg=None):
    """Data operations."""
    data_dir = os.path.join(data_home, "cifar-10-batches-bin")
    if not training:
        data_dir = os.path.join(data_home, "cifar-10-verify-bin")

    rank_size, rank_id = _get_rank_info()
    if training:
        data_set = ds.Cifar10Dataset(data_dir, num_shards=rank_size, shard_id=rank_id, shuffle=True)
    else:
        data_set = ds.Cifar10Dataset(data_dir, num_shards=rank_size, shard_id=rank_id, shuffle=False)

    resize_height = cifar_cfg.image_height
    resize_width = cifar_cfg.image_width

    # define map operations
    random_crop_op = vision.RandomCrop((32, 32), (4, 4, 4, 4))  # padding_mode default CONSTANT
    random_horizontal_op = vision.RandomHorizontalFlip()
    resize_op = vision.Resize((resize_height, resize_width))  # interpolation default BILINEAR
    rescale_op = vision.Rescale(1.0 / 255.0, 0.0)
    #normalize_op = vision.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    normalize_op = vision.Normalize((0.4914, 0.4822, 0.4465), (0.24703233, 0.24348505, 0.26158768))
    changeswap_op = vision.HWC2CHW()
    type_cast_op = c_transforms.TypeCast(mstype.int32)

    c_trans = []
    if training:
        c_trans = [random_crop_op, random_horizontal_op]
    c_trans += [resize_op, rescale_op, normalize_op, changeswap_op]

    # apply map operations on images
    data_set = data_set.map(operations=type_cast_op, input_columns="label")
    data_set = data_set.map(operations=c_trans, input_columns="image")

    # apply batch operations
    data_set = data_set.batch(batch_size=cifar_cfg.batch_size, drop_remainder=True)

    # apply repeat operations
    data_set = data_set.repeat(repeat_num)

    return data_set
