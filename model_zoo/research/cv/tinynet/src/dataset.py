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
    """Creat ImageNet training dataset"""
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
    """Creat ImageNet validation dataset"""
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
