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
"""
Data operations, will be used in train.py and eval.py
"""
import os

import mindspore.common.dtype as mstype
import mindspore.dataset as ds
import mindspore.dataset.transforms.c_transforms as C
import mindspore.dataset.vision.c_transforms as vision
from src.model_utils.config import config as imagenet_cfg

def create_dataset_cifar(dataset_path,
                         do_train,
                         repeat_num=1,
                         batch_size=32,
                         target="Ascend"):
    """
    create a train or evaluate cifar10 dataset
    Args:
        dataset_path(string): the path of dataset.
        do_train(bool): whether dataset is used for train or eval.
        repeat_num(int): the repeat times of dataset. Default: 1
        batch_size(int): the batch size of dataset. Default: 32
        target(str): the device target. Default: Ascend

    Returns:
        dataset
    """
    if target == "CPU":
        device_num = 1
    else:
        device_num, rank_id = _get_rank_info()

    if device_num == 1:
        data_set = ds.Cifar10Dataset(dataset_path,
                                     num_parallel_workers=8,
                                     shuffle=True)
    else:
        data_set = ds.Cifar10Dataset(dataset_path,
                                     num_parallel_workers=8,
                                     shuffle=True,
                                     num_shards=device_num,
                                     shard_id=rank_id)

    # define map operations
    if do_train:
        trans = [
            vision.RandomCrop((32, 32), (4, 4, 4, 4)),
            vision.RandomHorizontalFlip(prob=0.5),
            vision.RandomColorAdjust(brightness=0.4, contrast=0.4, saturation=0.4),
            vision.Resize((227, 227)),
            vision.Rescale(1.0 / 255.0, 0.0),
            vision.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]),
            vision.CutOut(112),
            vision.HWC2CHW()
        ]
    else:
        trans = [
            vision.Resize((227, 227)),
            vision.Rescale(1.0 / 255.0, 0.0),
            vision.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]),
            vision.HWC2CHW()
        ]

    type_cast_op = C.TypeCast(mstype.int32)

    data_set = data_set.map(operations=type_cast_op,
                            input_columns="label",
                            num_parallel_workers=8)
    data_set = data_set.map(operations=trans,
                            input_columns="image",
                            num_parallel_workers=8)

    # apply batch operations
    data_set = data_set.batch(batch_size, drop_remainder=True)

    # apply dataset repeat operation
    data_set = data_set.repeat(repeat_num)

    return data_set

def create_dataset_imagenet(dataset_path, repeat_num=1, training=True,
                            num_parallel_workers=None, shuffle=None):
    """
    create a train or eval imagenet2012 dataset for resnet50

    Args:
        dataset_path(string): the path of dataset.
        do_train(bool): whether dataset is used for train or eval.
        repeat_num(int): the repeat times of dataset. Default: 1
        batch_size(int): the batch size of dataset. Default: 32
        target(str): the device target. Default: Ascend

    Returns:
        dataset
    """

    device_num, rank_id = _get_rank_info()

    if device_num == 1:
        data_set = ds.ImageFolderDataset(dataset_path, num_parallel_workers=num_parallel_workers, shuffle=shuffle)
    else:
        data_set = ds.ImageFolderDataset(dataset_path, num_parallel_workers=num_parallel_workers, shuffle=shuffle,
                                         num_shards=device_num, shard_id=rank_id)

    assert imagenet_cfg.image_height == imagenet_cfg.image_width, "image_height not equal image_width"
    image_size = imagenet_cfg.image_height
    mean = [0.485 * 255, 0.456 * 255, 0.406 * 255]
    std = [0.229 * 255, 0.224 * 255, 0.225 * 255]

    # define map operations
    if training:
        transform_img = [
            vision.RandomCropDecodeResize(image_size, scale=(0.08, 1.0), ratio=(0.75, 1.333)),
            vision.RandomHorizontalFlip(prob=0.5),
            vision.RandomColorAdjust(0.4, 0.4, 0.4, 0.1),
            vision.Normalize(mean=mean, std=std),
            vision.HWC2CHW()
        ]
    else:
        transform_img = [
            vision.Decode(),
            vision.Resize(256),
            vision.CenterCrop(image_size),
            vision.Normalize(mean=mean, std=std),
            vision.HWC2CHW()
        ]

    transform_label = [C.TypeCast(mstype.int32)]

    data_set = data_set.map(input_columns="image", num_parallel_workers=8, operations=transform_img)
    data_set = data_set.map(input_columns="label", num_parallel_workers=8, operations=transform_label)

    # apply batch operations
    data_set = data_set.batch(imagenet_cfg.batch_size, drop_remainder=True)

    # apply dataset repeat operation
    data_set = data_set.repeat(repeat_num)

    return data_set


def _get_rank_info():
    """
    get rank size and rank id
    """
    rank_size = int(os.environ.get("RANK_SIZE", 1))

    if rank_size > 1:
        from mindspore.communication.management import init, get_rank, get_group_size
        init()
        rank_size = get_group_size()
        rank_id = get_rank()
    else:
        rank_size = rank_id = None

    return rank_size, rank_id
