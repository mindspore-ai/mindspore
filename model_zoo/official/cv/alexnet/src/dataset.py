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
Produce the dataset
"""

import os

import mindspore.dataset as ds
import mindspore.dataset.transforms.c_transforms as C
import mindspore.dataset.vision.c_transforms as CV
from mindspore.common import dtype as mstype
from mindspore.communication.management import get_rank, get_group_size
from .config import alexnet_cifar10_cfg, alexnet_imagenet_cfg


def create_dataset_cifar10(data_path, batch_size=32, repeat_size=1, status="train", target="Ascend"):
    """
    create dataset for train or test
    """

    if target == "Ascend":
        device_num, rank_id = _get_rank_info()

    if target != "Ascend" or device_num == 1:
        cifar_ds = ds.Cifar10Dataset(data_path)
    else:
        cifar_ds = ds.Cifar10Dataset(data_path, num_parallel_workers=8, shuffle=True,
                                     num_shards=device_num, shard_id=rank_id)
    rescale = 1.0 / 255.0
    shift = 0.0
    cfg = alexnet_cifar10_cfg

    resize_op = CV.Resize((cfg.image_height, cfg.image_width))
    rescale_op = CV.Rescale(rescale, shift)
    normalize_op = CV.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    if status == "train":
        random_crop_op = CV.RandomCrop([32, 32], [4, 4, 4, 4])
        random_horizontal_op = CV.RandomHorizontalFlip()
    channel_swap_op = CV.HWC2CHW()
    typecast_op = C.TypeCast(mstype.int32)
    cifar_ds = cifar_ds.map(input_columns="label", operations=typecast_op, num_parallel_workers=8)
    if status == "train":
        cifar_ds = cifar_ds.map(input_columns="image", operations=random_crop_op, num_parallel_workers=8)
        cifar_ds = cifar_ds.map(input_columns="image", operations=random_horizontal_op, num_parallel_workers=8)
    cifar_ds = cifar_ds.map(input_columns="image", operations=resize_op, num_parallel_workers=8)
    cifar_ds = cifar_ds.map(input_columns="image", operations=rescale_op, num_parallel_workers=8)
    cifar_ds = cifar_ds.map(input_columns="image", operations=normalize_op, num_parallel_workers=8)
    cifar_ds = cifar_ds.map(input_columns="image", operations=channel_swap_op, num_parallel_workers=8)

    cifar_ds = cifar_ds.shuffle(buffer_size=cfg.buffer_size)
    cifar_ds = cifar_ds.batch(batch_size, drop_remainder=True)
    cifar_ds = cifar_ds.repeat(repeat_size)
    return cifar_ds


def create_dataset_imagenet(dataset_path, batch_size=32, repeat_num=1, training=True,
                            num_parallel_workers=None, shuffle=None, sampler=None, class_indexing=None):
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
    cfg = alexnet_imagenet_cfg

    num_parallel_workers = 16
    if device_num == 1:
        num_parallel_workers = 48
        ds.config.set_prefetch_size(8)
    else:
        ds.config.set_numa_enable(True)
    data_set = ds.ImageFolderDataset(dataset_path, num_parallel_workers=4,
                                     shuffle=shuffle, sampler=sampler, class_indexing=class_indexing,
                                     num_shards=device_num, shard_id=rank_id)

    assert cfg.image_height == cfg.image_width, "imagenet_cfg.image_height not equal imagenet_cfg.image_width"
    image_size = cfg.image_height
    mean = [0.485 * 255, 0.456 * 255, 0.406 * 255]
    std = [0.229 * 255, 0.224 * 255, 0.225 * 255]

    # define map operations
    if training:
        transform_img = [
            CV.RandomCropDecodeResize(image_size, scale=(0.08, 1.0), ratio=(0.75, 1.333)),
            CV.RandomHorizontalFlip(prob=0.5),
            CV.Normalize(mean=mean, std=std),
            CV.HWC2CHW()
        ]
    else:
        transform_img = [
            CV.Decode(),
            CV.Resize((256, 256)),
            CV.CenterCrop(image_size),
            CV.Normalize(mean=mean, std=std),
            CV.HWC2CHW()
        ]

    data_set = data_set.map(input_columns="image", num_parallel_workers=num_parallel_workers,
                            operations=transform_img)

    data_set = data_set.batch(batch_size, drop_remainder=True)

    # apply dataset repeat operation
    if repeat_num > 1:
        data_set = data_set.repeat(repeat_num)

    return data_set


def _get_rank_info():
    """
    get rank size and rank id
    """
    rank_size = int(os.environ.get("RANK_SIZE", 1))

    if rank_size > 1:
        rank_size = get_group_size()
        rank_id = get_rank()
    else:
        rank_size = 1
        rank_id = 0

    return rank_size, rank_id
