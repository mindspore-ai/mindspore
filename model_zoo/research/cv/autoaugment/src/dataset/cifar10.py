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
Helpers for creating Cifar-10 datasets (optionally with AutoAugment enabled).
"""

import os

import mindspore.common.dtype as mstype
from mindspore.communication.management import init, get_rank, get_group_size
import mindspore.dataset as ds
import mindspore.dataset.transforms.c_transforms as C2
import mindspore.dataset.vision.c_transforms as C

from .autoaugment import Augment


def _get_rank_info():
    """Get rank size and rank id."""
    rank_size = int(os.environ.get('RANK_SIZE', 1))

    if rank_size > 1:
        rank_size = get_group_size()
        rank_id = get_rank()
    else:
        rank_size = 1
        rank_id = 0

    return rank_size, rank_id


def create_dataset(dataset_path, do_train=True, repeat_num=1, batch_size=32,
                   target='Ascend', distribute=False, augment=True):
    """
    Create a train or test cifar10 dataset.

    Args:
        dataset_path (string): Path to the dataset.
        do_train (bool): Whether dataset is used for training or testing.
        repeat_num (int): Repeat times of the dataset.
        batch_size (int): Batch size of the dataset.
        target (str): Device target.
        distribute (bool): For distributed training or not.
        augment (bool): Whether to enable auto-augment or not.

    Returns:
        dataset
    """
    if target == 'Ascend':
        rank_size, rank_id = _get_rank_info()
    else:
        if distribute:
            init()
            rank_id = get_rank()
            rank_size = get_group_size()
        else:
            rank_size = 1

    num_shards = None if rank_size == 1 else rank_size
    shard_id = None if rank_size == 1 else rank_id
    dataset = ds.Cifar10Dataset(
        dataset_path, usage='train' if do_train else 'test',
        num_parallel_workers=8, shuffle=True,
        num_shards=num_shards, shard_id=shard_id,
    )

    # Define map operations
    MEAN = [0.4914, 0.4822, 0.4465]
    STD = [0.2023, 0.1994, 0.2010]
    trans = []
    if do_train and augment:
        trans += [
            Augment(mean=MEAN, std=STD),
        ]
    else:
        if do_train:
            trans += [
                C.RandomCrop((32, 32), (4, 4, 4, 4)),
                C.RandomHorizontalFlip(),
            ]
        trans += [
            C.Rescale(1. / 255., 0.),
            C.Normalize(MEAN, STD),
            C.HWC2CHW(),
        ]
    dataset = dataset.map(operations=trans,
                          input_columns='image', num_parallel_workers=8)

    type_cast_op = C2.TypeCast(mstype.int32)
    dataset = dataset.map(operations=type_cast_op,
                          input_columns='label', num_parallel_workers=8)

    # Apply batch operations
    dataset = dataset.batch(batch_size, drop_remainder=True)

    # Apply dataset repeat operation
    dataset = dataset.repeat(repeat_num)

    return dataset
