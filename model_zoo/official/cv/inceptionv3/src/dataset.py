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
import mindspore.dataset.transforms.c_transforms as C2
import mindspore.dataset.vision.c_transforms as C


def create_dataset_imagenet(dataset_path, do_train, cfg, repeat_num=1):
    """
    create a train or eval dataset

    Args:
        dataset_path(string): the path of dataset.
        do_train(bool): whether dataset is used for train or eval.
        cfg (dict): the config for creating dataset.
        repeat_num(int): the repeat times of dataset. Default: 1.

    Returns:
        dataset
    """
    if cfg.group_size == 1:
        data_set = ds.ImageFolderDataset(dataset_path, num_parallel_workers=cfg.work_nums, shuffle=True)
    else:
        data_set = ds.ImageFolderDataset(dataset_path, num_parallel_workers=cfg.work_nums, shuffle=True,
                                         num_shards=cfg.group_size, shard_id=cfg.rank)
    # define map operations
    if do_train:
        trans = [
            C.RandomCropDecodeResize(299, scale=(0.08, 1.0), ratio=(0.75, 1.333)),
            C.RandomHorizontalFlip(prob=0.5),
            C.RandomColorAdjust(brightness=0.4, contrast=0.4, saturation=0.4)
        ]
    else:
        trans = [
            C.Decode(),
            C.Resize(299),
            C.CenterCrop(299)
        ]
    trans += [
        C.Rescale(1.0 / 255.0, 0.0),
        C.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        C.HWC2CHW()
    ]
    type_cast_op = C2.TypeCast(mstype.int32)
    data_set = data_set.map(operations=trans, input_columns="image", num_parallel_workers=cfg.work_nums)
    data_set = data_set.map(operations=type_cast_op, input_columns="label", num_parallel_workers=cfg.work_nums)
    # apply batch operations
    data_set = data_set.batch(cfg.batch_size, drop_remainder=True)
    # apply dataset repeat operation
    data_set = data_set.repeat(repeat_num)
    return data_set


def create_dataset_cifar10(dataset_path, do_train, cfg, repeat_num=1):
    """
    create a train or eval dataset

    Args:
        dataset_path(string): the path of dataset.
        do_train(bool): whether dataset is used for train or eval.
        cfg (dict): the config for creating dataset.
        repeat_num(int): the repeat times of dataset. Default: 1.

    Returns:
        dataset
    """
    dataset_path = os.path.join(dataset_path, "cifar-10-batches-bin" if do_train else "cifar-10-verify-bin")
    if cfg.group_size == 1:
        data_set = ds.Cifar10Dataset(dataset_path, num_parallel_workers=cfg.work_nums, shuffle=True)
    else:
        data_set = ds.Cifar10Dataset(dataset_path, num_parallel_workers=cfg.work_nums, shuffle=True,
                                     num_shards=cfg.group_size, shard_id=cfg.rank)

    # define map operations
    trans = []
    if do_train:
        trans.append(C.RandomCrop((32, 32), (4, 4, 4, 4)))
        trans.append(C.RandomHorizontalFlip(prob=0.5))

    trans.append(C.Resize((299, 299)))
    trans.append(C.Rescale(1.0 / 255.0, 0.0))
    trans.append(C.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]))
    trans.append(C.HWC2CHW())

    type_cast_op = C2.TypeCast(mstype.int32)
    data_set = data_set.map(operations=trans, input_columns="image", num_parallel_workers=cfg.work_nums)
    data_set = data_set.map(operations=type_cast_op, input_columns="label", num_parallel_workers=cfg.work_nums)
    # apply batch operations
    data_set = data_set.batch(cfg.batch_size, drop_remainder=do_train)
    # apply dataset repeat operation
    data_set = data_set.repeat(repeat_num)
    return data_set
