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
create train or eval dataset.
"""
import os
import mindspore.common.dtype as mstype
import mindspore.dataset as ds
import mindspore.dataset.vision.c_transforms as C
import mindspore.dataset.transforms.c_transforms as C2
from mindspore.communication.management import init, get_rank, get_group_size


def create_dataset1(dataset_path, do_train, repeat_num=1, batch_size=32, target="Ascend"):
    """
    create a train or evaluate cifar10 dataset for mobilenet
    Args:
        dataset_path(string): the path of dataset.
        do_train(bool): whether dataset is used for train or eval.
        repeat_num(int): the repeat times of dataset. Default: 1
        batch_size(int): the batch size of dataset. Default: 32
        target(str): the device target. Default: Ascend

    Returns:
        dataset
    """
    if target == "Ascend":
        device_num, rank_id = _get_rank_info()
    else:
        init()
        rank_id = get_rank()
        device_num = get_group_size()

    if device_num == 1:
        data_set = ds.Cifar10Dataset(dataset_path, num_parallel_workers=12, shuffle=True)
    else:
        data_set = ds.Cifar10Dataset(dataset_path, num_parallel_workers=12, shuffle=True,
                                     num_shards=device_num, shard_id=rank_id)

    # define map operations
    trans = []
    if do_train:
        trans += [
            C.RandomCrop((32, 32), (4, 4, 4, 4)),
            C.RandomHorizontalFlip(prob=0.5)
        ]

    trans += [
        C.Resize((224, 224)),
        C.Rescale(1.0 / 255.0, 0.0),
        C.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]),
        C.HWC2CHW()
    ]

    type_cast_op = C2.TypeCast(mstype.int32)

    data_set = data_set.map(operations=type_cast_op, input_columns="label", num_parallel_workers=12)
    data_set = data_set.map(operations=trans, input_columns="image", num_parallel_workers=12)

    # apply batch operations
    data_set = data_set.batch(batch_size, drop_remainder=True)
    # apply dataset repeat operation
    data_set = data_set.repeat(repeat_num)

    return data_set


def create_dataset2(dataset_path, do_train, repeat_num=1, batch_size=32, target="Ascend"):
    """
    create a train or eval imagenet2012 dataset for mobilenet

    Args:
        dataset_path(string): the path of dataset.
        do_train(bool): whether dataset is used for train or eval.
        repeat_num(int): the repeat times of dataset. Default: 1
        batch_size(int): the batch size of dataset. Default: 32
        target(str): the device target. Default: Ascend

    Returns:
        dataset
    """
    if target == "Ascend":
        device_num, rank_id = _get_rank_info()
    else:
        init()
        rank_id = get_rank()
        device_num = get_group_size()

    if device_num == 1:
        data_set = ds.ImageFolderDataset(dataset_path, num_parallel_workers=12, shuffle=True)
    else:
        data_set = ds.ImageFolderDataset(dataset_path, num_parallel_workers=12, shuffle=True,
                                         num_shards=device_num, shard_id=rank_id)

    image_size = 224
    mean = [0.485 * 255, 0.456 * 255, 0.406 * 255]
    std = [0.229 * 255, 0.224 * 255, 0.225 * 255]

    # define map operations
    if do_train:
        trans = [
            C.RandomCropDecodeResize(image_size, scale=(0.08, 1.0), ratio=(0.75, 1.333)),
            C.RandomHorizontalFlip(prob=0.5),
            C.Normalize(mean=mean, std=std),
            C.HWC2CHW()
        ]
    else:
        trans = [
            C.Decode(),
            C.Resize(256),
            C.CenterCrop(image_size),
            C.Normalize(mean=mean, std=std),
            C.HWC2CHW()
        ]

    type_cast_op = C2.TypeCast(mstype.int32)

    data_set = data_set.map(operations=trans, input_columns="image", num_parallel_workers=12)
    data_set = data_set.map(operations=type_cast_op, input_columns="label", num_parallel_workers=12)

    # apply batch operations
    data_set = data_set.batch(batch_size, drop_remainder=True)

    # apply dataset repeat operation
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
