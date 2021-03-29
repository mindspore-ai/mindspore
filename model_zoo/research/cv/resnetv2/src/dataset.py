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
""" dataset.py """
import os
import mindspore.common.dtype as mstype
import mindspore.dataset as ds
import mindspore.dataset.vision.c_transforms as C
import mindspore.dataset.transforms.c_transforms as C2
from mindspore.communication.management import get_rank, get_group_size

def create_dataset1(dataset_path, do_train=True, repeat_num=1, batch_size=32, target="Ascend", distribute=False):
    """
    create a train or evaluate cifar10 dataset for PreActResnet
    Args:
        dataset_path(string): the path of dataset.
        do_train(bool): whether dataset is used for train or eval.
        repeat_num(int): the repeat times of dataset. Default: 1
        batch_size(int): the batch size of dataset. Default: 32
        target(str): the device target. Default: Ascend
        distribute(bool): data for distribute or not. Default: False

    Returns:
        dataset
    """
    if target == "Ascend":
        rank_size, rank_id = _get_rank_info()
    else:
        if distribute:
            raise ValueError("run distribute for GPU not yet supported")
        rank_size = 1
    if rank_size == 1:
        data_set = ds.Cifar10Dataset(dataset_path, num_parallel_workers=8, shuffle=True)
    else:
        data_set = ds.Cifar10Dataset(dataset_path, num_parallel_workers=8, shuffle=True,
                                     num_shards=rank_size, shard_id=rank_id)

    # define map operations
    trans = []
    if do_train:
        trans += [
            C.RandomCrop((32, 32), (4, 4, 4, 4)),
            C.RandomHorizontalFlip()
        ]

    trans += [
        C.Rescale(1.0 / 255.0, 0.0),
        C.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]),
        C.HWC2CHW()
    ]

    type_cast_op = C2.TypeCast(mstype.int32)
    data_set = data_set.map(operations=type_cast_op, input_columns="label", num_parallel_workers=8)
    data_set = data_set.map(operations=trans, input_columns="image", num_parallel_workers=8)

    # apply batch operations
    data_set = data_set.batch(batch_size, drop_remainder=True)
    # apply dataset repeat operation
    data_set = data_set.repeat(repeat_num)

    return data_set


def create_dataset2(dataset_path, do_train=True, repeat_num=1, batch_size=32, target="Ascend", distribute=False):
    """
    create a train or evaluate cifar100 dataset for PreActResnet
    Args:
        dataset_path(string): the path of dataset.
        do_train(bool): whether dataset is used for train or eval.
        repeat_num(int): the repeat times of dataset. Default: 1
        batch_size(int): the batch size of dataset. Default: 32
        target(str): the device target. Default: Ascend
        distribute(bool): data for distribute or not. Default: False

    Returns:
        dataset
    """
    if target == "Ascend":
        device_num, rank_id = _get_rank_info()
    else:
        device_num = 1
    if device_num == 1:
        data_set = ds.Cifar100Dataset(dataset_path, num_parallel_workers=8, shuffle=True)
    else:
        data_set = ds.Cifar100Dataset(dataset_path, num_parallel_workers=8, shuffle=True,
                                      num_shards=device_num, shard_id=rank_id)

    # define map operations
    trans = []
    if do_train:
        trans += [
            C.RandomCrop((32, 32), (4, 4, 4, 4)),
            C.RandomHorizontalFlip()
        ]

    trans += [
        C.Rescale(1.0 / 255.0, 0.0),
        C.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]),
        C.HWC2CHW()
    ]

    type_cast_op = C2.TypeCast(mstype.int32)
    data_set = data_set.map(operations=type_cast_op, input_columns="fine_label", num_parallel_workers=8)
    data_set = data_set.map(operations=trans, input_columns="image", num_parallel_workers=8)

    def del_column(col1, col2, col3, batch_info):
        return(col1, col2,)

    # apply batch operations
    data_set = data_set.batch(batch_size, per_batch_map=del_column,
                              input_columns=['image', 'fine_label', 'coarse_label'],
                              output_columns=['image', 'label'],
                              drop_remainder=True)
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
