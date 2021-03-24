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
import mindspore.common.dtype as mstype
import mindspore.dataset.engine as de
import mindspore.dataset.vision.c_transforms as C
import mindspore.dataset.transforms.c_transforms as C2
from mindspore.communication.management import init, get_rank, get_group_size

def create_dataset_ImageNet(dataset_path, do_train, repeat_num=1, batch_size=32, target="Ascend"):
    """
        create a train or eval imagenet2012 dataset for hardnet

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
        init("nccl")
        rank_id = get_rank()
        device_num = get_group_size()

    image_size = 224
    mean = [0.485 * 255, 0.456 * 255, 0.406 * 255]
    std = [0.229 * 255, 0.224 * 255, 0.225 * 255]

    if device_num == 1:
        ds = de.ImageFolderDataset(dataset_path, num_parallel_workers=8, shuffle=True)
    else:
        ds = de.ImageFolderDataset(dataset_path, num_parallel_workers=8, shuffle=True,
                                   num_shards=device_num, shard_id=rank_id)

    # define map operations
    if do_train:
        trans = [
            C.Decode(),
            C.RandomResizedCrop(image_size),
            C.RandomHorizontalFlip(),
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

    ds = ds.map(input_columns="image", num_parallel_workers=8, operations=trans)
    ds = ds.map(input_columns="label", num_parallel_workers=8, operations=type_cast_op)

    # apply batch operations
    ds = ds.batch(batch_size, drop_remainder=True)

    # apply dataset repeat operation
    ds = ds.repeat(repeat_num)

    return ds

def _get_rank_info():
    """
    get rank size and rank id
    """

    rank_size = int(os.environ.get("RANK_SIZE"))

    if rank_size > 1:
        rank_size = int(os.environ.get("RANK_SIZE"))
        rank_id = int(os.environ.get("RANK_ID"))
    else:
        rank_size = 1
        rank_id = 0

    return rank_size, rank_id
