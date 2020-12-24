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
from src.config import config
import mindspore.common.dtype as mstype
import mindspore.dataset as ds
import mindspore.dataset.transforms.c_transforms as C2
import mindspore.dataset.vision.c_transforms as C


def create_dataset(dataset_path, do_train, device_num=1, rank=0):
    """
    create a train or eval dataset

    Args:
        dataset_path(string): the path of dataset.
        do_train(bool): whether dataset is used for train or eval.
        rank (int): The shard ID within num_shards (default=None).
        group_size (int): Number of shards that the dataset should be divided into (default=None).
        repeat_num(int): the repeat times of dataset. Default: 1.

    Returns:
        dataset
    """

    if device_num == 1:
        data_set = ds.ImageFolderDataset(dataset_path, num_parallel_workers=8, shuffle=True)
    else:
        data_set = ds.ImageFolderDataset(dataset_path, num_parallel_workers=8, shuffle=True,
                                         num_shards=device_num, shard_id=rank)
    # define map operations
    if do_train:
        trans = [
            C.RandomCropDecodeResize(224),
            C.RandomHorizontalFlip(prob=0.5),
            C.RandomColorAdjust(brightness=0.4, contrast=0.4, saturation=0.4)
        ]
    else:
        trans = [
            C.Decode(),
            C.Resize(239),
            C.CenterCrop(224)
        ]
    trans += [
        C.Normalize(mean=[0.485 * 255, 0.456 * 255, 0.406 * 255], std=[0.229 * 255, 0.224 * 255, 0.225 * 255]),
        C.HWC2CHW(),
    ]

    type_cast_op = C2.TypeCast(mstype.int32)
    data_set = data_set.map(input_columns="image", operations=trans, num_parallel_workers=8)
    data_set = data_set.map(input_columns="label", operations=type_cast_op, num_parallel_workers=8)
    # apply batch operations
    data_set = data_set.batch(config.batch_size, drop_remainder=True)
    return data_set
