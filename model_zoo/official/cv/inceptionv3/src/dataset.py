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
import mindspore.common.dtype as mstype
import mindspore.dataset.engine as de
import mindspore.dataset.transforms.c_transforms as C2
import mindspore.dataset.transforms.vision.c_transforms as C
from src.config import config_gpu as cfg


def create_dataset(dataset_path, do_train, rank, group_size, repeat_num=1):
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
    if group_size == 1:
        ds = de.ImageFolderDatasetV2(dataset_path, num_parallel_workers=cfg.work_nums, shuffle=True)
    else:
        ds = de.ImageFolderDatasetV2(dataset_path, num_parallel_workers=cfg.work_nums, shuffle=True,
                                     num_shards=group_size, shard_id=rank)
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
    ds = ds.map(input_columns="image", operations=trans, num_parallel_workers=cfg.work_nums)
    ds = ds.map(input_columns="label", operations=type_cast_op, num_parallel_workers=cfg.work_nums)
    # apply batch operations
    ds = ds.batch(cfg.batch_size, drop_remainder=True)
    # apply dataset repeat operation
    ds = ds.repeat(repeat_num)
    return ds
