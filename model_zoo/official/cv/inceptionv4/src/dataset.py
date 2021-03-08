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
"""Create train or eval dataset."""
import os
import mindspore.common.dtype as mstype
import mindspore.dataset as de
import mindspore.dataset.vision.c_transforms as C
import mindspore.dataset.transforms.c_transforms as C2
from src.config import config


device_id = int(os.getenv('DEVICE_ID', '0'))
device_num = int(os.getenv('RANK_SIZE', '1'))


def create_dataset(dataset_path, do_train, repeat_num=1, batch_size=32, shard_id=0):
    """
    Create a train or eval dataset.

    Args:
        dataset_path (str): The path of dataset.
        do_train (bool): Whether dataset is used for train or eval.
        repeat_num (int): The repeat times of dataset. Default: 1.
        batch_size (int): The batch size of dataset. Default: 32.

    Returns:
        Dataset.
    """

    do_shuffle = bool(do_train)

    if device_num == 1 or not do_train:
        ds = de.ImageFolderDataset(dataset_path, num_parallel_workers=config.work_nums, shuffle=do_shuffle)
    else:
        ds = de.ImageFolderDataset(dataset_path, num_parallel_workers=config.work_nums,
                                   shuffle=do_shuffle, num_shards=device_num, shard_id=shard_id)

    image_length = 299
    if do_train:
        trans = [
            C.RandomCropDecodeResize(image_length, scale=(0.08, 1.0), ratio=(0.75, 1.333)),
            C.RandomHorizontalFlip(prob=0.5),
            C.RandomColorAdjust(brightness=0.4, contrast=0.4, saturation=0.4)
            ]
    else:
        trans = [
            C.Decode(),
            C.Resize(image_length),
            C.CenterCrop(image_length)
            ]
    trans += [
        C.Rescale(1.0 / 255.0, 0.0),
        C.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        C.HWC2CHW()
    ]

    type_cast_op = C2.TypeCast(mstype.int32)

    ds = ds.map(input_columns="label", operations=type_cast_op, num_parallel_workers=config.work_nums)
    ds = ds.map(input_columns="image", operations=trans, num_parallel_workers=config.work_nums)

    # apply batch operations
    ds = ds.batch(batch_size, drop_remainder=True)

    # apply dataset repeat operation
    ds = ds.repeat(repeat_num)
    return ds
