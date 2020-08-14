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


def create_dataset(dataset_path, config, do_train, repeat_num=1):
    """
    create a train or eval dataset

    Args:
        dataset_path(string): the path of dataset.
        config(dict): config of dataset.
        do_train(bool): whether dataset is used for train or eval.
        repeat_num(int): the repeat times of dataset. Default: 1.

    Returns:
        dataset
    """
    rank = config.rank
    group_size = config.group_size
    if group_size == 1:
        ds = de.ImageFolderDatasetV2(dataset_path, num_parallel_workers=config.work_nums, shuffle=True)
    else:
        ds = de.ImageFolderDatasetV2(dataset_path, num_parallel_workers=config.work_nums, shuffle=True,
                                     num_shards=group_size, shard_id=rank)
    # define map operations
    if do_train:
        trans = [
            C.RandomCropDecodeResize(config.image_size),
            C.RandomHorizontalFlip(prob=0.5),
            C.RandomColorAdjust(brightness=0.4, saturation=0.5) # fast mode
            #C.RandomColorAdjust(brightness=0.4, contrast=0.5, saturation=0.5, hue=0.2)
            ]
    else:
        trans = [
            C.Decode(),
            C.Resize(int(config.image_size/0.875)),
            C.CenterCrop(config.image_size)
            ]
    trans += [
        C.Rescale(1.0 / 255.0, 0.0),
        C.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        C.HWC2CHW()
    ]
    type_cast_op = C2.TypeCast(mstype.int32)
    ds = ds.map(input_columns="image", operations=trans, num_parallel_workers=config.work_nums)
    ds = ds.map(input_columns="label", operations=type_cast_op, num_parallel_workers=config.work_nums)
    # apply batch operations
    ds = ds.batch(config.batch_size, drop_remainder=True)
    # apply dataset repeat operation
    ds = ds.repeat(repeat_num)
    return ds
