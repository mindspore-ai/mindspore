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
import mindspore.dataset as ds
import mindspore.dataset.transforms.c_transforms as C2
import mindspore.dataset.vision.c_transforms as C


def create_dataset(dataset_path, do_train, batch_size=16, device_num=1, rank=0):
    """
    create a train or eval dataset

    Args:
        dataset_path(string): the path of dataset.
        do_train(bool): whether dataset is used for train or eval.
        batch_size(int): the batch size of dataset. Default: 16.
        device_num (int): Number of shards that the dataset should be divided into (default=1).
        rank (int): The shard ID within num_shards (default=0).

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
            C.RandomCropDecodeResize(299),
            C.RandomHorizontalFlip(prob=0.5),
            C.RandomColorAdjust(brightness=0.4, contrast=0.4, saturation=0.4)
        ]
    else:
        trans = [
            C.Decode(),
            C.Resize(320),
            C.CenterCrop(299)
        ]
    trans += [
        C.Normalize(mean=[127.5, 127.5, 127.5], std=[127.5, 127.5, 127.5]),
        C.HWC2CHW(),
        C2.TypeCast(mstype.float32)
    ]

    type_cast_op = C2.TypeCast(mstype.int32)
    data_set = data_set.map(input_columns="image", operations=trans, num_parallel_workers=8)
    data_set = data_set.map(input_columns="label", operations=type_cast_op, num_parallel_workers=8)
    # apply batch operations
    data_set = data_set.batch(batch_size, drop_remainder=True)
    return data_set
