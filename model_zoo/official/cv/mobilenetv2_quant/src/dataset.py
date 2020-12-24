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
from functools import partial
import mindspore.common.dtype as mstype
import mindspore.dataset as ds
import mindspore.dataset.vision.c_transforms as C
import mindspore.dataset.transforms.c_transforms as C2
import mindspore.dataset.transforms.py_transforms as P2
import mindspore.dataset.vision.py_transforms as P


def create_dataset(dataset_path, do_train, config, device_target, repeat_num=1, batch_size=32):
    """
    create a train or eval dataset

    Args:
        dataset_path(string): the path of dataset.
        do_train(bool): whether dataset is used for train or eval.
        repeat_num(int): the repeat times of dataset. Default: 1.
        batch_size(int): the batch size of dataset. Default: 32.

    Returns:
        dataset
    """
    if device_target == "Ascend":
        rank_size = int(os.getenv("RANK_SIZE"))
        rank_id = int(os.getenv("RANK_ID"))
        columns_list = ['image', 'label']
        if config.data_load_mode == "mindrecord":
            load_func = partial(ds.MindDataset, dataset_path, columns_list)
        else:
            load_func = partial(ds.ImageFolderDataset, dataset_path)
        if do_train:
            if rank_size == 1:
                data_set = load_func(num_parallel_workers=8, shuffle=True)
            else:
                data_set = load_func(num_parallel_workers=8, shuffle=True,
                                     num_shards=rank_size, shard_id=rank_id)
        else:
            data_set = load_func(num_parallel_workers=8, shuffle=False)
    elif device_target == "GPU":
        if do_train:
            from mindspore.communication.management import get_rank, get_group_size
            data_set = ds.ImageFolderDataset(dataset_path, num_parallel_workers=8, shuffle=True,
                                             num_shards=get_group_size(), shard_id=get_rank())
        else:
            data_set = ds.ImageFolderDataset(dataset_path, num_parallel_workers=8, shuffle=True)
    else:
        raise ValueError("Unsupported device_target.")

    resize_height = 224

    if do_train:
        buffer_size = 20480
        # apply shuffle operations
        data_set = data_set.shuffle(buffer_size=buffer_size)

    # define map operations
    decode_op = C.Decode()
    resize_crop_decode_op = C.RandomCropDecodeResize(resize_height, scale=(0.08, 1.0), ratio=(0.75, 1.333))
    horizontal_flip_op = C.RandomHorizontalFlip(prob=0.5)

    resize_op = C.Resize(256)
    center_crop = C.CenterCrop(resize_height)
    normalize_op = C.Normalize(mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
                               std=[0.229 * 255, 0.224 * 255, 0.225 * 255])
    change_swap_op = C.HWC2CHW()

    if do_train:
        trans = [resize_crop_decode_op, horizontal_flip_op, normalize_op, change_swap_op]
    else:
        trans = [decode_op, resize_op, center_crop, normalize_op, change_swap_op]

    type_cast_op = C2.TypeCast(mstype.int32)

    data_set = data_set.map(operations=trans, input_columns="image", num_parallel_workers=16)
    data_set = data_set.map(operations=type_cast_op, input_columns="label", num_parallel_workers=8)

    # apply batch operations
    data_set = data_set.batch(batch_size, drop_remainder=True)

    # apply dataset repeat operation
    data_set = data_set.repeat(repeat_num)

    return data_set


def create_dataset_py(dataset_path, do_train, config, device_target, repeat_num=1, batch_size=32):
    """
    create a train or eval dataset

    Args:
        dataset_path(string): the path of dataset.
        do_train(bool): whether dataset is used for train or eval.
        repeat_num(int): the repeat times of dataset. Default: 1.
        batch_size(int): the batch size of dataset. Default: 32.

    Returns:
        dataset
    """
    if device_target == "Ascend":
        rank_size = int(os.getenv("RANK_SIZE"))
        rank_id = int(os.getenv("RANK_ID"))
        if do_train:
            if rank_size == 1:
                data_set = ds.ImageFolderDataset(dataset_path, num_parallel_workers=8, shuffle=True)
            else:
                data_set = ds.ImageFolderDataset(dataset_path, num_parallel_workers=8, shuffle=True,
                                                 num_shards=rank_size, shard_id=rank_id)
        else:
            data_set = ds.ImageFolderDataset(dataset_path, num_parallel_workers=8, shuffle=False)
    else:
        raise ValueError("Unsupported device target.")

    resize_height = 224

    if do_train:
        buffer_size = 20480
        # apply shuffle operations
        data_set = data_set.shuffle(buffer_size=buffer_size)

    # define map operations
    decode_op = P.Decode()
    resize_crop_op = P.RandomResizedCrop(resize_height, scale=(0.08, 1.0), ratio=(0.75, 1.333))
    horizontal_flip_op = P.RandomHorizontalFlip(prob=0.5)

    resize_op = P.Resize(256)
    center_crop = P.CenterCrop(resize_height)
    to_tensor = P.ToTensor()
    normalize_op = P.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    if do_train:
        trans = [decode_op, resize_crop_op, horizontal_flip_op, to_tensor, normalize_op]
    else:
        trans = [decode_op, resize_op, center_crop, to_tensor, normalize_op]

    compose = P2.Compose(trans)

    data_set = data_set.map(operations=compose, input_columns="image", num_parallel_workers=8,
                            python_multiprocessing=True)

    # apply batch operations
    data_set = data_set.batch(batch_size, drop_remainder=True)

    # apply dataset repeat operation
    data_set = data_set.repeat(repeat_num)

    return data_set
