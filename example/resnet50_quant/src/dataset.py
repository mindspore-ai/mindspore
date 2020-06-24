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
import mindspore.dataset.engine as de
import mindspore.dataset.transforms.vision.c_transforms as C
import mindspore.dataset.transforms.c_transforms as C2
import mindspore.dataset.transforms.vision.py_transforms as P
from mindspore.communication.management import init, get_rank, get_group_size
from src.config import config


def create_dataset(dataset_path, do_train, repeat_num=1, batch_size=32, target="Ascend"):
    """
    create a train or eval dataset

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
        device_num = int(os.getenv("RANK_SIZE"))
        rank_id = int(os.getenv("RANK_ID"))
    else:
        init("nccl")
        rank_id = get_rank()
        device_num = get_group_size()

    columns_list = ['image', 'label']
    if config.data_load_mode == "mindrecord":
        load_func = partial(de.MindDataset, dataset_path, columns_list)
    else:
        load_func = partial(de.ImageFolderDatasetV2, dataset_path)
    if device_num == 1:
        ds = load_func(num_parallel_workers=8, shuffle=True)
    else:
        ds = load_func(num_parallel_workers=8, shuffle=True,
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

    ds = ds.map(input_columns="image", num_parallel_workers=8, operations=trans)
    ds = ds.map(input_columns="label", num_parallel_workers=8, operations=type_cast_op)

    # apply batch operations
    ds = ds.batch(batch_size, drop_remainder=True)

    # apply dataset repeat operation
    ds = ds.repeat(repeat_num)

    return ds


def create_dataset_py(dataset_path, do_train, repeat_num=1, batch_size=32, target="Ascend"):
    """
    create a train or eval dataset

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
        device_num = int(os.getenv("RANK_SIZE"))
        rank_id = int(os.getenv("RANK_ID"))
    else:
        init("nccl")
        rank_id = get_rank()
        device_num = get_group_size()

    if do_train:
        if device_num == 1:
            ds = de.ImageFolderDatasetV2(dataset_path, num_parallel_workers=8, shuffle=True)
        else:
            ds = de.ImageFolderDatasetV2(dataset_path, num_parallel_workers=8, shuffle=True,
                                         num_shards=device_num, shard_id=rank_id)
    else:
        ds = de.ImageFolderDatasetV2(dataset_path, num_parallel_workers=8, shuffle=False)

    image_size = 224

    # define map operations
    decode_op = P.Decode()
    resize_crop_op = P.RandomResizedCrop(image_size, scale=(0.08, 1.0), ratio=(0.75, 1.333))
    horizontal_flip_op = P.RandomHorizontalFlip(prob=0.5)

    resize_op = P.Resize(256)
    center_crop = P.CenterCrop(image_size)
    to_tensor = P.ToTensor()
    normalize_op = P.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    # define map operations
    if do_train:
        trans = [decode_op, resize_crop_op, horizontal_flip_op, to_tensor, normalize_op]
    else:
        trans = [decode_op, resize_op, center_crop, to_tensor, normalize_op]

    compose = P.ComposeOp(trans)
    ds = ds.map(input_columns="image", operations=compose(), num_parallel_workers=8, python_multiprocessing=True)

    # apply batch operations
    ds = ds.batch(batch_size, drop_remainder=True)

    # apply dataset repeat operation
    ds = ds.repeat(repeat_num)

    return ds
