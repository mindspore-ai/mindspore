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
create train or eval dataset.
"""
import os
import mindspore.common.dtype as mstype
import mindspore.dataset.engine as de
import mindspore.dataset.transforms.py_transforms as py_transforms
import mindspore.dataset.transforms.c_transforms as c_transforms
import mindspore.dataset.vision.py_transforms as py_vision
from mindspore.dataset.vision import Inter

def create_dataset(dataset_path, do_train, config, platform, repeat_num=1, batch_size=1):
    """
    create a train or eval dataset

    Args:
        dataset_path(string): the path of dataset.
        do_train(bool): whether dataset is used for train or eval.
        repeat_num(int): the repeat times of dataset. Default: 1
        batch_size(int): the batch size of dataset. Default: 32

    Returns:
        dataset
    """
    if platform == "Ascend":
        rank_size = int(os.getenv("RANK_SIZE"))
        rank_id = int(os.getenv("RANK_ID"))
        if rank_size == 1:
            ds = de.MindDataset(dataset_path, num_parallel_workers=8, shuffle=True)
        else:
            ds = de.MindDataset(dataset_path, num_parallel_workers=8, shuffle=True,
                                num_shards=rank_size, shard_id=rank_id)
    elif platform == "GPU":
        if do_train:
            from mindspore.communication.management import get_rank, get_group_size
            ds = de.MindDataset(dataset_path, num_parallel_workers=8, shuffle=True,
                                num_shards=get_group_size(), shard_id=get_rank())
        else:
            ds = de.MindDataset(dataset_path, num_parallel_workers=8, shuffle=False)
    else:
        raise ValueError("Unsupported platform.")

    resize_height = config.image_height
    resize_width = config.image_width
    buffer_size = 1000

    # define map operations
    random_resize_crop_bicubic = py_vision.RandomResizedCrop(size=(resize_height, resize_width),
                                                             scale=(0.08, 1.0), ratio=(3./4., 4./3.),
                                                             interpolation=Inter.BICUBIC)
    random_horizontal_flip_op = py_vision.RandomHorizontalFlip(0.5)
    color_jitter = 0.4
    adjust_range = (max(0, 1 - color_jitter), 1 + color_jitter)
    random_color_jitter_op = py_vision.RandomColorAdjust(brightness=adjust_range,
                                                         contrast=adjust_range,
                                                         saturation=adjust_range)

    decode_p = py_vision.Decode()
    resize_p = py_vision.Resize(int(resize_height), interpolation=Inter.BICUBIC)
    center_crop_p = py_vision.CenterCrop(resize_height)
    totensor = py_vision.ToTensor()
    normalize_p = py_vision.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

    if do_train:
        trans = py_transforms.Compose([decode_p, random_resize_crop_bicubic, random_horizontal_flip_op,
                                       random_color_jitter_op, totensor, normalize_p])
    else:
        trans = py_transforms.Compose([decode_p, resize_p, center_crop_p, totensor, normalize_p])

    type_cast_op = c_transforms.TypeCast(mstype.int32)

    ds = ds.map(input_columns="image", operations=trans, num_parallel_workers=8)
    ds = ds.map(input_columns="label_list", operations=type_cast_op, num_parallel_workers=8)

    # apply shuffle operations
    ds = ds.shuffle(buffer_size=buffer_size)

    # apply batch operations
    ds = ds.batch(batch_size, drop_remainder=True)

    # apply dataset repeat operation
    ds = ds.repeat(repeat_num)
    return ds
