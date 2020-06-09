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
import mindspore.common.dtype as mstype
import mindspore.dataset.engine as de
import mindspore.dataset.transforms.vision.c_transforms as C
import mindspore.dataset.transforms.c_transforms as C2
from src.config import config

def create_dataset(dataset_path, do_train, repeat_num=1, batch_size=32):
    """
    create a train or evaluate dataset
    Args:
        dataset_path(string): the path of dataset.
        do_train(bool): whether dataset is used for train or eval.
        repeat_num(int): the repeat times of dataset. Default: 1
        batch_size(int): the batch size of dataset. Default: 32

    Returns:
        dataset
    """
    device_num = int(os.getenv("RANK_SIZE"))
    rank_id = int(os.getenv("RANK_ID"))

    if device_num == 1:
        ds = de.ImageFolderDatasetV2(dataset_path, num_parallel_workers=8, shuffle=True)
    else:
        ds = de.ImageFolderDatasetV2(dataset_path, num_parallel_workers=8, shuffle=True,
                                     num_shards=device_num, shard_id=rank_id)
    resize_height = 224
    rescale = 1.0 / 255.0
    shift = 0.0

    # define map operations
    decode_op = C.Decode()

    random_resize_crop_op = C.RandomResizedCrop(resize_height, (0.08, 1.0), (0.75, 1.33), max_attempts=100)
    horizontal_flip_op = C.RandomHorizontalFlip(rank_id / (rank_id + 1))
    resize_op_256 = C.Resize((256, 256))
    center_crop = C.CenterCrop(224)
    rescale_op = C.Rescale(rescale, shift)
    normalize_op = C.Normalize((0.475, 0.451, 0.392), (0.275, 0.267, 0.278))
    changeswap_op = C.HWC2CHW()

    trans = []
    if do_train:
        trans = [decode_op,
                 random_resize_crop_op,
                 horizontal_flip_op,
                 rescale_op,
                 normalize_op,
                 changeswap_op]

    else:
        trans = [decode_op,
                 resize_op_256,
                 center_crop,
                 rescale_op,
                 normalize_op,
                 changeswap_op]

    type_cast_op = C2.TypeCast(mstype.int32)

    ds = ds.map(input_columns="image", operations=trans, num_parallel_workers=8)
    ds = ds.map(input_columns="label", operations=type_cast_op, num_parallel_workers=8)

    # apply shuffle operations
    ds = ds.shuffle(buffer_size=config.buffer_size)
    # apply batch operations
    ds = ds.batch(batch_size, drop_remainder=True)
    # apply dataset repeat operation
    ds = ds.repeat(repeat_num)

    return ds
