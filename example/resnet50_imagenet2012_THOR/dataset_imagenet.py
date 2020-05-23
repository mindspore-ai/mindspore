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
import mindspore.dataset.transforms.c_transforms as C2
import mindspore.dataset.transforms.vision.c_transforms as V_C


def create_dataset(dataset_path, do_train, repeat_num=1, batch_size=32):
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

    device_num = int(os.getenv("RANK_SIZE"))
    rank_id = int(os.getenv("RANK_ID"))

    if device_num == 1:
        ds = de.ImageFolderDatasetV2(dataset_path, num_parallel_workers=8, shuffle=False)
    else:
        ds = de.ImageFolderDatasetV2(dataset_path, num_parallel_workers=8, shuffle=True,
                                     num_shards=device_num, shard_id=rank_id)

    image_size = 224
    mean = [0.485 * 255, 0.456 * 255, 0.406 * 255]
    std = [0.229 * 255, 0.224 * 255, 0.225 * 255]
    if do_train:
        transform_img = [
            V_C.RandomCropDecodeResize(image_size, scale=(0.08, 1.0), ratio=(0.75, 1.333)),
            V_C.RandomHorizontalFlip(prob=0.5),
            V_C.Normalize(mean=mean, std=std),
            V_C.HWC2CHW()
        ]
    else:
        transform_img = [
            V_C.Decode(),
            V_C.Resize((256, 256)),
            V_C.CenterCrop(image_size),
            V_C.Normalize(mean=mean, std=std),
            V_C.HWC2CHW()
        ]
    # type_cast_op = C2.TypeCast(mstype.float16)
    type_cast_op = C2.TypeCast(mstype.int32)

    ds = ds.map(input_columns="image", operations=transform_img, num_parallel_workers=8)
    ds = ds.map(input_columns="label", operations=type_cast_op, num_parallel_workers=8)

    # apply shuffle operations
    # ds = ds.shuffle(buffer_size=config.buffer_size)

    # apply batch operations
    ds = ds.batch(batch_size, drop_remainder=True)

    # apply dataset repeat operation
    ds = ds.repeat(repeat_num)

    return ds
