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


def create_dataset(dataset_path, do_train, config, platform, repeat_num=1, batch_size=32):
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
            ds = de.ImageFolderDatasetV2(dataset_path, num_parallel_workers=8, shuffle=True)
        else:
            ds = de.ImageFolderDatasetV2(dataset_path, num_parallel_workers=8, shuffle=True,
                                         num_shards=rank_size, shard_id=rank_id)
    elif platform == "GPU":
        from mindspore.communication.management import get_rank, get_group_size
        ds = de.ImageFolderDatasetV2(dataset_path, num_parallel_workers=8, shuffle=True,
                                     num_shards=get_group_size(), shard_id=get_rank())
    else:
        raise ValueError("Unsupport platform.")

    resize_height = config.image_height
    resize_width = config.image_width
    buffer_size = 1000

    # define map operations
    decode_op = C.Decode()
    resize_crop_op = C.RandomCropDecodeResize(resize_height, scale=(0.08, 1.0), ratio=(0.75, 1.333))
    horizontal_flip_op = C.RandomHorizontalFlip(prob=0.5)

    resize_op = C.Resize((256, 256))
    center_crop = C.CenterCrop(resize_width)
    rescale_op = C.RandomColorAdjust(brightness=0.4, contrast=0.4, saturation=0.4)
    normalize_op = C.Normalize(mean=[0.485*255, 0.456*255, 0.406*255], std=[0.229*255, 0.224*255, 0.225*255])
    change_swap_op = C.HWC2CHW()

    if do_train:
        trans = [resize_crop_op, horizontal_flip_op, rescale_op, normalize_op, change_swap_op]
    else:
        trans = [decode_op, resize_op, center_crop, normalize_op, change_swap_op]

    type_cast_op = C2.TypeCast(mstype.int32)

    ds = ds.map(input_columns="image", operations=trans, num_parallel_workers=8)
    ds = ds.map(input_columns="label", operations=type_cast_op, num_parallel_workers=8)

    # apply shuffle operations
    ds = ds.shuffle(buffer_size=buffer_size)

    # apply batch operations
    ds = ds.batch(batch_size, drop_remainder=True)

    # apply dataset repeat operation
    ds = ds.repeat(repeat_num)

    return ds
