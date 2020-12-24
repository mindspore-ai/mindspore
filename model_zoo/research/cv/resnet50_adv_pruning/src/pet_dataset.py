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
import mindspore.dataset as ds
import mindspore.dataset.vision.c_transforms as C
import mindspore.dataset.vision.py_transforms as P
import mindspore.dataset.transforms.c_transforms as C2
import mindspore.dataset.transforms.py_transforms as P2
from mindspore.dataset.vision import Inter


def create_dataset(dataset_path, do_train, config, platform, repeat_num=1, batch_size=100):
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
            data_set = ds.MindDataset(
                dataset_path, num_parallel_workers=8, shuffle=True)
        else:
            data_set = ds.MindDataset(dataset_path, num_parallel_workers=8, shuffle=True,
                                      num_shards=rank_size, shard_id=rank_id)
    elif platform == "GPU":
        if do_train:
            from mindspore.communication.management import get_rank, get_group_size
            data_set = ds.MindDataset(dataset_path, num_parallel_workers=8, shuffle=True,
                                      num_shards=get_group_size(), shard_id=get_rank())
        else:
            data_set = ds.MindDataset(
                dataset_path, num_parallel_workers=8, shuffle=False)
    else:
        raise ValueError("Unsupport platform.")

    resize_height = config.image_height
    buffer_size = 1000

    # define map operations
    resize_crop_op = C.RandomCropDecodeResize(
        resize_height, scale=(0.08, 1.0), ratio=(0.75, 1.333))
    horizontal_flip_op = C.RandomHorizontalFlip(prob=0.5)

    color_op = C.RandomColorAdjust(
        brightness=0.4, contrast=0.4, saturation=0.4)
    rescale_op = C.Rescale(1 / 255.0, 0)
    normalize_op = C.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    change_swap_op = C.HWC2CHW()

    # define python operations
    decode_p = P.Decode()
    resize_p = P.Resize(256, interpolation=Inter.BILINEAR)
    center_crop_p = P.CenterCrop(224)
    totensor = P.ToTensor()
    normalize_p = P.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    composeop = P2.Compose(
        [decode_p, resize_p, center_crop_p, totensor, normalize_p])
    if do_train:
        trans = [resize_crop_op, horizontal_flip_op, color_op,
                 rescale_op, normalize_op, change_swap_op]
    else:
        trans = composeop
    type_cast_op = C2.TypeCast(mstype.int32)

    data_set = data_set.map(input_columns="image", operations=trans,
                            num_parallel_workers=8)
    data_set = data_set.map(input_columns="label_list",
                            operations=type_cast_op, num_parallel_workers=8)

    # apply shuffle operations
    data_set = data_set.shuffle(buffer_size=buffer_size)

    # apply batch operations
    data_set = data_set.batch(batch_size, drop_remainder=True)

    # apply dataset repeat operation
    data_set = data_set.repeat(repeat_num)

    return data_set
