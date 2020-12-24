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
""" create train dataset. """

from functools import partial

import mindspore.common.dtype as mstype
import mindspore.dataset as ds
import mindspore.dataset.transforms.c_transforms as C2
import mindspore.dataset.vision.c_transforms as C


def create_dataset(dataset_path, config, repeat_num=1, batch_size=32):
    """
    create a train dataset

    Args:
        dataset_path(string): the path of dataset.
        config(EasyDict)：the basic config for training
        repeat_num(int): the repeat times of dataset. Default: 1.
        batch_size(int): the batch size of dataset. Default: 32.

    Returns:
        dataset
    """

    load_func = partial(ds.Cifar10Dataset, dataset_path)
    data_set = load_func(num_parallel_workers=8, shuffle=False)

    resize_height = config.image_height
    resize_width = config.image_width

    mean = [0.485 * 255, 0.456 * 255, 0.406 * 255]
    std = [0.229 * 255, 0.224 * 255, 0.225 * 255]

    # define map operations
    resize_op = C.Resize((resize_height, resize_width))
    normalize_op = C.Normalize(mean=mean, std=std)
    changeswap_op = C.HWC2CHW()
    c_trans = [resize_op, normalize_op, changeswap_op]

    type_cast_op = C2.TypeCast(mstype.int32)

    data_set = data_set.map(operations=c_trans, input_columns="image",
                            num_parallel_workers=8)
    data_set = data_set.map(operations=type_cast_op,
                            input_columns="label", num_parallel_workers=8)

    # apply batch operations
    data_set = data_set.batch(batch_size, drop_remainder=True)

    # apply dataset repeat operation
    data_set = data_set.repeat(repeat_num)

    return data_set
