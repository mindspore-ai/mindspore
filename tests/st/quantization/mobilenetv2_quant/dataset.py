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
import mindspore.dataset as ds
import mindspore.common.dtype as mstype
import mindspore.dataset.vision.c_transforms as C
import mindspore.dataset.transforms.c_transforms as C2


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
    cifar_ds = load_func(num_parallel_workers=8, shuffle=False)

    resize_height = config.image_height
    resize_width = config.image_width
    rescale = 1.0 / 255.0
    shift = 0.0

    # define map operations
    # interpolation default BILINEAR
    resize_op = C.Resize((resize_height, resize_width))
    rescale_op = C.Rescale(rescale, shift)
    normalize_op = C.Normalize(
        (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    changeswap_op = C.HWC2CHW()
    type_cast_op = C2.TypeCast(mstype.int32)

    c_trans = [resize_op, rescale_op, normalize_op, changeswap_op]

    # apply map operations on images
    cifar_ds = cifar_ds.map(input_columns="label", operations=type_cast_op)
    cifar_ds = cifar_ds.map(input_columns="image", operations=c_trans)

    # apply batch operations
    cifar_ds = cifar_ds.batch(batch_size, drop_remainder=True)

    # apply dataset repeat operation
    cifar_ds = cifar_ds.repeat(repeat_num)

    return cifar_ds
