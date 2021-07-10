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
"""create dataset"""
import os
import numpy as np
import mindspore.dataset as ds
from mindspore.common import dtype as mstype
import mindspore.dataset.transforms.c_transforms as CT
from mindspore.communication.management import get_rank, get_group_size


def create_dataset(data_path,
                   flatten_size,
                   batch_size,
                   repeat_size=1,
                   num_parallel_workers=1):
    """create_dataset"""
    device_num, rank_id = _get_rank_info()

    if device_num == 1:
        mnist_ds = ds.MnistDataset(data_path)
    else:
        mnist_ds = ds.MnistDataset(data_path, num_parallel_workers=8, shuffle=True,
                                   num_shards=device_num, shard_id=rank_id)
    type_cast_op = CT.TypeCast(mstype.float32)
    onehot_op = CT.OneHot(num_classes=10)

    mnist_ds = mnist_ds.map(input_columns="label",
                            operations=onehot_op,
                            num_parallel_workers=num_parallel_workers)
    mnist_ds = mnist_ds.map(input_columns="label",
                            operations=type_cast_op,
                            num_parallel_workers=num_parallel_workers)
    mnist_ds = mnist_ds.map(input_columns="image",
                            operations=lambda x: ((x - 127.5) / 127.5).astype("float32"),
                            num_parallel_workers=num_parallel_workers)
    mnist_ds = mnist_ds.map(input_columns="image",
                            operations=lambda x: (x.reshape((flatten_size,))),
                            num_parallel_workers=num_parallel_workers)
    buffer_size = 10000
    mnist_ds = mnist_ds.shuffle(buffer_size=buffer_size)
    mnist_ds = mnist_ds.batch(batch_size, drop_remainder=True)
    mnist_ds = mnist_ds.repeat(repeat_size)

    return mnist_ds


def one_hot(num_classes=10, arr=None):
    """onehot process"""
    if arr is not None:
        arr = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    return np.eye(num_classes)[arr]

def _get_rank_info():
    """
    get rank size and rank id
    """
    rank_size = int(os.environ.get("RANK_SIZE", 1))

    if rank_size > 1:
        rank_size = get_group_size()
        rank_id = get_rank()
    else:
        rank_size = 1
        rank_id = 0

    return rank_size, rank_id
