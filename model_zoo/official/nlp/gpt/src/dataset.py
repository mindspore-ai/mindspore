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
Create dataset for training and evaluating
"""

import os
import mindspore.dataset as ds
import mindspore.dataset.transforms.c_transforms as C
import mindspore.common.dtype as mstype


def create_dataset(batch_size, data_path, device_num=1, rank=0, drop=True):
    """
    Create dataset

    Inputs:
        batch_size: batch size
        data_path: path of your MindRecord files
        device_num: total device number
        rank: current rank id
        drop: whether drop remainder

    Returns:
        dataset: the dataset for training or evaluating
    """
    home_path = os.path.join(os.getcwd(), data_path)
    data = [os.path.join(home_path, name) for name in os.listdir(data_path) if name.endswith("mindrecord")]
    print(data)
    dataset = ds.MindDataset(data, columns_list=["input_ids"], shuffle=True, num_shards=device_num, shard_id=rank)
    type_cast_op = C.TypeCast(mstype.int32)
    dataset = dataset.map(input_columns="input_ids", operations=type_cast_op)
    dataset = dataset.batch(batch_size, drop_remainder=drop)
    dataset = dataset.repeat(1)
    return dataset
