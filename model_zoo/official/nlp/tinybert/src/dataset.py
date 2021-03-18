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

"""create tinybert dataset"""

import os
from enum import Enum
import mindspore.common.dtype as mstype
import mindspore.dataset as ds
import mindspore.dataset.transforms.c_transforms as C


class DataType(Enum):
    """Enumerate supported dataset format"""
    TFRECORD = 1
    MINDRECORD = 2


def create_tinybert_dataset(task='td', batch_size=32, device_num=1, rank=0,
                            do_shuffle="true", data_dir=None, schema_dir=None,
                            data_type=DataType.TFRECORD):
    """create tinybert dataset"""
    files = os.listdir(data_dir)
    data_files = []
    for file_name in files:
        if "record" in file_name and "db" not in file_name:
            data_files.append(os.path.join(data_dir, file_name))
    if task == "td":
        columns_list = ["input_ids", "input_mask", "segment_ids", "label_ids"]
    else:
        columns_list = ["input_ids", "input_mask", "segment_ids"]

    shard_equal_rows = True
    shuffle = (do_shuffle == "true")
    if device_num == 1:
        shard_equal_rows = False
        shuffle = False

    if data_type == DataType.MINDRECORD:
        data_set = ds.MindDataset(data_files, columns_list=columns_list,
                                  shuffle=(do_shuffle == "true"), num_shards=device_num, shard_id=rank)
    else:
        data_set = ds.TFRecordDataset(data_files, schema_dir if schema_dir != "" else None, columns_list=columns_list,
                                      shuffle=shuffle, num_shards=device_num, shard_id=rank,
                                      shard_equal_rows=shard_equal_rows)
    if device_num == 1 and shuffle is True:
        data_set = data_set.shuffle(10000)

    type_cast_op = C.TypeCast(mstype.int32)
    data_set = data_set.map(operations=type_cast_op, input_columns="segment_ids")
    data_set = data_set.map(operations=type_cast_op, input_columns="input_mask")
    data_set = data_set.map(operations=type_cast_op, input_columns="input_ids")
    if task == "td":
        data_set = data_set.map(operations=type_cast_op, input_columns="label_ids")
    # apply batch operations
    data_set = data_set.batch(batch_size, drop_remainder=True)

    return data_set
