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

"""create tinybert dataset"""

from enum import Enum
import mindspore.common.dtype as mstype
import mindspore.dataset.engine.datasets as de
import mindspore.dataset.transforms.c_transforms as C


class DataType(Enum):
    """Enumerate supported dataset format"""
    TFRECORD = 1
    MINDRECORD = 2


def create_dataset(batch_size=32, device_num=1, rank=0, do_shuffle="true", data_dir=None,
                   data_type='tfrecord', seq_length=128, task_type=mstype.int32, drop_remainder=True):
    """create tinybert dataset"""
    if isinstance(data_dir, list):
        data_files = data_dir
    else:
        data_files = [data_dir]

    columns_list = ["input_ids", "input_mask", "segment_ids", "label_ids"]

    shuffle = (do_shuffle == "true")

    if data_type == 'mindrecord':
        ds = de.MindDataset(data_files, columns_list=columns_list, shuffle=shuffle, num_shards=device_num,
                            shard_id=rank)
    else:
        ds = de.TFRecordDataset(data_files, columns_list=columns_list, shuffle=shuffle, num_shards=device_num,
                                shard_id=rank, shard_equal_rows=(device_num == 1))

    if device_num == 1 and shuffle is True:
        ds = ds.shuffle(10000)

    type_cast_op = C.TypeCast(mstype.int32)
    slice_op = C.Slice(slice(0, seq_length, 1))
    label_type = mstype.int32 if task_type == 'classification' else mstype.float32
    ds = ds.map(operations=[type_cast_op, slice_op], input_columns=["segment_ids"])
    ds = ds.map(operations=[type_cast_op, slice_op], input_columns=["input_mask"])
    ds = ds.map(operations=[type_cast_op, slice_op], input_columns=["input_ids"])
    ds = ds.map(operations=[C.TypeCast(label_type), slice_op], input_columns=["label_ids"])
    # apply batch operations
    ds = ds.batch(batch_size, drop_remainder=drop_remainder)

    return ds
