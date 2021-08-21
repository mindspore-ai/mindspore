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
"""Read data."""

import mindspore.common.dtype as mstype
import mindspore.dataset as ds
import mindspore.dataset.transforms.c_transforms as C


def create_train_dataset(data_file=None, do_shuffle=True, device_num=1, rank=0, batch_size=1,
                         num=None, num_parallel_workers=1):
    """Read train data"""
    dataset = ds.MindDataset(data_file,
                             columns_list=["input_mask", "src_ids", "pos_ids", "sent_ids", "wn_concept_ids",
                                           "nell_concept_ids", "start_positions", "end_positions"],
                             shuffle=do_shuffle, num_shards=device_num, shard_id=rank,
                             num_samples=num, num_parallel_workers=num_parallel_workers)

    type_int32 = C.TypeCast(mstype.int32)
    type_float32 = C.TypeCast(mstype.float32)
    dataset = dataset.map(operations=type_int32, input_columns="src_ids")
    dataset = dataset.map(operations=type_int32, input_columns="pos_ids")
    dataset = dataset.map(operations=type_int32, input_columns="sent_ids")
    dataset = dataset.map(operations=type_int32, input_columns="wn_concept_ids")
    dataset = dataset.map(operations=type_int32, input_columns="nell_concept_ids")
    dataset = dataset.map(operations=type_float32, input_columns="input_mask")
    dataset = dataset.map(operations=type_int32, input_columns="start_positions")
    dataset = dataset.map(operations=type_int32, input_columns="end_positions")

    dataset = dataset.batch(batch_size, True, 8)

    return dataset


def create_dev_dataset(data_file=None, do_shuffle=True, batch_size=1, repeat_count=1):
    """Read dev data"""
    dataset = ds.MindDataset(data_file,
                             columns_list=["input_mask", "src_ids", "pos_ids", "sent_ids",
                                           "wn_concept_ids", "nell_concept_ids", "unique_id"],
                             shuffle=do_shuffle)

    type_int32 = C.TypeCast(mstype.int32)
    type_float32 = C.TypeCast(mstype.float32)
    dataset = dataset.map(operations=type_int32, input_columns="src_ids")
    dataset = dataset.map(operations=type_int32, input_columns="pos_ids")
    dataset = dataset.map(operations=type_int32, input_columns="sent_ids")
    dataset = dataset.map(operations=type_int32, input_columns="wn_concept_ids")
    dataset = dataset.map(operations=type_int32, input_columns="nell_concept_ids")
    dataset = dataset.map(operations=type_float32, input_columns="input_mask")
    dataset = dataset.map(operations=type_int32, input_columns="unique_id")

    dataset = dataset.repeat(repeat_count)
    dataset = dataset.batch(batch_size, drop_remainder=True)

    return dataset
