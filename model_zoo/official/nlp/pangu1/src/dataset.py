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
"""
Create dataset for training and evaluating
"""

import os
import numpy as np
import mindspore.dataset as ds
import mindspore.dataset.transforms.c_transforms as C
import mindspore.common.dtype as mstype


def get_input_data(input_ids, eod_id, rank, dis):
    """
    Generate position_id and attention_mask according to input_ids considering eod reset

    Inputs:
        input_ids: the input token ids
        eod_id: the id for <EOD>

    returns:
        input_ids: the input token ids
        position_id: the position ids cosidering eod reset
        attention_mask: the attention mask considering eod reset
    """
    rank = int(rank)
    input_ids = input_ids[rank*dis: (rank+1)*dis]
    seq_length = input_ids.shape[1] - 1

    batch_input_ids = input_ids
    batch_position_ids = np.ones((dis, seq_length))
    batch_attention_mask = np.ones((dis, seq_length, seq_length))
    for bs_i, _ in enumerate(range(len(input_ids))):
        local_ids = input_ids[bs_i]
        batch_attention_mask[bs_i] = np.tril(np.ones(shape=(seq_length, seq_length)))
        batch_position_ids[bs_i] = np.arange(seq_length)
        eod_index = batch_position_ids[bs_i, local_ids[:-1] == eod_id].astype(np.int32)
        prev_index = 0
        for i in range(eod_index.size):
            index = eod_index[i]
            batch_attention_mask[bs_i, (index+1):, :(index+1)] = 0
            batch_position_ids[bs_i, (index+1):] -= (index + 1 - prev_index)
            prev_index = index + 1
    return batch_input_ids, batch_position_ids, batch_attention_mask


def create_dataset(batch_size, data_path, device_num=1, rank=0, drop=True, data_start_index=0,
                   eod_reset=False, eod_id=9, column_name='input_ids'):
    """
    Create dataset

    Inputs:
        batch_size: batch size
        data_path: path of your MindRecord files
        device_num: total device number
        rank: current rank id
        drop: whether drop remainder
        eod_reset: whether enable position reset and attention mask reset
        eod_id: the id for <EOD>
        column_name: the column name of the mindrecord file. Default is input_ids

    Returns:
        dataset_restore: the dataset for training or evaluating
    """
    ds.config.set_seed(1)
    home_path = os.path.join(os.getcwd(), data_path)
    files = os.listdir(data_path)
    dis = int(batch_size / device_num)
    if dis <= 0:
        raise ValueError(
            "batch size {} should be a multiple of device number {}.".format(batch_size,
                                                                             device_num))

    data = [
        os.path.join(home_path, name) for name in files
        if not name.endswith(".db")
    ]

    dataset = ds.MindDataset(data[data_start_index:], columns_list=[column_name], shuffle=False)
    type_cast_op = C.TypeCast(mstype.int32)
    type_cast_op_float = C.TypeCast(mstype.float16)
    if eod_reset:
        map_func = (lambda input_ids: get_input_data(input_ids, eod_id, rank, dis))
        dataset = dataset.batch(batch_size, drop_remainder=drop)
        dataset = dataset.map(operations=map_func, input_columns=[column_name],
                              output_columns=["input_ids", "position_id", "attention_mask"],
                              column_order=["input_ids", "position_id", "attention_mask"])
        dataset = dataset.map(input_columns="position_id", operations=type_cast_op)
        dataset = dataset.map(input_columns="attention_mask", operations=type_cast_op_float)
    else:
        raise ValueError("Not supported here")
    dataset = dataset.map(input_columns="input_ids", operations=type_cast_op)
    dataset = dataset.repeat(1)
    return dataset
