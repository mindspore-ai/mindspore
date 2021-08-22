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


def get_input_data_batch_slice_map(input_ids, eod_id, rank, dis, eod_reset):
    """
    Generate position_id and attention_mask according to input_ids considering eod reset

    Inputs:
        input_ids: the input token ids
        eod_id: the id for <EOD>
        rank: the current rank
        dis: the slice value for each rank
        eod_reset: whether to open eod reset or not
    returns:
        input_ids: the input token ids
        position_id: the position ids cosidering eod reset
        attention_mask: the attention mask considering eod reset
    """
    rank = int(rank)
    input_ids = input_ids[rank * dis: (rank + 1) * dis]
    if not eod_reset:
        return input_ids
    seq_length = input_ids.shape[1] - 1
    # Initialize position_ids and attention_mask
    batch_input_ids = input_ids
    batch_position_ids = np.ones((dis, seq_length))
    batch_attention_mask = np.ones((dis, seq_length, seq_length))

    # Loop through batches
    for bs_i, _ in enumerate(range(len(input_ids))):
        # Get normal position_ids and attention_mask
        local_ids = input_ids[bs_i]
        batch_attention_mask[bs_i] = np.tril(np.ones(shape=(seq_length, seq_length)))
        batch_position_ids[bs_i] = np.arange(seq_length)
        # Find eod_of_document
        eod_index = batch_position_ids[bs_i, local_ids[:-1] == eod_id].astype(np.int32)
        prev_index = 0
        for i in range(eod_index.size):
            # Reset position_ids and attention_mask considering EOD
            index = eod_index[i]
            batch_attention_mask[bs_i, (index + 1):, :(index + 1)] = 0
            batch_position_ids[bs_i, (index + 1):] -= (index + 1 - prev_index)
            prev_index = index + 1
    return batch_input_ids, batch_position_ids, batch_attention_mask


def create_dataset(batch_size, data_path, device_num=1, rank=0, drop=True, full_batch=False, data_start_index=0,
                   eod_reset=False, eod_id=9, column_name='input_ids', epoch=1, num_samples=None):
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
        epoch: The repeat times of the dataset
    Returns:
        dataset_restore: the dataset for training or evaluating
    """
    ds.config.set_seed(1)
    # Control the size of data queue in the consideration of the memory
    ds.config.set_prefetch_size(1)

    # Get path for source data files
    home_path = os.path.join(os.getcwd(), data_path)
    files = os.listdir(data_path)
    data = [
        os.path.join(home_path, name) for name in files
        if not name.endswith(".db")
    ]
    # Ensure the order of mindrecords is same in all machines, otherwise it will meet loss converge problem.
    data.sort()

    # Load data files and preprocess
    dataset = ds.MindDataset(data[data_start_index:], columns_list=[column_name],
                             shuffle=False, num_samples=num_samples)
    type_cast_op = C.TypeCast(mstype.int32)
    type_cast_op_float = C.TypeCast(mstype.float16)

    if full_batch:
        # no need to slice from the inputs
        rank = 0
        dis = batch_size
    else:
        # Each card slice a small batch from the full batch
        dis = int(batch_size / device_num)
        if batch_size % device_num != 0:
            raise ValueError(
                f"batch size {batch_size} should be a multiple of device number {device_num}."
                " You should change the args: per_batch_size.")

    map_func = (lambda input_ids: get_input_data_batch_slice_map(input_ids, eod_id, rank, dis, eod_reset))
    # If eod_reset enabled, another two inputs will be generated through input_ids
    if eod_reset:
        dataset = dataset.batch(batch_size, drop_remainder=drop)
        dataset = dataset.map(operations=map_func, input_columns=[column_name],
                              output_columns=[column_name, "position_id", "attention_mask"],
                              column_order=[column_name, "position_id", "attention_mask"])
        dataset = dataset.map(input_columns="position_id", operations=type_cast_op)
        dataset = dataset.map(input_columns="attention_mask", operations=type_cast_op_float)
    else:
        dataset = dataset.map(input_columns=[column_name], operations=type_cast_op)
        dataset = dataset.batch(batch_size, drop_remainder=drop)
        dataset = dataset.map(operations=map_func, input_columns=[column_name],
                              output_columns=[column_name])
    dataset = dataset.map(input_columns=column_name, operations=type_cast_op)
    dataset = dataset.repeat(epoch)
    return dataset
