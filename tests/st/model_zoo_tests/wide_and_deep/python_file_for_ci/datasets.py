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
"""train_imagenet."""

import os
from enum import Enum
import numpy as np
import mindspore.dataset as ds
import mindspore.common.dtype as mstype


class DataType(Enum):
    """
    Enumerate supported dataset format.
    """
    MINDRECORD = 1
    TFRECORD = 2
    H5 = 3


def _get_tf_dataset(data_dir, train_mode=True, epochs=1, batch_size=1000,
                    line_per_sample=1000, rank_size=None, rank_id=None):
    """
    get_tf_dataset
    """
    dataset_files = []
    file_prefix_name = 'train' if train_mode else 'test'
    shuffle = train_mode
    for (dirpath, _, filenames) in os.walk(data_dir):
        for filename in filenames:
            if file_prefix_name in filename and "tfrecord" in filename:
                dataset_files.append(os.path.join(dirpath, filename))
    schema = ds.Schema()
    schema.add_column('feat_ids', de_type=mstype.int32)
    schema.add_column('feat_vals', de_type=mstype.float32)
    schema.add_column('label', de_type=mstype.float32)
    if rank_size is not None and rank_id is not None:
        data_set = ds.TFRecordDataset(dataset_files=dataset_files, shuffle=shuffle, schema=schema,
                                      num_parallel_workers=8,
                                      num_shards=rank_size, shard_id=rank_id, shard_equal_rows=True)
    else:
        data_set = ds.TFRecordDataset(dataset_files=dataset_files, shuffle=shuffle, schema=schema,
                                      num_parallel_workers=8)
    data_set = data_set.batch(int(batch_size / line_per_sample),
                              drop_remainder=True)
    data_set = data_set.map(operations=(lambda x, y, z: (
        np.array(x).flatten().reshape(batch_size, 39),
        np.array(y).flatten().reshape(batch_size, 39),
        np.array(z).flatten().reshape(batch_size, 1))),
                            input_columns=['feat_ids', 'feat_vals', 'label'],
                            num_parallel_workers=8)
    # if train_mode:
    data_set = data_set.repeat(epochs)
    return data_set


def _get_mindrecord_dataset(directory, train_mode=True, epochs=1, batch_size=1000,
                            line_per_sample=1000, rank_size=None, rank_id=None):
    """
    Get dataset with mindrecord format.

    Args:
        directory (str): Dataset directory.
        train_mode (bool): Whether dataset is use for train or eval (default=True).
        epochs (int): Dataset epoch size (default=1).
        batch_size (int): Dataset batch size (default=1000).
        line_per_sample (int): The number of sample per line (default=1000).
        rank_size (int): The number of device, not necessary for single device (default=None).
        rank_id (int): Id of device, not necessary for single device (default=None).

    Returns:
        Dataset.
    """
    file_prefix_name = 'train_input_part.mindrecord' if train_mode else 'test_input_part.mindrecord'
    file_suffix_name = '00' if train_mode else '0'
    shuffle = train_mode

    if rank_size is not None and rank_id is not None:
        data_set = ds.MindDataset(os.path.join(directory, file_prefix_name + file_suffix_name),
                                  columns_list=['feat_ids', 'feat_vals', 'label'],
                                  num_shards=rank_size, shard_id=rank_id, shuffle=shuffle,
                                  num_parallel_workers=8)
    else:
        data_set = ds.MindDataset(os.path.join(directory, file_prefix_name + file_suffix_name),
                                  columns_list=['feat_ids', 'feat_vals', 'label'],
                                  shuffle=shuffle, num_parallel_workers=8)
    data_set = data_set.batch(int(batch_size / line_per_sample), drop_remainder=True)
    data_set = data_set.map(operations=(lambda x, y, z: (np.array(x).flatten().reshape(batch_size, 39),
                                                         np.array(y).flatten().reshape(batch_size, 39),
                                                         np.array(z).flatten().reshape(batch_size, 1))),
                            input_columns=['feat_ids', 'feat_vals', 'label'],
                            num_parallel_workers=8)
    data_set = data_set.repeat(epochs)
    return data_set


def create_dataset(data_dir, train_mode=True, epochs=1, batch_size=1000,
                   data_type=DataType.TFRECORD, line_per_sample=1000, rank_size=None, rank_id=None):
    """
    create_dataset
    """
    if data_type == DataType.TFRECORD:
        return _get_tf_dataset(data_dir, train_mode, epochs, batch_size,
                               line_per_sample, rank_size=rank_size, rank_id=rank_id)
    return _get_mindrecord_dataset(data_dir, train_mode, epochs,
                                   batch_size, line_per_sample,
                                   rank_size, rank_id)
