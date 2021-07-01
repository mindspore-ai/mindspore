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
# ===========================================================================
"""Get dataset"""
import os

import mindspore.dataset as ds
import numpy as np


def get_mindrecord_dataset(directory, train_mode=True, epochs=1, batch_size=1000,
                           rank_size=None, rank_id=None, line_per_sample=1000):
    """Get Mindrecord dataset"""
    file_prefix_name = 'train_input_part.mindrecord' if train_mode else 'test_input_part.mindrecord'
    file_suffix_name = '00' if train_mode else '0'
    shuffle = train_mode
    if rank_size is not None and rank_id is not None:
        data_set = ds.MindDataset(os.path.join(directory, file_prefix_name + file_suffix_name),
                                  columns_list=['cats_vals', 'num_vals', 'label'],
                                  num_shards=rank_size, shard_id=rank_id, shuffle=shuffle,
                                  num_parallel_workers=8)
    else:
        data_set = ds.MindDataset(os.path.join(directory, file_prefix_name + file_suffix_name),
                                  columns_list=['cats_vals', 'num_vals', 'label'],
                                  shuffle=shuffle, num_parallel_workers=8)
    data_set = data_set.batch(int(batch_size / line_per_sample), drop_remainder=True)
    data_set = data_set.map(operations=(lambda x, y, z: (np.array(x).flatten().reshape(batch_size, 26),
                                                         np.array(y).flatten().reshape(batch_size, 13),
                                                         np.array(z).flatten().reshape(batch_size, 1))),
                            input_columns=['cats_vals', 'num_vals', 'label'],
                            column_order=['cats_vals', 'num_vals', 'label'],
                            num_parallel_workers=8)
    data_set = data_set.repeat(epochs)
    return data_set
