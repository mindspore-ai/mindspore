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
"""Dataset loader."""

from functools import partial

import numpy as np
from mindspore.dataset import GeneratorDataset


def create(data_path, per_item_num_paths, train, users=None, **kwargs):
    """
    Create a dataset for TBNet.

    Args:
        data_path (str): The csv datafile path.
        per_item_num_paths (int): The number of paths per item.
        train (bool): True to create for training with columns:
            'item', 'relation1', 'entity', 'relation2', 'hist_item', 'rating'
            otherwise:
            'user', 'item', 'relation1', 'entity', 'relation2', 'hist_item', 'rating'
        users (Union[list[int], int], optional): Users data to be loaded, if None is provided, all data will be loaded.
        **kwargs (any): Other arguments for GeneratorDataset(), except 'source' and 'column_names'.

    Returns:
        GeneratorDataset, the generator dataset that reads from the csv datafile.
    """
    if isinstance(users, int):
        users = (users,)
    kwargs['source'] = partial(csv_generator, data_path, per_item_num_paths, users, train)

    if train:
        kwargs['column_names'] = ['item', 'relation1', 'entity', 'relation2', 'hist_item', 'rating']
    else:
        kwargs['column_names'] = ['user', 'item', 'relation1', 'entity', 'relation2', 'hist_item', 'rating']
    return GeneratorDataset(**kwargs)


def csv_generator(csv_path, per_item_num_paths, users, train):
    """Generator for csv datafile."""
    expected_columns = 3 + (per_item_num_paths * 4)
    file = open(csv_path)
    for line in file:
        line = line.strip()
        if not line or line[0] == '#':
            continue
        id_list = line.split(',')
        if len(id_list) < expected_columns:
            raise ValueError(f'Expecting {expected_columns} values but got {len(id_list)} only!')
        id_list = list(map(int, id_list))
        user = id_list[0]
        if users and user not in users:
            continue
        item = id_list[1]
        rating = id_list[2]

        relation1 = np.empty(shape=(per_item_num_paths,), dtype=np.int)
        entity = np.empty_like(relation1)
        relation2 = np.empty_like(relation1)
        hist_item = np.empty_like(relation1)

        for p in range(per_item_num_paths):
            offset = 3 + (p * 4)
            relation1[p] = id_list[offset]
            entity[p] = id_list[offset + 1]
            relation2[p] = id_list[offset + 2]
            hist_item[p] = id_list[offset + 3]

        if train:
            # item, relation1, entity, relation2, hist_item, rating
            yield np.array(item, dtype=np.int), relation1, entity, relation2, hist_item, \
                np.array(rating, dtype=np.float32)
        else:
            # user, item, relation1, entity, relation2, hist_item, rating
            yield np.array(user, dtype=np.int), np.array(item, dtype=np.int),\
                relation1, entity, relation2, hist_item, np.array(rating, dtype=np.float32)
