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
process dataset.
"""
import math
import numpy as np
import pandas as pd
import mindspore.dataset as ds

class STGCNDataset:
    """ BRDNetDataset.
    Args:
        mode: 0 means train;1 means val;2 means test

    """
    def __init__(self, data_path, n_his, n_pred, zscore, mode=0):

        self.df = pd.read_csv(data_path, header=None)
        self.data_col = self.df.shape[0]
        # recommended dataset split rate as train: val: test = 60: 20: 20, 70: 15: 15 or 80: 10: 10
        # using dataset split rate as train: val: test = 70: 15: 15
        self.val_and_test_rate = 0.15

        self.len_val = int(math.floor(self.data_col * self.val_and_test_rate))
        self.len_test = int(math.floor(self.data_col * self.val_and_test_rate))
        self.len_train = int(self.data_col - self.len_val - self.len_test)

        self.dataset_train = self.df[: self.len_train]
        self.dataset_val = self.df[self.len_train: self.len_train + self.len_val]
        self.dataset_test = self.df[self.len_train + self.len_val:]

        self.dataset_train = zscore.fit_transform(self.dataset_train)
        self.dataset_val = zscore.transform(self.dataset_val)
        self.dataset_test = zscore.transform(self.dataset_test)

        if mode == 0:
            self.dataset = self.dataset_train
        elif mode == 1:
            self.dataset = self.dataset_val
        else:
            self.dataset = self.dataset_test

        self.n_his = n_his
        self.n_pred = n_pred
        self.n_vertex = self.dataset.shape[1]
        self.len_record = len(self.dataset)
        self.num = self.len_record - self.n_his - self.n_pred

        self.x = np.zeros([self.num, 1, self.n_his, self.n_vertex], np.float32)
        self.y = np.zeros([self.num, self.n_vertex], np.float32)

        for i in range(self.num):
            head = i
            tail = i + self.n_his
            self.x[i, :, :, :] = self.dataset[head: tail].reshape(1, self.n_his, self.n_vertex)
            self.y[i] = self.dataset[tail + self.n_pred - 1]


    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            x[index]: input of network
            y[index]: label of network
        """

        return self.x[index], self.y[index]

    def __len__(self):
        return self.num


def load_weighted_adjacency_matrix(file_path):
    df = pd.read_csv(file_path, header=None)
    return df.to_numpy()


def create_dataset(data_path, batch_size, n_his, n_pred, zscore, is_sigle, device_num=1, device_id=0, mode=0):
    """
    generate dataset for train or test.
    """
    data = STGCNDataset(data_path, n_his, n_pred, zscore, mode=mode)
    shuffle = True
    if mode != 0:
        shuffle = False
    if not is_sigle:
        dataset = ds.GeneratorDataset(data, column_names=["inputs", "labels"], num_parallel_workers=32, \
         shuffle=shuffle, num_shards=device_num, shard_id=device_id)
    else:
        dataset = ds.GeneratorDataset(data, column_names=["inputs", "labels"], num_parallel_workers=32, shuffle=shuffle)
    dataset = dataset.batch(batch_size)
    return dataset
