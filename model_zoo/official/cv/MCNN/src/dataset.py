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
Data operations, will be used in train.py and eval.py
"""
import os
import math
import mindspore.dataset as ds


def create_dataset(data_loader, target="Ascend", train=True):
    datalist = []
    labellist = []
    for blob in data_loader:
        datalist.append(blob['data'])
        labellist.append(blob['gt_density'])

    class GetDatasetGenerator:
        def __init__(self):

            self.__data = datalist
            self.__label = labellist

        def __getitem__(self, index):
            return (self.__data[index], self.__label[index])

        def __len__(self):
            return len(self.__data)

    class MySampler():
        def __init__(self, dataset, local_rank, world_size):
            self.__num_data = len(dataset)
            self.__local_rank = local_rank
            self.__world_size = world_size
            self.samples_per_rank = int(math.ceil(self.__num_data / float(self.__world_size)))
            self.total_num_samples = self.samples_per_rank * self.__world_size

        def __iter__(self):
            indices = list(range(self.__num_data))
            indices.extend(indices[:self.total_num_samples-len(indices)])
            indices = indices[self.__local_rank:self.total_num_samples:self.__world_size]
            return iter(indices)

        def __len__(self):
            return self.samples_per_rank

    dataset_generator = GetDatasetGenerator()
    sampler = MySampler(dataset_generator, local_rank=0, world_size=8)

    if target == "Ascend":
        # device_num, rank_id = _get_rank_info()
        device_num = int(os.getenv("RANK_SIZE"))
        rank_id = int(os.getenv("DEVICE_ID"))
        sampler = MySampler(dataset_generator, local_rank=rank_id, world_size=8)
    if target != "Ascend" or device_num == 1 or (not train):
        data_set = ds.GeneratorDataset(dataset_generator, ["data", "gt_density"])
    else:
        data_set = ds.GeneratorDataset(dataset_generator, ["data", "gt_density"], num_parallel_workers=8,
                                       num_shards=device_num, shard_id=rank_id, sampler=sampler)

    return data_set
