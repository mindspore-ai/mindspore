# Copyright 2025 Huawei Technologies Co., Ltd
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
""" fakedata management base module. """
from mindspore import Tensor
from mindspore.communication.management import get_rank
from mindspore.communication.management import get_group_size
import numpy as np
import os


class FakeDataInitMode:
    RandomInit = 0
    OnesInit = 1
    UniqueInit = 2
    ZerosInit = 3


class FakeData:
    def __init__(self, size=1024, batch_size=32, image_size=(3, 224, 224),
                 num_classes=10, random_offset=0, use_parallel=False, rol=0.0001,
                 fakedata_mode=FakeDataInitMode.RandomInit, image_dtype=np.float32,
                 label_dtype=np.float32):
        self.size = size
        self.rank_batch_size = batch_size
        self.total_batch_size = self.rank_batch_size
        self.random_offset = random_offset
        self.image_size = image_size
        self.num_classes = num_classes
        self.rol = rol
        self.rank_size = 1
        self.rank_id = 0
        self.batch_index = 0
        self.image_data_type = image_dtype
        self.label_data_type = label_dtype
        self.is_onehot = True
        self.fakedata_mode = fakedata_mode
        if use_parallel:
            self.rank_size = get_group_size()
            self.rank_id = get_rank()
        if os.getenv("MS_ROLE") == "MS_PSERVER" and use_parallel:
            self.rank_size = int(os.getenv("MS_WORKER_NUM"))
            self.rank_id = 0
        self.total_batch_size = self.rank_batch_size * self.rank_size
        assert self.size % self.total_batch_size == 0
        self.total_batch_data_size = (self.rank_size, self.rank_batch_size) + image_size

    def get_dataset_size(self):
        return int(self.size / self.total_batch_size)

    def get_repeat_count(self):
        return 1

    def set_image_data_type(self, data_type):
        self.image_data_type = data_type

    def set_label_data_type(self, data_type):
        self.label_data_type = data_type

    def set_label_onehot(self, is_onehot=True):
        self.is_onehot = is_onehot

    def create_tuple_iterator(self, num_epochs=-1, do_copy=False):
        return self

    def __getitem__(self, batch_index):
        if batch_index * self.total_batch_size >= len(self):
            raise IndexError("{} index out of range".format(self.__class__.__name__))
        rng_state = np.random.get_state()
        np.random.seed(batch_index + self.random_offset)
        if self.fakedata_mode == FakeDataInitMode.OnesInit:
            img = np.ones(self.total_batch_data_size)
        elif self.fakedata_mode == FakeDataInitMode.ZerosInit:
            img = np.zeros(self.total_batch_data_size)
        elif self.fakedata_mode == FakeDataInitMode.UniqueInit:
            total_size = 1
            for i in self.total_batch_data_size:
                total_size = total_size * i
            img = np.reshape(np.arange(total_size) * self.rol, self.total_batch_data_size)
        else:
            img = np.random.randn(*self.total_batch_data_size)
        target = np.random.randint(0, self.num_classes, size=(self.rank_size, self.rank_batch_size))
        np.random.set_state(rng_state)
        img = img[self.rank_id]
        target = target[self.rank_id]
        img_ret = img.astype(self.image_data_type)
        target_ret = target.astype(self.label_data_type)
        if self.is_onehot:
            target_onehot = np.zeros(shape=(self.rank_batch_size, self.num_classes))
            target_onehot[np.arange(self.rank_batch_size), target] = 1
            target_ret = target_onehot.astype(self.label_data_type)
        return Tensor(img_ret), Tensor(target_ret)

    def __len__(self):
        return self.size

    def __iter__(self):
        self.batch_index = 0
        return self

    def reset(self):
        self.batch_index = 0

    def __next__(self):
        if self.batch_index * self.total_batch_size < len(self):
            data = self[self.batch_index]
            self.batch_index += 1
            return data
        raise StopIteration


class RecommendFakeData(FakeData):
    def __init__(self, img, size=1024, batch_size=32, image_size=(3, 224, 224),
                 num_classes=10, use_parallel=False, pipeline_stages=1):
        super().__init__(size=size, batch_size=batch_size, image_size=image_size,
                         num_classes=num_classes, use_parallel=use_parallel)
        self.img = img
        self.rank_size = self.rank_size // pipeline_stages
        self.total_batch_data_size = (self.rank_size, self.rank_batch_size) + image_size

    def __getitem__(self, batch_index):
        if batch_index * self.total_batch_size >= len(self):
            raise IndexError("{} index out of range".format(self.__class__.__name__))
        rng_state = np.random.get_state()
        np.random.seed(batch_index + self.random_offset)
        img = self.img
        img = img.reshape(*self.total_batch_data_size)
        target = np.random.randint(0, self.num_classes, size=(self.rank_size, self.rank_batch_size))
        np.random.set_state(rng_state)
        index = self.rank_id % 8
        img = img[index]
        target = target[index]
        img_ret = img.astype(self.image_data_type)
        target_ret = target.astype(self.label_data_type)
        if self.is_onehot:
            target_onehot = np.zeros(shape=(self.rank_batch_size, self.num_classes))
            target_onehot[np.arange(self.rank_batch_size), target] = 1
            target_ret = target_onehot.astype(self.label_data_type)
        return Tensor(img_ret), Tensor(target_ret)
