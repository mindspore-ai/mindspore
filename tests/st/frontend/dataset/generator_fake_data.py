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
"""GeneratorFakeData."""
import numpy as np


class FakeDataInitMode:
    RandomInit = 0
    OnesInit = 1
    UniqueInit = 2
    ZerosInit = 3


class GeneratorFakeData:
    """A fake dataset that returns randomly generated images and returns them as PIL images, can be
       applied to dataset_sink_mode=True.
       image data type is np.float32 in default
       label data type is np.float32 in default
       label data is onehot in default

    Args:
        size (int, optional): size of the dataset. Default: 1024 images
        batch_size (int, optional): how many samples per batch to load. Default: 32 images
        image_size(tuple, optional): size if the returned images. Default: (3, 224, 224)
        num_classes(int, optional): number of classes in the dataset. Default: 10
        random_offset (int): offsets the index-based random seed used to
            generate each image. Default: 0

    Examples:
        fake_dataset = GeneratorFakeData()
        ds_train = ds.GeneratorDataset(fake_dataset, ["data", "label"])
        ...
        model = Model(net, loss, opt)
        model.train(epoch_num, ds_train)
    """

    def __init__(self, size=1024, batch_size=32, image_size=(3, 224, 224),
                 num_classes=10, random_offset=0,
                 fakedata_mode=FakeDataInitMode.OnesInit, dtype=np.float32):
        self.size = size
        self.rank_batch_size = batch_size
        self.total_batch_size = self.rank_batch_size
        self.random_offset = random_offset
        self.image_size = image_size
        self.num_classes = num_classes
        self.rank_size = 1
        self.rank_id = 0
        self.batch_index = 0
        self.image_data_type = dtype
        self.label_data_type = dtype
        self.is_onehot = True
        self.fakedata_mode = fakedata_mode

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

    def __next__(self):
        batch_index = self.batch_index
        self.batch_index += 1
        if batch_index * self.total_batch_size >= self.size:
            raise StopIteration

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
            img = np.reshape(np.arange(total_size) * 0.0001, self.total_batch_data_size)
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
        return img_ret, target_ret

    def __len__(self):
        return self.size // self.total_batch_size

    def __iter__(self):
        self.batch_index = 0
        return self

    def reset(self):
        self.batch_index = 0
