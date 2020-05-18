# Copyright 2019 Huawei Technologies Co., Ltd
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
# ==============================================================================

import numpy as np
import time

import mindspore.dataset as ds
from mindspore import log as logger


def gen():
    for i in range(100):
        yield np.array(i),


class Augment:
    def __init__(self, loss):
        self.loss = loss

    def preprocess(self, input):
        return input

    def update(self, data):
        self.loss = data["loss"]


def test_simple_sync_wait():
    """
    Test simple sync wait: test sync in dataset pipeline  
    """
    logger.info("test_simple_sync_wait")
    batch_size = 4
    dataset = ds.GeneratorDataset(gen, column_names=["input"])

    aug = Augment(0)
    dataset = dataset.sync_wait(condition_name="policy", callback=aug.update)
    dataset = dataset.map(input_columns=["input"], operations=[aug.preprocess])
    dataset = dataset.batch(batch_size)

    count = 0
    for data in dataset.create_dict_iterator():
        assert (data["input"][0] == count)
        count += batch_size
        data = {"loss": count}
        dataset.sync_update(condition_name="policy", data=data)


def test_simple_shuffle_sync():
    """
    Test simple shuffle sync: test shuffle before sync  
    """
    logger.info("test_simple_shuffle_sync")
    shuffle_size = 4
    batch_size = 10

    dataset = ds.GeneratorDataset(gen, column_names=["input"])

    aug = Augment(0)
    dataset = dataset.shuffle(shuffle_size)
    dataset = dataset.sync_wait(condition_name="policy", callback=aug.update)
    dataset = dataset.map(input_columns=["input"], operations=[aug.preprocess])
    dataset = dataset.batch(batch_size)

    count = 0
    for data in dataset.create_dict_iterator():
        count += 1
        # time.sleep(0.5)
        data = {"loss": count}
        dataset.sync_update(condition_name="policy", data=data)


def test_two_sync():
    """
    Test two sync: dataset pipeline with with two sync_operators  
    """
    logger.info("test_two_sync")
    batch_size = 6

    dataset = ds.GeneratorDataset(gen, column_names=["input"])

    aug = Augment(0)
    # notice that with our design, we need to have step_size = shuffle size
    dataset = dataset.sync_wait(condition_name="every batch", callback=aug.update)

    dataset = dataset.map(input_columns=["input"], operations=[aug.preprocess])

    dataset = dataset.sync_wait(num_batch=2, condition_name="every 2 batches")

    dataset = dataset.batch(batch_size)

    count = 0
    for data in dataset.create_dict_iterator():
        count += 1
        data = {"loss": count}
        dataset.sync_update(condition_name="every batch", data=data)
        if count % 2 == 0:
            dataset.sync_update(condition_name="every 2 batches")


def test_sync_epoch():
    """
    Test sync wait with epochs: test sync with epochs in dataset pipeline  
    """
    logger.info("test_sync_epoch")
    batch_size = 30
    dataset = ds.GeneratorDataset(gen, column_names=["input"])

    aug = Augment(0)
    dataset = dataset.sync_wait(condition_name="policy", callback=aug.update)
    dataset = dataset.map(input_columns=["input"], operations=[aug.preprocess])
    dataset = dataset.batch(batch_size, drop_remainder=True)

    for epochs in range(3):
        aug.update({"loss": 0})
        count = 0
        for data in dataset.create_dict_iterator():
            assert (data["input"][0] == count)
            count += batch_size
            data = {"loss": count}
            dataset.sync_update(condition_name="policy", data=data)


def test_multiple_iterators():
    """
    Test sync wait with multiple iterators: will start multiple 
    """
    logger.info("test_sync_epoch")
    batch_size = 30
    dataset = ds.GeneratorDataset(gen, column_names=["input"])

    aug = Augment(0)
    dataset = dataset.sync_wait(condition_name="policy", callback=aug.update)
    dataset = dataset.map(input_columns=["input"], operations=[aug.preprocess])
    dataset = dataset.batch(batch_size, drop_remainder=True)
    # 2nd dataset 
    dataset2 = ds.GeneratorDataset(gen, column_names=["input"])

    aug = Augment(0)
    dataset2 = dataset2.sync_wait(condition_name="policy", callback=aug.update)
    dataset2 = dataset2.map(input_columns=["input"], operations=[aug.preprocess])
    dataset2 = dataset2.batch(batch_size, drop_remainder=True)

    for item1, item2 in zip(dataset.create_dict_iterator(), dataset2.create_dict_iterator()):
        assert (item1["input"][0] == item2["input"][0])
        data1 = {"loss": item1["input"][0]}
        data2 = {"loss": item2["input"][0]}
        dataset.sync_update(condition_name="policy", data=data1)
        dataset2.sync_update(condition_name="policy", data=data2)


def test_sync_exception_01():
    """
    Test sync: with shuffle in sync mode 
    """
    logger.info("test_sync_exception_01")
    shuffle_size = 4
    batch_size = 10

    dataset = ds.GeneratorDataset(gen, column_names=["input"])

    aug = Augment(0)
    dataset = dataset.sync_wait(condition_name="policy", callback=aug.update)
    dataset = dataset.map(input_columns=["input"], operations=[aug.preprocess])

    try:
        dataset = dataset.shuffle(shuffle_size)
    except BaseException as e:
        assert "shuffle" in str(e)
    dataset = dataset.batch(batch_size)


def test_sync_exception_02():
    """
    Test sync: with duplicated condition name  
    """
    logger.info("test_sync_exception_02")
    batch_size = 6

    dataset = ds.GeneratorDataset(gen, column_names=["input"])

    aug = Augment(0)
    # notice that with our design, we need to have step_size = shuffle size
    dataset = dataset.sync_wait(condition_name="every batch", callback=aug.update)

    dataset = dataset.map(input_columns=["input"], operations=[aug.preprocess])

    try:
        dataset = dataset.sync_wait(num_batch=2, condition_name="every batch")
    except BaseException as e:
        assert "name" in str(e)
    dataset = dataset.batch(batch_size)


if __name__ == "__main__":
    test_simple_sync_wait()
    test_simple_shuffle_sync()
    test_two_sync()
    test_sync_exception_01()
    test_sync_exception_02()
    test_sync_epoch()
    test_multiple_iterators()
