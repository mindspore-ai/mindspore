# Copyright 2019-2022 Huawei Technologies Co., Ltd

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
import pytest
import mindspore.dataset as ds
from mindspore import log as logger


def gen():
    for i in range(100):
        yield (np.array(i),)


class Augment:
    def __init__(self, loss):
        self.loss = loss

    def preprocess(self, input_):
        return input_

    def update(self, data):
        self.loss = data["loss"]


def test_simple_sync_wait():
    """
    Feature: Sync_wait op
    Description: Test sync_wait op in dataset pipeline
    Expectation: Runs successfully
    """
    logger.info("test_simple_sync_wait")
    batch_size = 4
    dataset = ds.GeneratorDataset(gen, column_names=["input"])

    aug = Augment(0)
    dataset = dataset.sync_wait(condition_name="policy", callback=aug.update)
    dataset = dataset.map(operations=[aug.preprocess], input_columns=["input"])
    dataset = dataset.batch(batch_size)
    count = 0
    for data in dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
        assert data["input"][0] == count
        count += batch_size
        data = {"loss": count}
        dataset.sync_update(condition_name="policy", data=data)


def test_simple_shuffle_sync():
    """
    Feature: Sync_wait op
    Description: Test sync_wait op after shuffle op
    Expectation: Runs successfully
    """
    logger.info("test_simple_shuffle_sync")
    shuffle_size = 4
    batch_size = 10

    dataset = ds.GeneratorDataset(gen, column_names=["input"])

    aug = Augment(0)
    dataset = dataset.shuffle(shuffle_size)
    dataset = dataset.sync_wait(condition_name="policy", callback=aug.update)
    dataset = dataset.map(operations=[aug.preprocess], input_columns=["input"])
    dataset = dataset.batch(batch_size)

    count = 0
    for data in dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
        count += 1
        data = {"loss": count}
        dataset.sync_update(condition_name="policy", data=data)


def test_two_sync():
    """
    Feature: Sync_wait op
    Description: Test sync_wait op in dataset pipeline with two sync_wait ops
    Expectation: Runs successfully
    """
    logger.info("test_two_sync")
    batch_size = 6

    dataset = ds.GeneratorDataset(gen, column_names=["input"])

    aug = Augment(0)
    # notice that with our design, we need to have step_size = shuffle size
    dataset = dataset.sync_wait(condition_name="every batch", callback=aug.update)

    dataset = dataset.map(operations=[aug.preprocess], input_columns=["input"])

    dataset = dataset.sync_wait(num_batch=2, condition_name="every 2 batches")

    dataset = dataset.batch(batch_size)

    count = 0
    for data in dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
        count += 1
        data = {"loss": count}
        dataset.sync_update(condition_name="every batch", data=data)
        if count % 2 == 0:
            dataset.sync_update(condition_name="every 2 batches")


def test_sync_epoch():
    """
    Feature: Sync_wait op
    Description: Test sync_wait op with epochs in dataset pipeline
    Expectation: Runs successfully
    """
    logger.info("test_sync_epoch")
    batch_size = 30
    dataset = ds.GeneratorDataset(gen, column_names=["input"])

    aug = Augment(0)
    dataset = dataset.sync_wait(condition_name="policy", callback=aug.update)
    dataset = dataset.map(operations=[aug.preprocess], input_columns=["input"])
    dataset = dataset.batch(batch_size, drop_remainder=True)

    for _ in range(3):
        aug.update({"loss": 0})
        count = 0
        for data in dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
            assert data["input"][0] == count
            count += batch_size
            data = {"loss": count}
            dataset.sync_update(condition_name="policy", data=data)


def test_multiple_iterators():
    """
    Feature: Sync_wait op
    Description: Test sync_wait op with multiple iterators
    Expectation: Runs successfully
    """
    logger.info("test_sync_epoch")
    batch_size = 30
    dataset = ds.GeneratorDataset(gen, column_names=["input"])

    aug = Augment(0)
    dataset = dataset.sync_wait(condition_name="policy", callback=aug.update)
    dataset = dataset.map(operations=[aug.preprocess], input_columns=["input"])
    dataset = dataset.batch(batch_size, drop_remainder=True)
    # 2nd dataset
    dataset2 = ds.GeneratorDataset(gen, column_names=["input"])

    aug = Augment(0)
    dataset2 = dataset2.sync_wait(condition_name="policy", callback=aug.update)
    dataset2 = dataset2.map(operations=[aug.preprocess], input_columns=["input"])
    dataset2 = dataset2.batch(batch_size, drop_remainder=True)

    for item1, item2 in zip(dataset.create_dict_iterator(num_epochs=1, output_numpy=True),
                            dataset2.create_dict_iterator(num_epochs=1, output_numpy=True)):
        assert item1["input"][0] == item2["input"][0]
        data1 = {"loss": item1["input"][0]}
        data2 = {"loss": item2["input"][0]}
        dataset.sync_update(condition_name="policy", data=data1)
        dataset2.sync_update(condition_name="policy", data=data2)


def test_sync_exception_01():
    """
    Feature: Sync_wait op
    Description: Test sync_wait op followed by shuffle op
    Expectation: Error is raised as expected
    """
    logger.info("test_sync_exception_01")
    shuffle_size = 4

    dataset = ds.GeneratorDataset(gen, column_names=["input"])

    aug = Augment(0)
    dataset = dataset.sync_wait(condition_name="policy", callback=aug.update)
    dataset = dataset.map(operations=[aug.preprocess], input_columns=["input"])

    with pytest.raises(RuntimeError) as e:
        dataset.shuffle(shuffle_size)
    assert "No shuffle after sync operators" in str(e.value)


def test_sync_exception_02():
    """
    Feature: Sync_wait op
    Description: Test sync_wait op with duplicated condition name
    Expectation: Error is raised as expected
    """
    logger.info("test_sync_exception_02")

    dataset = ds.GeneratorDataset(gen, column_names=["input"])

    aug = Augment(0)
    dataset = dataset.sync_wait(condition_name="every batch", callback=aug.update)

    dataset = dataset.map(operations=[aug.preprocess], input_columns=["input"])

    with pytest.raises(RuntimeError) as e:
        dataset.sync_wait(num_batch=2, condition_name="every batch")
    assert "Condition name is already in use" in str(e.value)


def test_sync_exception_03():
    """
    Feature: Sync_wait op
    Description: Test sync_wait op with wrong batch size
    Expectation: Error is raised as expected
    """
    logger.info("test_sync_exception_03")

    dataset = ds.GeneratorDataset(gen, column_names=["input"])

    aug = Augment(0)
    # try to create dataset with batch_size < 0
    with pytest.raises(ValueError) as e:
        dataset.sync_wait(condition_name="every batch", num_batch=-1, callback=aug.update)
    assert "num_batch need to be greater than 0." in str(e.value)


def test_sync_exception_04():
    """
    Feature: Sync_wait op
    Description: Test sync_wait op with negative batch size in update
    Expectation: Error is raised as expected
    """
    logger.info("test_sync_exception_04")

    dataset = ds.GeneratorDataset(gen, column_names=["input"])

    aug = Augment(0)
    # try to create dataset with batch_size < 0
    dataset = dataset.sync_wait(condition_name="every batch", callback=aug.update)
    dataset = dataset.map(operations=[aug.preprocess], input_columns=["input"])
    count = 0
    with pytest.raises(RuntimeError) as e:
        for _ in dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
            count += 1
            data = {"loss": count}
            dataset.sync_update(condition_name="every batch", num_batch=-1, data=data)
    assert "Sync_update batch size can only be positive" in str(e.value)


def test_sync_exception_05():
    """
    Feature: Sync_wait op
    Description: Test sync_wait op with wrong condition name in update
    Expectation: Error is raised as expected
    """
    logger.info("test_sync_exception_05")

    dataset = ds.GeneratorDataset(gen, column_names=["input"])
    count = 0
    aug = Augment(0)
    dataset = dataset.sync_wait(condition_name="every batch", callback=aug.update)
    dataset = dataset.map(operations=[aug.preprocess], input_columns=["input"])
    with pytest.raises(RuntimeError) as e:
        for _ in dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
            dataset.disable_sync()
            count += 1
            data = {"loss": count}
            dataset.disable_sync()
            dataset.sync_update(condition_name="every", data=data)
    assert "Condition name not found" in str(e.value)


def test_simple_sync_wait_empty_condition_name():
    """
    Feature: Sync_wait op
    Description: Test where callback is none, and sync_wait and sync_update condition_name is empty string ('')
    Expectation: Runs successfully
    """
    logger.info("test_simple_sync_wait_empty_condition_name")
    batch_size = 10
    dataset = ds.GeneratorDataset(gen, column_names=["input"])

    aug = Augment(0)
    dataset = dataset.sync_wait(condition_name='', num_batch=1)
    dataset = dataset.map(input_columns=["input"], operations=[aug.preprocess])
    dataset = dataset.batch(batch_size)

    count = 0
    for data in dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
        count += 1
        data = {"loss": count}
        dataset.sync_update(condition_name="", data=data)


def test_sync_exception_06():
    """
    Feature: Sync_wait op
    Description: Test sync_wait op with string batch size
    Expectation: Error is raised as expected
    """
    logger.info("test_sync_exception_03")

    dataset = ds.GeneratorDataset(gen, column_names=["input"])

    aug = Augment(0)
    # try to create dataset with batch_size < 0
    with pytest.raises(TypeError) as e:
        dataset.sync_wait(condition_name="every batch", num_batch="123", callback=aug.update)
    assert "is not of type [<class 'int'>]" in str(e.value)


if __name__ == "__main__":
    test_simple_sync_wait()
    test_simple_shuffle_sync()
    test_two_sync()
    test_sync_exception_01()
    test_sync_exception_02()
    test_sync_exception_03()
    test_sync_exception_04()
    test_sync_exception_05()
    test_sync_exception_06()
    test_sync_epoch()
    test_multiple_iterators()
    test_simple_sync_wait_empty_condition_name()
