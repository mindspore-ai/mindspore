# Copyright 2022 Huawei Technologies Co., Ltd
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
# See the License foNtest_resr the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""
Test Gtzan dataset operations.
"""
import numpy as np
import pytest

import mindspore.dataset as ds
from mindspore import log as logger

DATA_DIR = "../data/dataset/testGTZANData"


def test_gtzan_basic():
    """
    Feature: GTZANDataset
    Description: Test basic usage of GTZAN
    Expectation: The dataset is as expected
    """
    logger.info("Test GTZANDataset Op")

    # case 1: test loading whole dataset.
    data1 = ds.GTZANDataset(DATA_DIR)
    num_iter1 = 0
    for _ in data1.create_dict_iterator(output_numpy=True, num_epochs=1):
        num_iter1 += 1
    assert num_iter1 == 3

    # case 2: test num_samples.
    data2 = ds.GTZANDataset(DATA_DIR, num_samples=2)
    num_iter2 = 0
    for _ in data2.create_dict_iterator(output_numpy=True, num_epochs=1):
        num_iter2 += 1
    assert num_iter2 == 2

    # case 3: test repeat.
    data3 = ds.GTZANDataset(DATA_DIR, num_samples=2)
    data3 = data3.repeat(5)
    num_iter3 = 0
    for _ in data3.create_dict_iterator(output_numpy=True, num_epochs=1):
        num_iter3 += 1
    assert num_iter3 == 10

    # case 4: test batch with drop_remainder=False.
    data4 = ds.GTZANDataset(DATA_DIR, num_samples=3)
    assert data4.get_dataset_size() == 3
    assert data4.get_batch_size() == 1
    data4 = data4.batch(batch_size=2)  # drop_remainder is default to be False.
    assert data4.get_dataset_size() == 2
    assert data4.get_batch_size() == 2

    # case 5: test batch with drop_remainder=True.
    data5 = ds.GTZANDataset(DATA_DIR, num_samples=3)
    assert data5.get_dataset_size() == 3
    assert data5.get_batch_size() == 1
    # the rest of incomplete batch will be dropped.
    data5 = data5.batch(batch_size=2, drop_remainder=True)
    assert data5.get_dataset_size() == 1
    assert data5.get_batch_size() == 2


def test_gtzan_distribute_sampler():
    """
    Feature: GTZANDataset
    Description: Test GTZAN dataset with DistributedSampler
    Expectation: The results are as expected
    """
    logger.info("Test GTZAN with DistributedSampler")

    label_list1, label_list2 = [], []
    num_shards = 3
    shard_id = 0

    data1 = ds.GTZANDataset(DATA_DIR, usage="all", num_shards=num_shards, shard_id=shard_id)
    count = 0
    for item1 in data1.create_dict_iterator(output_numpy=True, num_epochs=1):
        label_list1.append(item1["label"])
        count = count + 1
    assert count == 1

    num_shards = 3
    shard_id = 0
    sampler = ds.DistributedSampler(num_shards, shard_id)
    data2 = ds.GTZANDataset(DATA_DIR, usage="all", sampler=sampler)
    count = 0
    for item2 in data2.create_dict_iterator(output_numpy=True, num_epochs=1):
        label_list2.append(item2["label"])
        count = count + 1
    np.testing.assert_array_equal(label_list1, label_list2)
    assert count == 1


def test_gtzan_exception():
    """
    Feature: GTZANDataset
    Description: Test error cases for GTZANDataset
    Expectation: The results are as expected
    """
    logger.info("Test error cases for GTZANDataset")
    error_msg_1 = "sampler and shuffle cannot be specified at the same time"
    with pytest.raises(RuntimeError, match=error_msg_1):
        ds.GTZANDataset(DATA_DIR, shuffle=False, sampler=ds.PKSampler(3))

    error_msg_2 = "sampler and sharding cannot be specified at the same time"
    with pytest.raises(RuntimeError, match=error_msg_2):
        ds.GTZANDataset(DATA_DIR, sampler=ds.PKSampler(3),
                        num_shards=2, shard_id=0)

    error_msg_3 = "num_shards is specified and currently requires shard_id as well"
    with pytest.raises(RuntimeError, match=error_msg_3):
        ds.GTZANDataset(DATA_DIR, num_shards=10)

    error_msg_4 = "shard_id is specified but num_shards is not"
    with pytest.raises(RuntimeError, match=error_msg_4):
        ds.GTZANDataset(DATA_DIR, shard_id=0)

    error_msg_5 = "Input shard_id is not within the required interval"
    with pytest.raises(ValueError, match=error_msg_5):
        ds.GTZANDataset(DATA_DIR, num_shards=5, shard_id=-1)
    with pytest.raises(ValueError, match=error_msg_5):
        ds.GTZANDataset(DATA_DIR, num_shards=5, shard_id=5)
    with pytest.raises(ValueError, match=error_msg_5):
        ds.GTZANDataset(DATA_DIR, num_shards=2, shard_id=5)

    error_msg_6 = "num_parallel_workers exceeds"
    with pytest.raises(ValueError, match=error_msg_6):
        ds.GTZANDataset(DATA_DIR, shuffle=False, num_parallel_workers=0)
    with pytest.raises(ValueError, match=error_msg_6):
        ds.GTZANDataset(DATA_DIR, shuffle=False, num_parallel_workers=256)
    with pytest.raises(ValueError, match=error_msg_6):
        ds.GTZANDataset(DATA_DIR, shuffle=False, num_parallel_workers=-2)

    error_msg_7 = "Argument shard_id"
    with pytest.raises(TypeError, match=error_msg_7):
        ds.GTZANDataset(DATA_DIR, num_shards=2, shard_id="0")

    def exception_func(item):
        raise Exception("Error occur!")

    error_msg_8 = "The corresponding data file is"

    with pytest.raises(RuntimeError, match=error_msg_8):
        data = ds.GTZANDataset(DATA_DIR)
        data = data.map(operations=exception_func, input_columns=["waveform"], num_parallel_workers=1)
        for _ in data.create_dict_iterator(output_numpy=True, num_epochs=1):
            pass


def test_gtzan_sequential_sampler():
    """
    Feature: GTZANDataset
    Description: Test GTZANDataset with SequentialSampler
    Expectation: The results are as expected
    """
    logger.info("Test GTZANDataset Op with SequentialSampler")
    num_samples = 2
    sampler = ds.SequentialSampler(num_samples=num_samples)
    data1 = ds.GTZANDataset(DATA_DIR, sampler=sampler)
    data2 = ds.GTZANDataset(DATA_DIR, shuffle=False, num_samples=num_samples)
    label_list1, label_list2 = [], []
    num_iter = 0
    for item1, item2 in zip(data1.create_dict_iterator(output_numpy=True, num_epochs=1),
                            data2.create_dict_iterator(output_numpy=True, num_epochs=1)):
        label_list1.append(item1["label"])
        label_list2.append(item2["label"])
        num_iter += 1
    np.testing.assert_array_equal(label_list1, label_list2)
    assert num_iter == num_samples


def test_gtzan_usage():
    """
    Feature: GTZANDataset
    Description: Test GTZANDataset usage
    Expectation: The results are as expected
    """
    logger.info("Test GTZANDataset usage")

    def test_config(usage, gtzan_path=None):
        gtzan_path = DATA_DIR if gtzan_path is None else gtzan_path
        try:
            data = ds.GTZANDataset(gtzan_path, usage=usage, shuffle=False)
            num_rows = 0
            for _ in data.create_dict_iterator(num_epochs=1, output_numpy=True):
                num_rows += 1
        except (ValueError, TypeError, RuntimeError) as e:
            return str(e)
        return num_rows

    assert test_config("valid") == 3
    assert test_config("all") == 3
    assert "usage is not within the valid set of ['train', 'valid', 'test', 'all']" in test_config("invalid")
    assert "Argument usage with value ['list'] is not of type [<class 'str'>]" in test_config(["list"])

    # change this directory to the folder that contains all gtzan files.
    all_files_path = None
    # the following tests on the entire datasets.
    if all_files_path is not None:
        assert test_config("train", all_files_path) == 3
        assert test_config("valid", all_files_path) == 3
        assert ds.GTZANDataset(all_files_path, usage="train").get_dataset_size() == 3
        assert ds.GTZANDataset(all_files_path, usage="valid").get_dataset_size() == 3


if __name__ == '__main__':
    test_gtzan_basic()
    test_gtzan_distribute_sampler()
    test_gtzan_exception()
    test_gtzan_sequential_sampler()
    test_gtzan_usage()
