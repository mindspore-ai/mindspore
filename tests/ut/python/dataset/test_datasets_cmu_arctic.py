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
Test CMUArctic dataset operations
"""
import numpy as np
import pytest

import mindspore.dataset as ds
from mindspore import log as logger


DATA_DIR = "../data/dataset/testCMUArcticData"


def test_cmu_arctic_basic():
    """
    Feature: CMUArcticDataset
    Description: Test basic name of CMUArctic
    Expectation: The dataset is as expected
    """
    logger.info("Test CMUArcticDataset Op")

    # case 1: test loading fault dataset.
    data1 = ds.CMUArcticDataset(DATA_DIR)
    num_iter1 = 0
    for _ in data1.create_dict_iterator(output_numpy=True, num_epochs=1):
        num_iter1 += 1
    assert num_iter1 == 3

    # case 2: test num_samples.
    data2 = ds.CMUArcticDataset(DATA_DIR, num_samples=1)
    num_iter2 = 0
    for _ in data2.create_dict_iterator(output_numpy=True, num_epochs=1):
        num_iter2 += 1
    assert num_iter2 == 1

    # case 3: test repeat.
    data3 = ds.CMUArcticDataset(DATA_DIR, name="aew", num_samples=3)
    data3 = data3.repeat(3)
    num_iter3 = 0
    for _ in data3.create_dict_iterator(output_numpy=True, num_epochs=1):
        num_iter3 += 1
    assert num_iter3 == 9

    # case 4: test batch with drop_remainder=False.
    data4 = ds.CMUArcticDataset(DATA_DIR, name="aew", num_samples=3)
    assert data4.get_dataset_size() == 3
    assert data4.get_batch_size() == 1
    data4 = data4.batch(batch_size=2)  # drop_remainder is default to be False.
    assert data4.get_dataset_size() == 2
    assert data4.get_batch_size() == 2

    # case 5: test batch with drop_remainder=True.
    data5 = ds.CMUArcticDataset(DATA_DIR, name="aew", num_samples=3)
    assert data5.get_dataset_size() == 3
    assert data5.get_batch_size() == 1
    # the rest of incomplete batch will be dropped.
    data5 = data5.batch(batch_size=2, drop_remainder=True)
    assert data5.get_dataset_size() == 1
    assert data5.get_batch_size() == 2


def test_cmu_arctic_distribute_sampler():
    """
    Feature: CMUArcticDataset
    Description: Test CMUArctic dataset with DistributedSampler
    Expectation: The results are as expected
    """
    logger.info("Test CMUArctic with sharding")

    num_shards = 3
    shard_id = 0

    data1 = ds.CMUArcticDataset(DATA_DIR, name="aew", num_shards=num_shards, shard_id=shard_id)
    count = 0
    for _ in data1.create_dict_iterator(output_numpy=True, num_epochs=1):
        count = count + 1
    assert count == 1

    num_shards = 3
    shard_id = 0
    sampler = ds.DistributedSampler(num_shards, shard_id)
    data2 = ds.CMUArcticDataset(DATA_DIR, name="aew", sampler=sampler)
    count = 0
    for _ in data2.create_dict_iterator(output_numpy=True, num_epochs=1):
        count = count + 1
    assert count == 1


def test_cmu_arctic_exception():
    """
    Feature: CMUArcticDataset
    Description: Test error cases for CMUArcticDataset
    Expectation: The results are as expected
    """
    logger.info("Test error cases for CMUArcticDataset")

    error_msg_1 = "sampler and shuffle cannot be specified at the same time"
    with pytest.raises(RuntimeError, match=error_msg_1):
        ds.CMUArcticDataset(DATA_DIR, shuffle=False, sampler=ds.PKSampler(3))

    error_msg_2 = "sampler and sharding cannot be specified at the same time"
    with pytest.raises(RuntimeError, match=error_msg_2):
        ds.CMUArcticDataset(DATA_DIR, sampler=ds.PKSampler(3), num_shards=2, shard_id=0)

    error_msg_3 = "num_shards is specified and currently requires shard_id as well"
    with pytest.raises(RuntimeError, match=error_msg_3):
        ds.CMUArcticDataset(DATA_DIR, num_shards=10)

    error_msg_4 = "shard_id is specified but num_shards is not"
    with pytest.raises(RuntimeError, match=error_msg_4):
        ds.CMUArcticDataset(DATA_DIR, shard_id=0)

    error_msg_5 = "Input shard_id is not within the required interval"
    with pytest.raises(ValueError, match=error_msg_5):
        ds.CMUArcticDataset(DATA_DIR, num_shards=5, shard_id=-1)
    with pytest.raises(ValueError, match=error_msg_5):
        ds.CMUArcticDataset(DATA_DIR, num_shards=5, shard_id=5)
    with pytest.raises(ValueError, match=error_msg_5):
        ds.CMUArcticDataset(DATA_DIR, num_shards=2, shard_id=5)

    error_msg_6 = "num_parallel_workers exceeds"
    with pytest.raises(ValueError, match=error_msg_6):
        ds.CMUArcticDataset(DATA_DIR, shuffle=False, num_parallel_workers=0)
    with pytest.raises(ValueError, match=error_msg_6):
        ds.CMUArcticDataset(DATA_DIR, shuffle=False, num_parallel_workers=256)
    with pytest.raises(ValueError, match=error_msg_6):
        ds.CMUArcticDataset(DATA_DIR, shuffle=False, num_parallel_workers=-2)

    error_msg_7 = "Argument shard_id"
    with pytest.raises(TypeError, match=error_msg_7):
        ds.CMUArcticDataset(DATA_DIR, num_shards=2, shard_id="0")

    def exception_func(item):
        raise Exception("Error occur!")

    error_msg_8 = "The corresponding data files"
    with pytest.raises(RuntimeError, match=error_msg_8):
        data = ds.CMUArcticDataset(DATA_DIR)
        data = data.map(operations=exception_func, input_columns=["waveform"], num_parallel_workers=1)
        for _ in data.create_dict_iterator(output_numpy=True, num_epochs=1):
            pass


def test_cmu_arctic_sequential_sampler():
    """
    Feature: CMUArcticDataset
    Description: Test CMUArcticDataset with SequentialSampler
    Expectation: The results are as expected
    """
    logger.info("Test CMUArcticDataset Op with SequentialSampler")

    num_samples = 2
    sampler = ds.SequentialSampler(num_samples=num_samples)
    data1 = ds.CMUArcticDataset(DATA_DIR, name="aew", sampler=sampler)
    data2 = ds.CMUArcticDataset(DATA_DIR, name="aew", shuffle=False, num_samples=num_samples)

    utterance_id_expected = ['a0001', 'a0002']
    utterance_id_list1, utterance_id_list2 = [], []

    sample_rate_expected = [16000, 16000]
    sample_rate_list1, sample_rate_list2 = [], []

    transcript_expected = ['Dog.', 'Cat.']
    transcript_list1, transcript_list2 = [], []
    num_iter = 0
    for item1, item2 in zip(data1.create_dict_iterator(output_numpy=True, num_epochs=1),
                            data2.create_dict_iterator(output_numpy=True, num_epochs=1)):
        transcript_list1.append(item1["transcript"])
        transcript_list2.append(item2["transcript"])
        sample_rate_list1.append(item1["sample_rate"])
        sample_rate_list2.append(item2["sample_rate"])
        utterance_id_list1.append(item1["utterance_id"])
        utterance_id_list2.append(item2["utterance_id"])
        num_iter += 1

    np.testing.assert_array_equal(transcript_list1, transcript_expected)
    np.testing.assert_array_equal(transcript_list2, transcript_expected)
    np.testing.assert_array_equal(utterance_id_list1, utterance_id_expected)
    np.testing.assert_array_equal(utterance_id_list2, utterance_id_expected)
    np.testing.assert_array_equal(sample_rate_list1, sample_rate_expected)
    np.testing.assert_array_equal(sample_rate_list2, sample_rate_expected)
    assert num_iter == num_samples


def test_cmu_arctic_name():
    """
    Feature: CMUArcticDataset
    Description: Test CMUArcticDataset name
    Expectation: The results are as expected
    """
    logger.info("Test CMUArcticDataset name")

    def test_config(name, cmu_arctic_path=None):
        cmu_arctic_path = DATA_DIR if cmu_arctic_path is None else cmu_arctic_path
        try:
            data = ds.CMUArcticDataset(cmu_arctic_path, name=name, shuffle=False)
            num_rows = 0
            for _ in data.create_dict_iterator(num_epochs=1, output_numpy=True):
                num_rows += 1
        except (ValueError, TypeError, RuntimeError) as e:
            return str(e)
        return num_rows

    assert test_config("aew") == 3
    assert "Input name is not within the valid set of ['aew', 'ahw', 'aup', 'awb', 'axb', 'bdl', 'clb', 'eey', "\
           "'fem', 'gka', 'jmk', 'ksp', 'ljm', 'lnh', 'rms', 'rxr', 'slp', 'slt']." in test_config("invalid")
    assert "Argument name with value ['list'] is not of type [<class 'str'>]" in test_config(["list"])

    all_files_path = None
    if all_files_path is not None:
        assert test_config("aew", all_files_path) == 3
        assert ds.cmu_arcticDataset(all_files_path, name="aew").get_dataset_size() == 3


if __name__ == '__main__':
    test_cmu_arctic_basic()
    test_cmu_arctic_distribute_sampler()
    test_cmu_arctic_exception()
    test_cmu_arctic_sequential_sampler()
    test_cmu_arctic_name()
