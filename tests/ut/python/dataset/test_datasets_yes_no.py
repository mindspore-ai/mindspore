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
# ==============================================================================
import numpy as np
import pytest

import mindspore.dataset as ds
import mindspore.dataset.audio as audio
from mindspore import log as logger

DATA_DIR = "../data/dataset/testYesNoData/"


def test_yes_no_basic():
    """
    Feature: YesNo Dataset
    Description: Read all files
    Expectation: Output the amount of file
    """
    logger.info("Test YesNoDataset Op")

    data = ds.YesNoDataset(DATA_DIR)
    num_iter = 0
    for _ in data.create_dict_iterator(num_epochs=1):
        num_iter += 1
    assert num_iter == 3


def test_yes_no_num_samples():
    """
    Feature: YesNo Dataset
    Description: Test num_samples
    Expectation: Get certain number of samples
    """
    data = ds.YesNoDataset(DATA_DIR, num_samples=2)
    num_iter = 0
    for _ in data.create_dict_iterator(num_epochs=1):
        num_iter += 1
    assert num_iter == 2


def test_yes_no_repeat():
    """
    Feature: YesNo Dataset
    Description: Test repeat
    Expectation: Output the amount of file
    """
    data = ds.YesNoDataset(DATA_DIR, num_samples=2)
    data = data.repeat(5)
    num_iter = 0
    for _ in data.create_dict_iterator(num_epochs=1):
        num_iter += 1
    assert num_iter == 10


def test_yes_no_dataset_size():
    """
    Feature: YesNo Dataset
    Description: Test dataset_size
    Expectation: Output the size of dataset
    """
    data = ds.YesNoDataset(DATA_DIR, shuffle=False)
    assert data.get_dataset_size() == 3


def test_yes_no_sequential_sampler():
    """
    Feature: YesNo Dataset
    Description: Use SequentialSampler to sample data.
    Expectation: The number of samplers returned by dict_iterator is equal to the requested number of samples.
    """
    logger.info("Test YesNoDataset Op with SequentialSampler")
    num_samples = 2
    sampler = ds.SequentialSampler(num_samples=num_samples)
    data1 = ds.YesNoDataset(DATA_DIR, sampler=sampler)
    data2 = ds.YesNoDataset(DATA_DIR, shuffle=False, num_samples=num_samples)
    sample_rate_list1, sample_rate_list2 = [], []
    num_iter = 0
    for item1, item2 in zip(data1.create_dict_iterator(num_epochs=1),
                            data2.create_dict_iterator(num_epochs=1)):
        sample_rate_list1.append(item1["sample_rate"])
        sample_rate_list2.append(item2["sample_rate"])
        num_iter += 1
    np.testing.assert_array_equal(sample_rate_list1, sample_rate_list2)
    assert num_iter == num_samples


def test_yes_no_exception():
    """
    Feature: Error tests
    Description: Throw error messages when certain errors occur
    Expectation: Output error message
    """
    logger.info("Test error cases for YesNoDataset")
    error_msg_1 = "sampler and shuffle cannot be specified at the same time"
    with pytest.raises(RuntimeError, match=error_msg_1):
        ds.YesNoDataset(DATA_DIR, shuffle=False, sampler=ds.PKSampler(3))

    error_msg_2 = "sampler and sharding cannot be specified at the same time"
    with pytest.raises(RuntimeError, match=error_msg_2):
        ds.YesNoDataset(DATA_DIR, sampler=ds.PKSampler(3),
                        num_shards=2, shard_id=0)

    error_msg_3 = "num_shards is specified and currently requires shard_id as well"
    with pytest.raises(RuntimeError, match=error_msg_3):
        ds.YesNoDataset(DATA_DIR, num_shards=10)

    error_msg_4 = "shard_id is specified but num_shards is not"
    with pytest.raises(RuntimeError, match=error_msg_4):
        ds.YesNoDataset(DATA_DIR, shard_id=0)

    error_msg_5 = "Input shard_id is not within the required interval"
    with pytest.raises(ValueError, match=error_msg_5):
        ds.YesNoDataset(DATA_DIR, num_shards=5, shard_id=-1)
    with pytest.raises(ValueError, match=error_msg_5):
        ds.YesNoDataset(DATA_DIR, num_shards=5, shard_id=5)
    with pytest.raises(ValueError, match=error_msg_5):
        ds.YesNoDataset(DATA_DIR, num_shards=2, shard_id=5)

    error_msg_6 = "num_parallel_workers exceeds"
    with pytest.raises(ValueError, match=error_msg_6):
        ds.YesNoDataset(DATA_DIR, shuffle=False, num_parallel_workers=0)
    with pytest.raises(ValueError, match=error_msg_6):
        ds.YesNoDataset(DATA_DIR, shuffle=False, num_parallel_workers=256)
    with pytest.raises(ValueError, match=error_msg_6):
        ds.YesNoDataset(DATA_DIR, shuffle=False, num_parallel_workers=-2)

    error_msg_7 = "Argument shard_id"
    with pytest.raises(TypeError, match=error_msg_7):
        ds.YesNoDataset(DATA_DIR, num_shards=2, shard_id="0")

    def exception_func(item):
        raise Exception("Error occur!")

    error_msg_8 = "The corresponding data file is"
    with pytest.raises(RuntimeError, match=error_msg_8):
        data = ds.YesNoDataset(DATA_DIR)
        data = data.map(operations=exception_func, input_columns=[
                        "waveform"], num_parallel_workers=1)
        for _ in data.__iter__():
            pass
    with pytest.raises(RuntimeError, match=error_msg_8):
        data = ds.YesNoDataset(DATA_DIR)
        data = data.map(operations=exception_func, input_columns=[
                        "sample_rate"], num_parallel_workers=1)
        for _ in data.__iter__():
            pass


def test_yes_no_pipeline():
    """
    Feature: Pipeline test
    Description: Read a sample
    Expectation: The amount of each function are equal
    """
    # Original waveform
    dataset = ds.YesNoDataset(DATA_DIR, num_samples=1)
    band_biquad_op = audio.BandBiquad(8000, 200.0)
    # Filtered waveform by bandbiquad
    dataset = dataset.map(input_columns=["waveform"], operations=band_biquad_op, num_parallel_workers=2)
    num_iter = 0
    for _ in dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
        num_iter += 1
    assert num_iter == 1


if __name__ == '__main__':
    test_yes_no_basic()
    test_yes_no_num_samples()
    test_yes_no_repeat()
    test_yes_no_dataset_size()
    test_yes_no_sequential_sampler()
    test_yes_no_exception()
    test_yes_no_pipeline()
