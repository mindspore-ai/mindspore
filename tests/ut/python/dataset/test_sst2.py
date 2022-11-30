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
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""
Test SST2 dataset operators
"""
import mindspore.dataset as ds

DATA_DIR = '../data/dataset/testSST2/'


def test_sst2_dataset_basic():
    """
    Feature: SST2Dataset
    Description: Read data from train file
    Expectation: The data is processed successfully
    """
    buffer = []
    data = ds.SST2Dataset(DATA_DIR, usage="train", shuffle=False)
    data = data.repeat(2)
    data = data.skip(3)
    for d in data.create_dict_iterator(num_epochs=1, output_numpy=True):
        buffer.append(d)
    assert len(buffer) == 7


def test_sst2_dataset_quoted():
    """
    Feature: SST2Dataset
    Description: Read the data and compare it to expectations
    Expectation: The data is processed successfully
    """
    data = ds.SST2Dataset(DATA_DIR, usage="test", shuffle=False)
    buffer = []
    for d in data.create_dict_iterator(num_epochs=1, output_numpy=True):
        buffer.extend([d['sentence']])
    assert buffer == ["test read SST2dataset 1 .",
                      "test read SST2dataset 2 .",
                      "test read SST2dataset 3 ."]


def test_sst2_dataset_usage():
    """
    Feature: SST2Dataset.
    Description: Tead all files with usage all.
    Expectation: The data is processed successfully.
    """
    buffer = []
    data = ds.SST2Dataset(DATA_DIR, usage="dev", shuffle=False)
    for d in data.create_dict_iterator(num_epochs=1, output_numpy=True):
        buffer.append(d)
    assert len(buffer) == 4


def test_sst2_dataset_get_dataset_size():
    """
    Feature: SST2Dataset
    Description: Test get_dataset_size function
    Expectation: The data is processed successfully
    """
    data = ds.SST2Dataset(DATA_DIR, usage="dev", shuffle=False)
    size = data.get_dataset_size()
    assert size == 4


def test_sst2_dataset_distribution():
    """
    Feature: SST2Dataset
    Description: Test in a distributed state
    Expectation: The data is processed successfully
    """
    data = ds.SST2Dataset(DATA_DIR, usage="train", shuffle=False, num_shards=2, shard_id=0)
    count = 0
    for _ in data.create_dict_iterator(num_epochs=1, output_numpy=True):
        count += 1
    assert count == 3


def test_sst2_dataset_num_samples():
    """
    Feature: SST2Dataset
    Description: Test num_samples parameter
    Expectation: The data is processed successfully
    """
    data = ds.SST2Dataset(DATA_DIR, usage="test", shuffle=False, num_samples=2)
    count = 0
    for _ in data.create_dict_iterator(num_epochs=1, output_numpy=True):
        count += 1
    assert count == 2


def test_sst2_dataset_exception():
    """
    Feature: SST2Dataset
    Description: Test the wrong input
    Expectation: Unable to read data properly
    """
    def exception_func(item):
        raise Exception("Error occur!")
    try:
        data = ds.SST2Dataset(DATA_DIR, usage="test", shuffle=False)
        data = data.map(operations=exception_func, input_columns=["sentence"], num_parallel_workers=1)
        for _ in data.create_dict_iterator():
            pass
        assert False
    except RuntimeError as e:
        assert "map operation: [PyFunc] failed. The corresponding data file" in str(e)
    try:
        data = ds.SST2Dataset(DATA_DIR, usage="test", shuffle=False)
        data = data.map(operations=exception_func, input_columns=["sentence"], num_parallel_workers=1)
        for _ in data.create_dict_iterator():
            pass
        assert False
    except RuntimeError as e:
        assert "map operation: [PyFunc] failed. The corresponding data file" in str(e)


if __name__ == "__main__":
    test_sst2_dataset_basic()
    test_sst2_dataset_quoted()
    test_sst2_dataset_usage()
    test_sst2_dataset_get_dataset_size()
    test_sst2_dataset_distribution()
    test_sst2_dataset_num_samples()
    test_sst2_dataset_exception()
