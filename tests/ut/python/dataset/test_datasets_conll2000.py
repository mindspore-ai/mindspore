# Copyright 2021-2022 Huawei Technologies Co., Ltd
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
import pytest

import mindspore.dataset as ds
from mindspore import log as logger
from util import config_get_set_num_parallel_workers, config_get_set_seed

DATA_DIR = '../data/dataset/testCoNLL2000Dataset'


def test_conll2000_dataset_one_file():
    """
    Feature: CoNLL2000Dataset
    Description: Test CoNLL2000Dataset with test usage
    Expectation: Output is equal to the expected output
    """
    data = ds.CoNLL2000Dataset(DATA_DIR, usage="test", shuffle=False)
    count = 0
    for i in data.create_dict_iterator(num_epochs=1, output_numpy=True):
        logger.info("{}".format(i["word"]))
        count += 1
    assert count == 2


def test_conll2000_dataset_all_file():
    """
    Feature: CoNLL2000Dataset
    Description: Test CoNLL2000Dataset with all usage
    Expectation: Output is equal to the expected output
    """
    data = ds.CoNLL2000Dataset(DATA_DIR, usage="all", shuffle=False)
    count = 0
    for i in data.create_dict_iterator(num_epochs=1, output_numpy=True):
        logger.info("{}".format(i["word"]))
        count += 1
    assert count == 5


def test_conll2000_dataset_num_samples_none():
    """
    Feature: CoNLL2000Dataset
    Description: Test CoNLL2000Dataset with no num_samples (None by default)
    Expectation: Output is equal to the expected output
    """
    # Do not provide a num_samples argument, so it would be None by default
    data = ds.CoNLL2000Dataset(DATA_DIR, usage="test", shuffle=False)
    count = 0
    for i in data.create_dict_iterator(num_epochs=1, output_numpy=True):
        logger.info("{}".format(i["word"]))
        count += 1
    assert count == 2


def test_conll2000_dataset_shuffle_false_num_parallel_workers_4():
    """
    Feature: CoNLL2000Dataset
    Description: Test CoNLL2000Dataset with no shuffle and num_parallel_workers=4
    Expectation: Output is equal to the expected output
    """
    original_num_parallel_workers = config_get_set_num_parallel_workers(4)
    original_seed = config_get_set_seed(987)
    data = ds.CoNLL2000Dataset(DATA_DIR, usage="all", shuffle=False)
    count = 0
    numword = 5
    line = ["He", "reckons", "the", "current", "account", ".",
            "Challenge", "of", "the", "August", "month", ".",
            "The", "1.8", "billion", "in", "September", ".",
            "Her", "'s", "chancellor", "at", "Lawson", ".",
            "To", "economists", ",", "foreign", "exchange", "."]
    for i in data.create_dict_iterator(num_epochs=1, output_numpy=True):
        for j in range(numword):
            strs = i["word"][j]
            assert strs == line[count*6+j]
        count += 1
    assert count == 5
    # Restore configuration
    ds.config.set_num_parallel_workers(original_num_parallel_workers)
    ds.config.set_seed(original_seed)


def test_conll2000_dataset_shuffle_false_num_parallel_workers_1():
    """
    Feature: CoNLL2000Dataset
    Description: Test CoNLL2000Dataset with no shuffle and num_parallel_workers=1
    Expectation: Output is equal to the expected output
    """
    original_num_parallel_workers = config_get_set_num_parallel_workers(1)
    original_seed = config_get_set_seed(987)
    data = ds.CoNLL2000Dataset(DATA_DIR, usage="all", shuffle=False)
    count = 0
    numword = 6
    line = ["He", "reckons", "the", "current", "account", ".",
            "The", "1.8", "billion", "in", "September", ".",
            "Challenge", "of", "the", "August", "month", ".",
            "Her", "'s", "chancellor", "at", "Lawson", ".",
            "To", "economists", ",", "foreign", "exchange", "."]
    for i in data.create_dict_iterator(num_epochs=1, output_numpy=True):
        for j in range(numword):
            strs = i["word"][j]
            assert strs == line[count*6+j]
        count += 1
    assert count == 5
    # Restore configuration
    ds.config.set_num_parallel_workers(original_num_parallel_workers)
    ds.config.set_seed(original_seed)


def test_conll2000_dataset_shuffle_files_num_parallel_workers_4():
    """
    Feature: CoNLL2000Dataset
    Description: Test CoNLL2000Dataset with shuffle and num_parallel_workers=4
    Expectation: Output is equal to the expected output
    """
    original_num_parallel_workers = config_get_set_num_parallel_workers(4)
    original_seed = config_get_set_seed(135)
    data = ds.CoNLL2000Dataset(DATA_DIR, usage="all", shuffle=ds.Shuffle.FILES)
    count = 0
    numword = 6
    line = ["He", "reckons", "the", "current", "account", ".",
            "Challenge", "of", "the", "August", "month", ".",
            "The", "1.8", "billion", "in", "September", ".",
            "Her", "'s", "chancellor", "at", "Lawson", ".",
            "To", "economists", ",", "foreign", "exchange", "."]
    for i in data.create_dict_iterator(num_epochs=1, output_numpy=True):
        for j in range(numword):
            strs = i["word"][j]
            assert strs == line[count*6+j]
        count += 1
    assert count == 5
    # Restore configuration
    ds.config.set_num_parallel_workers(original_num_parallel_workers)
    ds.config.set_seed(original_seed)


def test_conll2000_dataset_shuffle_files_num_parallel_workers_1():
    """
    Feature: CoNLL2000Dataset
    Description: Test CoNLL2000Dataset with shuffle and num_parallel_workers=1
    Expectation: Output is equal to the expected output
    """
    original_num_parallel_workers = config_get_set_num_parallel_workers(1)
    original_seed = config_get_set_seed(135)
    data = ds.CoNLL2000Dataset(DATA_DIR, usage="all", shuffle=ds.Shuffle.FILES)
    count = 0
    numword = 6
    line = ["He", "reckons", "the", "current", "account", ".",
            "The", "1.8", "billion", "in", "September", ".",
            "Challenge", "of", "the", "August", "month", ".",
            "Her", "'s", "chancellor", "at", "Lawson", ".",
            "To", "economists", ",", "foreign", "exchange", "."]
    for i in data.create_dict_iterator(num_epochs=1, output_numpy=True):
        for j in range(numword):
            strs = i["word"][j]
            assert strs == line[count*6+j]
        count += 1
    assert count == 5
    # Restore configuration
    ds.config.set_num_parallel_workers(original_num_parallel_workers)
    ds.config.set_seed(original_seed)


def test_conll2000_dataset_shuffle_global_num_parallel_workers_4():
    """
    Feature: CoNLL2000Dataset
    Description: Test CoNLL2000Dataset with shuffle global and num_parallel_workers=4
    Expectation: Output is equal to the expected output
    """
    original_num_parallel_workers = config_get_set_num_parallel_workers(4)
    original_seed = config_get_set_seed(246)
    data = ds.CoNLL2000Dataset(DATA_DIR, usage="all", shuffle=ds.Shuffle.GLOBAL)
    count = 0
    numword = 6
    line = ["Challenge", "of", "the", "August", "month", ".",
            "To", "economists", ",", "foreign", "exchange", ".",
            "Her", "'s", "chancellor", "at", "Lawson", ".",
            "He", "reckons", "the", "current", "account", ".",
            "The", "1.8", "billion", "in", "September", "."]
    for i in data.create_dict_iterator(num_epochs=1, output_numpy=True):
        for j in range(numword):
            strs = i["word"][j]
            assert strs == line[count*6+j]
        count += 1
    assert count == 5
    # Restore configuration
    ds.config.set_num_parallel_workers(original_num_parallel_workers)
    ds.config.set_seed(original_seed)


def test_conll2000_dataset_shuffle_global_num_parallel_workers_1():
    """
    Feature: CoNLL2000Dataset
    Description: Test CoNLL2000Dataset with shuffle global and num_parallel_workers=1
    Expectation: Output is equal to the expected output
    """
    original_num_parallel_workers = config_get_set_num_parallel_workers(1)
    original_seed = config_get_set_seed(246)
    data = ds.CoNLL2000Dataset(DATA_DIR, usage="all", shuffle=ds.Shuffle.GLOBAL)
    count = 0
    numword = 6
    line = ["Challenge", "of", "the", "August", "month", ".",
            "The", "1.8", "billion", "in", "September", ".",
            "To", "economists", ",", "foreign", "exchange", ".",
            "Her", "'s", "chancellor", "at", "Lawson", ".",
            "He", "reckons", "the", "current", "account", "."]
    for i in data.create_dict_iterator(num_epochs=1, output_numpy=True):
        for j in range(numword):
            strs = i["word"][j]
            assert strs == line[count*6+j]
        count += 1
    assert count == 5
    # Restore configuration
    ds.config.set_num_parallel_workers(original_num_parallel_workers)
    ds.config.set_seed(original_seed)


def test_conll2000_dataset_num_samples():
    """
    Feature: CoNLL2000Dataset
    Description: Test CoNLL2000Dataset with num_samples
    Expectation: Output is equal to the expected output
    """
    data = ds.CoNLL2000Dataset(DATA_DIR, usage="test", shuffle=False, num_samples=2)
    count = 0
    for _ in data.create_dict_iterator(num_epochs=1, output_numpy=True):
        count += 1
    assert count == 2


def test_conll2000_dataset_distribution():
    """
    Feature: CoNLL2000Dataset
    Description: Test CoNLL2000Dataset with num_shards and shard_id parameters
    Expectation: Output is equal to the expected output
    """
    data = ds.CoNLL2000Dataset(DATA_DIR, usage="test", shuffle=False, num_shards=2, shard_id=1)
    count = 0
    for _ in data.create_dict_iterator(num_epochs=1, output_numpy=True):
        count += 1
    assert count == 1


def test_conll2000_dataset_repeat():
    """
    Feature: CoNLL2000Dataset
    Description: Test CoNLL2000Dataset with repeat op
    Expectation: Output is equal to the expected output
    """
    data = ds.CoNLL2000Dataset(DATA_DIR, usage="test", shuffle=False)
    data = data.repeat(3)
    count = 0
    numword = 6
    line = ["He", "reckons", "the", "current", "account", ".",
            "The", "1.8", "billion", "in", "September", ".",
            "He", "reckons", "the", "current", "account", ".",
            "The", "1.8", "billion", "in", "September", ".",
            "He", "reckons", "the", "current", "account", ".",
            "The", "1.8", "billion", "in", "September", ".",]
    for i in data.create_dict_iterator(num_epochs=1, output_numpy=True):
        for j in range(numword):
            strs = i["word"][j]
            assert strs == line[count*6+j]
        count += 1
    assert count == 6


def test_conll2000_dataset_get_datasetsize():
    """
    Feature: CoNLL2000Dataset
    Description: Test CoNLL2000Dataset get_dataset_size
    Expectation: Output is equal to the expected output
    """
    data = ds.CoNLL2000Dataset(DATA_DIR, usage="test", shuffle=False)
    size = data.get_dataset_size()
    assert size == 12


def test_conll2000_dataset_device_que():
    """
    Feature: CoNLL2000Dataset
    Description: Test CoNLL2000Dataset device_que
    Expectation: Runs successfully
    """
    data = ds.CoNLL2000Dataset(DATA_DIR, usage="test", shuffle=False)
    data = data.device_que()
    data.send()


def test_conll2000_dataset_exceptions():
    """
    Feature: CoNLL2000Dataset
    Description: Test CoNLL2000Dataset with invalid inputs
    Expectation: Correct error is raised as expected
    """
    with pytest.raises(ValueError) as error_info:
        _ = ds.CoNLL2000Dataset(DATA_DIR, usage="test", num_samples=-1)
    assert "num_samples exceeds the boundary" in str(error_info.value)

    with pytest.raises(ValueError) as error_info:
        _ = ds.CoNLL2000Dataset("NotExistFile", usage="test")
    assert "The folder NotExistFile does not exist or is not a directory or permission denied!" in str(error_info.value)

    with pytest.raises(ValueError) as error_info:
        _ = ds.TextFileDataset("")
    assert "Input dataset_files can not be empty" in str(error_info.value)


    def exception_func(item):
        raise Exception("Error occur!")
    with pytest.raises(RuntimeError) as error_info:
        data = data = ds.CoNLL2000Dataset(DATA_DIR, usage="test", shuffle=False)
        data = data.map(operations=exception_func, input_columns=["word"], num_parallel_workers=1)
        for _ in data.__iter__():
            pass
    assert "map operation: [PyFunc] failed. The corresponding data file is" in str(error_info.value)


if __name__ == "__main__":
    test_conll2000_dataset_one_file()
    test_conll2000_dataset_all_file()
    test_conll2000_dataset_num_samples_none()
    test_conll2000_dataset_shuffle_false_num_parallel_workers_4()
    test_conll2000_dataset_shuffle_false_num_parallel_workers_1()
    test_conll2000_dataset_shuffle_files_num_parallel_workers_4()
    test_conll2000_dataset_shuffle_files_num_parallel_workers_1()
    test_conll2000_dataset_shuffle_global_num_parallel_workers_4()
    test_conll2000_dataset_shuffle_global_num_parallel_workers_1()
    test_conll2000_dataset_num_samples()
    test_conll2000_dataset_distribution()
    test_conll2000_dataset_repeat()
    test_conll2000_dataset_get_datasetsize()
    test_conll2000_dataset_device_que()
    test_conll2000_dataset_exceptions()
