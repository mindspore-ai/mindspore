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

DATA_DIR = '../data/dataset/testUDPOSDataset/'


def test_udpos_dataset_one_file():
    """
    Feature: UDPOSDataset
    Description: Test UDPOSDataset with one file using test usage
    Expectation: Output is equal to the expected output
    """
    data = ds.UDPOSDataset(DATA_DIR, usage="test", shuffle=False)
    count = 0
    for i in data.create_dict_iterator(num_epochs=1, output_numpy=True):
        logger.info("{}".format(i["word"]))
        count += 1
    assert count == 1


def test_udpos_dataset_all_file():
    """
    Feature: UDPOSDataset
    Description: Test UDPOSDataset with all usage
    Expectation: Output is equal to the expected output
    """
    data = ds.UDPOSDataset(DATA_DIR, usage="all", shuffle=False)
    count = 0
    for i in data.create_dict_iterator(num_epochs=1, output_numpy=True):
        logger.info("{}".format(i["word"]))
        count += 1
    assert count == 6


def test_udpos_dataset_shuffle_false_four_parallel():
    """
    Feature: UDPOSDataset
    Description: Test UDPOSDataset with no shuffle and num_parallel_workers=4
    Expectation: Output is equal to the expected output
    """
    original_num_parallel_workers = config_get_set_num_parallel_workers(4)
    original_seed = config_get_set_seed(987)
    data = ds.UDPOSDataset(DATA_DIR, usage="all", shuffle=False)
    count = 0
    numword = 6
    line = ["From", "The", "Abed", "Come", "The", "Std",
            "What", "Like", "Good", "Mom", "Iike", "Good",
            "Abed", "...", "Zoom", "...", "Abed", "From",
            "Psg", "Bus", "Ori", "The", "Abed", "The",
            "...", "The", "ken", "Ori", "...", "Respect",
            "Bus", "Nine", "Job", "Mom", "Abed", "From"]
    for i in data.create_dict_iterator(num_epochs=1, output_numpy=True):
        for j in range(numword):
            strs = i["word"][j]
            assert strs == line[count*6+j]
        count += 1
    assert count == 6
    # Restore configuration
    ds.config.set_num_parallel_workers(original_num_parallel_workers)
    ds.config.set_seed(original_seed)


def test_udpos_dataset_shuffle_false_one_parallel():
    """
    Feature: UDPOSDataset
    Description: Test UDPOSDataset with no shuffle and num_parallel_workers=1
    Expectation: Output is equal to the expected output
    """
    original_num_parallel_workers = config_get_set_num_parallel_workers(1)
    original_seed = config_get_set_seed(987)
    data = ds.UDPOSDataset(DATA_DIR, usage="all", shuffle=False)
    count = 0
    numword = 6
    line = ["From", "The", "Abed", "Come", "The", "Std",
            "Psg", "Bus", "Ori", "The", "Abed", "The",
            "Bus", "Nine", "Job", "Mom", "Abed", "From",
            "What", "Like", "Good", "Mom", "Iike", "Good",
            "Abed", "...", "Zoom", "...", "Abed", "From",
            "...", "The", "ken", "Ori", "...", "Respect"]
    for i in data.create_dict_iterator(num_epochs=1, output_numpy=True):
        for j in range(numword):
            strs = i["word"][j]
            assert strs == line[count*6+j]
        count += 1
    assert count == 6
    # Restore configuration
    ds.config.set_num_parallel_workers(original_num_parallel_workers)
    ds.config.set_seed(original_seed)


def test_udpos_dataset_shuffle_files_four_parallel():
    """
    Feature: UDPOSDataset
    Description: Test UDPOSDataset with shuffle and num_parallel_workers=4
    Expectation: Output is equal to the expected output
    """
    original_num_parallel_workers = config_get_set_num_parallel_workers(4)
    original_seed = config_get_set_seed(135)
    data = ds.UDPOSDataset(DATA_DIR, usage="all", shuffle=ds.Shuffle.FILES)
    count = 0
    numword = 6
    line = ["Abed", "...", "Zoom", "...", "Abed", "From",
            "What", "Like", "Good", "Mom", "Iike", "Good",
            "From", "The", "Abed", "Come", "The", "Std",
            "...", "The", "ken", "Ori", "...", "Respect",
            "Psg", "Bus", "Ori", "The", "Abed", "The",
            "Bus", "Nine", "Job", "Mom", "Abed", "From"]
    for i in data.create_dict_iterator(num_epochs=1, output_numpy=True):
        for j in range(numword):
            strs = i["word"][j]
            assert strs == line[count*6+j]
        count += 1
    assert count == 6
    # Restore configuration
    ds.config.set_num_parallel_workers(original_num_parallel_workers)
    ds.config.set_seed(original_seed)


def test_udpos_dataset_shuffle_files_one_parallel():
    """
    Feature: UDPOSDataset
    Description: Test UDPOSDataset with shuffle and num_parallel_workers=1
    Expectation: Output is equal to the expected output
    """
    original_num_parallel_workers = config_get_set_num_parallel_workers(1)
    original_seed = config_get_set_seed(135)
    data = ds.UDPOSDataset(DATA_DIR, usage="all", shuffle=ds.Shuffle.FILES)
    count = 0
    numword = 6
    line = ["Abed", "...", "Zoom", "...", "Abed", "From",
            "...", "The", "ken", "Ori", "...", "Respect",
            "What", "Like", "Good", "Mom", "Iike", "Good",
            "From", "The", "Abed", "Come", "The", "Std",
            "Psg", "Bus", "Ori", "The", "Abed", "The",
            "Bus", "Nine", "Job", "Mom", "Abed", "From"]
    for i in data.create_dict_iterator(num_epochs=1, output_numpy=True):
        for j in range(numword):
            strs = i["word"][j]
            assert strs == line[count*6+j]
        count += 1
    assert count == 6
    # Restore configuration
    ds.config.set_num_parallel_workers(original_num_parallel_workers)
    ds.config.set_seed(original_seed)


def test_udpos_dataset_shuffle_global_four_parallel():
    """
    Feature: UDPOSDataset
    Description: Test UDPOSDataset with shuffle global and num_parallel_workers=4
    Expectation: Output is equal to the expected output
    """
    original_num_parallel_workers = config_get_set_num_parallel_workers(4)
    original_seed = config_get_set_seed(246)
    data = ds.UDPOSDataset(DATA_DIR, usage="all", shuffle=ds.Shuffle.GLOBAL)
    count = 0
    numword = 6
    line = ["Bus", "Nine", "Job", "Mom", "Abed", "From",
            "Abed", "...", "Zoom", "...", "Abed", "From",
            "From", "The", "Abed", "Come", "The", "Std",
            "Psg", "Bus", "Ori", "The", "Abed", "The",
            "What", "Like", "Good", "Mom", "Iike", "Good",
            "...", "The", "ken", "Ori", "...", "Respect"]
    for i in data.create_dict_iterator(num_epochs=1, output_numpy=True):
        for j in range(numword):
            strs = i["word"][j]
            assert strs == line[count*6+j]
        count += 1
    assert count == 6
    # Restore configuration
    ds.config.set_num_parallel_workers(original_num_parallel_workers)
    ds.config.set_seed(original_seed)


def test_udpos_dataset_shuffle_global_one_parallel():
    """
    Feature: UDPOSDataset
    Description: Test UDPOSDataset with shuffle global and num_parallel_workers=1
    Expectation: Output is equal to the expected output
    """
    original_num_parallel_workers = config_get_set_num_parallel_workers(1)
    original_seed = config_get_set_seed(246)
    data = ds.UDPOSDataset(DATA_DIR, usage="all", shuffle=ds.Shuffle.GLOBAL)
    count = 0
    numword = 6
    line = ["...", "The", "ken", "Ori", "...", "Respect",
            "Psg", "Bus", "Ori", "The", "Abed", "The",
            "From", "The", "Abed", "Come", "The", "Std",
            "Bus", "Nine", "Job", "Mom", "Abed", "From",
            "What", "Like", "Good", "Mom", "Iike", "Good",
            "Abed", "...", "Zoom", "...", "Abed", "From"]
    for i in data.create_dict_iterator(num_epochs=1, output_numpy=True):
        for j in range(numword):
            strs = i["word"][j]
            assert strs == line[count*6+j]
        count += 1
    assert count == 6
    # Restore configuration
    ds.config.set_num_parallel_workers(original_num_parallel_workers)
    ds.config.set_seed(original_seed)


def test_udpos_dataset_num_samples():
    """
    Feature: UDPOSDataset
    Description: Test UDPOSDataset with num_samples
    Expectation: Output is equal to the expected output
    """
    data = ds.UDPOSDataset(DATA_DIR, usage="test", shuffle=False, num_samples=2)
    count = 0
    for _ in data.create_dict_iterator(num_epochs=1, output_numpy=True):
        count += 1
    assert count == 1


def test_udpos_dataset_distribution():
    """
    Feature: UDPOSDataset
    Description: Test UDPOSDataset with num_shards and shard_id parameters
    Expectation: Output is equal to the expected output
    """
    data = ds.UDPOSDataset(DATA_DIR, usage="test", shuffle=False, num_shards=2, shard_id=1)
    count = 0
    for _ in data.create_dict_iterator(num_epochs=1, output_numpy=True):
        count += 1
    assert count == 1


def test_udpos_dataset_repeat():
    """
    Feature: UDPOSDataset
    Description: Test UDPOSDataset with repeat op
    Expectation: Output is equal to the expected output
    """
    data = ds.UDPOSDataset(DATA_DIR, usage="test", shuffle=False)
    data = data.repeat(3)
    count = 0
    numword = 6
    line = ["What", "Like", "Good", "Mom", "Iike", "Good",
            "What", "Like", "Good", "Mom", "Iike", "Good",
            "What", "Like", "Good", "Mom", "Iike", "Good"]
    for i in data.create_dict_iterator(num_epochs=1, output_numpy=True):
        for j in range(numword):
            strs = i["word"][j]
            assert strs == line[count*6+j]
        count += 1
    assert count == 3


def test_udpos_dataset_get_datasetsize():
    """
    Feature: UDPOSDataset
    Description: Test UDPOSDataset get_dataset_size
    Expectation: Output is equal to the expected output
    """
    data = ds.UDPOSDataset(DATA_DIR, usage="test", shuffle=False)
    size = data.get_dataset_size()
    assert size == 6


def test_udpos_dataset_device_que():
    """
    Feature: UDPOSDataset
    Description: Test UDPOSDataset device_que
    Expectation: Runs successfully
    """
    data = ds.UDPOSDataset(DATA_DIR, usage="test", shuffle=False)
    data = data.device_que()
    data.send()


def test_udpos_dataset_exceptions():
    """
    Feature: UDPOSDataset
    Description: Test UDPOSDataset with invalid inputs
    Expectation: Correct error is raised as expected
    """
    with pytest.raises(ValueError) as error_info:
        _ = ds.UDPOSDataset(DATA_DIR, usage="test", num_samples=-1)
    assert "num_samples exceeds the boundary" in str(error_info.value)

    with pytest.raises(ValueError) as error_info:
        _ = ds.UDPOSDataset("NotExistFile", usage="test")
    assert "The folder NotExistFile does not exist or is not a directory or permission denied!" in str(error_info.value)

    with pytest.raises(ValueError) as error_info:
        _ = ds.TextFileDataset("")
    assert "Input dataset_files can not be empty" in str(error_info.value)

    def exception_func(item):
        raise Exception("Error occur!")
    with pytest.raises(RuntimeError) as error_info:
        data = data = ds.UDPOSDataset(DATA_DIR, usage="test", shuffle=False)
        data = data.map(operations=exception_func, input_columns=["word"], num_parallel_workers=1)
        for _ in data.__iter__():
            pass
    assert "map operation: [PyFunc] failed. The corresponding data file is" in str(error_info.value)


if __name__ == "__main__":
    test_udpos_dataset_one_file()
    test_udpos_dataset_all_file()
    test_udpos_dataset_shuffle_false_four_parallel()
    test_udpos_dataset_shuffle_false_one_parallel()
    test_udpos_dataset_shuffle_files_one_parallel()
    test_udpos_dataset_shuffle_files_four_parallel()
    test_udpos_dataset_shuffle_global_four_parallel()
    test_udpos_dataset_shuffle_global_one_parallel()
    test_udpos_dataset_num_samples()
    test_udpos_dataset_distribution()
    test_udpos_dataset_repeat()
    test_udpos_dataset_get_datasetsize()
    test_udpos_dataset_device_que()
    test_udpos_dataset_exceptions()
