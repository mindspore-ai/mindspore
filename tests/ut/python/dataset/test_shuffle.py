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
import mindspore.dataset as ds
from mindspore import log as logger
from util import save_and_check_dict, config_get_set_seed

# Note: Number of rows in test.data dataset:  12
DATA_DIR = ["../data/dataset/testTFTestAllTypes/test.data"]
GENERATE_GOLDEN = False


def test_shuffle_01():
    """
    Feature: Shuffle op
    Description: Test shuffle op where buffer_size < number-of-rows-in-dataset
    Expectation: Output is equal to the expected output
    """
    logger.info("test_shuffle_01")
    # define parameters
    buffer_size = 5
    seed = 1

    # apply dataset operations
    data1 = ds.TFRecordDataset(DATA_DIR, shuffle=ds.Shuffle.FILES)
    original_seed = config_get_set_seed(seed)
    data1 = data1.shuffle(buffer_size=buffer_size)

    filename = "shuffle_01_result.npz"
    save_and_check_dict(data1, filename, generate_golden=GENERATE_GOLDEN)

    ds.config.set_seed(original_seed)


def test_shuffle_02():
    """
    Feature: Shuffle op
    Description: Test shuffle op where buffer_size == number-of-rows-in-dataset
    Expectation: Output is equal to the expected output
    """
    logger.info("test_shuffle_02")
    # define parameters
    buffer_size = 12
    seed = 1

    # apply dataset operations
    data1 = ds.TFRecordDataset(DATA_DIR, shuffle=ds.Shuffle.FILES)
    original_seed = config_get_set_seed(seed)
    data1 = data1.shuffle(buffer_size=buffer_size)

    filename = "shuffle_02_result.npz"
    save_and_check_dict(data1, filename, generate_golden=GENERATE_GOLDEN)

    ds.config.set_seed(original_seed)


def test_shuffle_03():
    """
    Feature: Shuffle op
    Description: Test shuffle op where buffer_size=2 (minimum size) and number-of-rows-in-dataset > 2
    Expectation: Output is equal to the expected output
    """
    logger.info("test_shuffle_03")
    # define parameters
    buffer_size = 2
    seed = 1

    # apply dataset operations
    data1 = ds.TFRecordDataset(DATA_DIR, shuffle=ds.Shuffle.FILES)
    original_seed = config_get_set_seed(seed)
    data1 = data1.shuffle(buffer_size)

    filename = "shuffle_03_result.npz"
    save_and_check_dict(data1, filename, generate_golden=GENERATE_GOLDEN)

    ds.config.set_seed(original_seed)


def test_shuffle_04():
    """
    Feature: Shuffle op
    Description: Test shuffle op where buffer_size=2 (minimum size) and number-of-rows-in-dataset=2
    Expectation: Output is equal to the expected output
    """
    logger.info("test_shuffle_04")
    # define parameters
    buffer_size = 2
    seed = 1

    # apply dataset operations
    data1 = ds.TFRecordDataset(DATA_DIR, num_samples=2)
    original_seed = config_get_set_seed(seed)
    data1 = data1.shuffle(buffer_size=buffer_size)

    filename = "shuffle_04_result.npz"
    save_and_check_dict(data1, filename, generate_golden=GENERATE_GOLDEN)

    ds.config.set_seed(original_seed)


def test_shuffle_05():
    """
    Feature: Shuffle op
    Description: Test shuffle op where buffer_size > number-of-rows-in-dataset
    Expectation: Output is equal to the expected output
    """
    logger.info("test_shuffle_05")
    # define parameters
    buffer_size = 13
    seed = 1

    # apply dataset operations
    data1 = ds.TFRecordDataset(DATA_DIR, shuffle=ds.Shuffle.FILES)
    original_seed = config_get_set_seed(seed)
    data1 = data1.shuffle(buffer_size=buffer_size)

    filename = "shuffle_05_result.npz"
    save_and_check_dict(data1, filename, generate_golden=GENERATE_GOLDEN)

    ds.config.set_seed(original_seed)


def test_shuffle_06():
    """
    Feature: Shuffle op
    Description: Test shuffle op with set seed for both datasets
    Expectation: Output is equal to the expected output
    """
    logger.info("test_shuffle_06")
    # define parameters
    buffer_size = 13
    seed = 1

    # apply dataset operations
    data1 = ds.TFRecordDataset(DATA_DIR, shuffle=ds.Shuffle.FILES)
    original_seed = config_get_set_seed(seed)
    data1 = data1.shuffle(buffer_size=buffer_size)

    data2 = ds.TFRecordDataset(DATA_DIR, shuffle=ds.Shuffle.FILES)
    data2 = data2.shuffle(buffer_size=buffer_size)

    for item1, item2 in zip(data1.create_dict_iterator(num_epochs=1, output_numpy=True),
                            data2.create_dict_iterator(num_epochs=1, output_numpy=True)):
        np.testing.assert_equal(item1, item2)

    ds.config.set_seed(original_seed)


def test_shuffle_exception_01():
    """
    Feature: Shuffle op
    Description: Test shuffle op where buffer_size < 0
    Expectation: Exception is raised as expected
    """
    logger.info("test_shuffle_exception_01")

    # apply dataset operations
    data1 = ds.TFRecordDataset(DATA_DIR)
    original_seed = config_get_set_seed(1)
    try:
        data1 = data1.shuffle(buffer_size=-1)
        sum([1 for _ in data1])

    except Exception as e:
        logger.info("Got an exception in DE: {}".format(str(e)))
        assert "Input buffer_size is not within the required interval of [2, 2147483647]" in str(e)

    ds.config.set_seed(original_seed)


def test_shuffle_exception_02():
    """
    Feature: Shuffle op
    Description: Test shuffle op where buffer_size=0
    Expectation: Exception is raised as expected
    """
    logger.info("test_shuffle_exception_02")

    # apply dataset operations
    data1 = ds.TFRecordDataset(DATA_DIR)
    original_seed = config_get_set_seed(1)
    try:
        data1 = data1.shuffle(buffer_size=0)
        sum([1 for _ in data1])

    except Exception as e:
        logger.info("Got an exception in DE: {}".format(str(e)))
        assert "Input buffer_size is not within the required interval of [2, 2147483647]" in str(e)

    ds.config.set_seed(original_seed)


def test_shuffle_exception_03():
    """
    Feature: Shuffle op
    Description: Test shuffle op where buffer_size=1
    Expectation: Exception is raised as expected
    """
    logger.info("test_shuffle_exception_03")

    # apply dataset operations
    data1 = ds.TFRecordDataset(DATA_DIR)
    original_seed = config_get_set_seed(1)
    try:
        data1 = data1.shuffle(buffer_size=1)
        sum([1 for _ in data1])

    except Exception as e:
        logger.info("Got an exception in DE: {}".format(str(e)))
        assert "Input buffer_size is not within the required interval of [2, 2147483647]" in str(e)

    ds.config.set_seed(original_seed)


def test_shuffle_exception_05():
    """
    Feature: Shuffle op
    Description: Test shuffle op where mandatory buffer_size input parameter is missing
    Expectation: Exception is raised as expected
    """
    logger.info("test_shuffle_exception_05")

    # apply dataset operations
    data1 = ds.TFRecordDataset(DATA_DIR)
    original_seed = config_get_set_seed(1)
    try:
        data1 = data1.shuffle()
        sum([1 for _ in data1])

    except Exception as e:
        logger.info("Got an exception in DE: {}".format(str(e)))
        assert "buffer_size" in str(e)

    ds.config.set_seed(original_seed)


def test_shuffle_exception_06():
    """
    Feature: Shuffle op
    Description: Test shuffle op where buffer_size has wrong type, boolean value False
    Expectation: Exception is raised as expected
    """
    logger.info("test_shuffle_exception_06")

    # apply dataset operations
    data1 = ds.TFRecordDataset(DATA_DIR)
    original_seed = config_get_set_seed(1)
    try:
        data1 = data1.shuffle(buffer_size=False)
        sum([1 for _ in data1])

    except Exception as e:
        logger.info("Got an exception in DE: {}".format(str(e)))
        assert "buffer_size" in str(e)

    ds.config.set_seed(original_seed)


def test_shuffle_exception_07():
    """
    Feature: Shuffle op
    Description: Test shuffle op where buffer_size has wrong type, boolean value True
    Expectation: Exception is raised as expected
    """
    logger.info("test_shuffle_exception_07")

    # apply dataset operations
    data1 = ds.TFRecordDataset(DATA_DIR)
    original_seed = config_get_set_seed(1)
    try:
        data1 = data1.shuffle(buffer_size=True)
        sum([1 for _ in data1])

    except Exception as e:
        logger.info("Got an exception in DE: {}".format(str(e)))
        assert "buffer_size" in str(e)

    ds.config.set_seed(original_seed)


if __name__ == '__main__':
    test_shuffle_01()
    test_shuffle_02()
    test_shuffle_03()
    test_shuffle_04()
    test_shuffle_05()
    test_shuffle_06()
    test_shuffle_exception_01()
    test_shuffle_exception_02()
    test_shuffle_exception_03()
    test_shuffle_exception_05()
    test_shuffle_exception_06()
    test_shuffle_exception_07()
    logger.info('\n')
