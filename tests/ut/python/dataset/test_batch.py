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
import mindspore.dataset as ds
from mindspore import log as logger
from util import save_and_check_dict

# Note: Number of rows in test.data dataset:  12
DATA_DIR = ["../data/dataset/testTFTestAllTypes/test.data"]
GENERATE_GOLDEN = False


def test_batch_01():
    """
    Test batch: batch_size>1, drop_remainder=True, no remainder exists
    """
    logger.info("test_batch_01")
    # define parameters
    batch_size = 2
    drop_remainder = True

    # apply dataset operations
    data1 = ds.TFRecordDataset(DATA_DIR, shuffle=ds.Shuffle.FILES)
    data1 = data1.batch(batch_size, drop_remainder)

    assert sum([1 for _ in data1]) == 6
    filename = "batch_01_result.npz"
    save_and_check_dict(data1, filename, generate_golden=GENERATE_GOLDEN)


def test_batch_02():
    """
    Test batch: batch_size>1, drop_remainder=True, remainder exists
    """
    logger.info("test_batch_02")
    # define parameters
    batch_size = 5
    drop_remainder = True

    # apply dataset operations
    data1 = ds.TFRecordDataset(DATA_DIR, shuffle=ds.Shuffle.FILES)
    data1 = data1.batch(batch_size, drop_remainder=drop_remainder)

    assert sum([1 for _ in data1]) == 2
    filename = "batch_02_result.npz"
    save_and_check_dict(data1, filename, generate_golden=GENERATE_GOLDEN)


def test_batch_03():
    """
    Test batch: batch_size>1, drop_remainder=False, no remainder exists
    """
    logger.info("test_batch_03")
    # define parameters
    batch_size = 3
    drop_remainder = False

    # apply dataset operations
    data1 = ds.TFRecordDataset(DATA_DIR, shuffle=ds.Shuffle.FILES)
    data1 = data1.batch(batch_size=batch_size, drop_remainder=drop_remainder)

    assert sum([1 for _ in data1]) == 4
    filename = "batch_03_result.npz"
    save_and_check_dict(data1, filename, generate_golden=GENERATE_GOLDEN)


def test_batch_04():
    """
    Test batch: batch_size>1, drop_remainder=False, remainder exists
    """
    logger.info("test_batch_04")
    # define parameters
    batch_size = 7
    drop_remainder = False

    # apply dataset operations
    data1 = ds.TFRecordDataset(DATA_DIR, shuffle=ds.Shuffle.FILES)
    data1 = data1.batch(batch_size, drop_remainder)

    assert sum([1 for _ in data1]) == 2
    filename = "batch_04_result.npz"
    save_and_check_dict(data1, filename, generate_golden=GENERATE_GOLDEN)


def test_batch_05():
    """
    Test batch: batch_size=1 (minimum valid size), drop_remainder default
    """
    logger.info("test_batch_05")
    # define parameters
    batch_size = 1

    # apply dataset operations
    data1 = ds.TFRecordDataset(DATA_DIR, shuffle=ds.Shuffle.FILES)
    data1 = data1.batch(batch_size)

    assert sum([1 for _ in data1]) == 12
    filename = "batch_05_result.npz"
    save_and_check_dict(data1, filename, generate_golden=GENERATE_GOLDEN)


def test_batch_06():
    """
    Test batch: batch_size = number-of-rows-in-dataset, drop_remainder=True, reorder params
    """
    logger.info("test_batch_06")
    # define parameters
    batch_size = 12
    drop_remainder = False

    # apply dataset operations
    data1 = ds.TFRecordDataset(DATA_DIR, shuffle=ds.Shuffle.FILES)
    data1 = data1.batch(drop_remainder=drop_remainder, batch_size=batch_size)

    assert sum([1 for _ in data1]) == 1
    filename = "batch_06_result.npz"
    save_and_check_dict(data1, filename, generate_golden=GENERATE_GOLDEN)


def test_batch_07():
    """
    Test batch: num_parallel_workers>1, drop_remainder=False, reorder params
    """
    logger.info("test_batch_07")
    # define parameters
    batch_size = 4
    drop_remainder = False
    num_parallel_workers = 2

    # apply dataset operations
    data1 = ds.TFRecordDataset(DATA_DIR, shuffle=ds.Shuffle.FILES)
    data1 = data1.batch(num_parallel_workers=num_parallel_workers, drop_remainder=drop_remainder,
                        batch_size=batch_size)

    assert sum([1 for _ in data1]) == 3
    filename = "batch_07_result.npz"
    save_and_check_dict(data1, filename, generate_golden=GENERATE_GOLDEN)


def test_batch_08():
    """
    Test batch: num_parallel_workers=1, drop_remainder default
    """
    logger.info("test_batch_08")
    # define parameters
    batch_size = 6
    num_parallel_workers = 1

    # apply dataset operations
    data1 = ds.TFRecordDataset(DATA_DIR, shuffle=ds.Shuffle.FILES)
    data1 = data1.batch(batch_size, num_parallel_workers=num_parallel_workers)

    assert sum([1 for _ in data1]) == 2
    filename = "batch_08_result.npz"
    save_and_check_dict(data1, filename, generate_golden=GENERATE_GOLDEN)


def test_batch_09():
    """
    Test batch: batch_size > number-of-rows-in-dataset, drop_remainder=False
    """
    logger.info("test_batch_09")
    # define parameters
    batch_size = 13
    drop_remainder = False

    # apply dataset operations
    data1 = ds.TFRecordDataset(DATA_DIR, shuffle=ds.Shuffle.FILES)
    data1 = data1.batch(batch_size, drop_remainder=drop_remainder)

    assert sum([1 for _ in data1]) == 1
    filename = "batch_09_result.npz"
    save_and_check_dict(data1, filename, generate_golden=GENERATE_GOLDEN)


def test_batch_10():
    """
    Test batch: batch_size > number-of-rows-in-dataset, drop_remainder=True
    """
    logger.info("test_batch_10")
    # define parameters
    batch_size = 99
    drop_remainder = True

    # apply dataset operations
    data1 = ds.TFRecordDataset(DATA_DIR, shuffle=ds.Shuffle.FILES)
    data1 = data1.batch(batch_size, drop_remainder=drop_remainder)

    assert sum([1 for _ in data1]) == 0
    filename = "batch_10_result.npz"
    save_and_check_dict(data1, filename, generate_golden=GENERATE_GOLDEN)


def test_batch_11():
    """
    Test batch: batch_size=1 and dataset-size=1
    """
    logger.info("test_batch_11")
    # define parameters
    batch_size = 1

    # apply dataset operations
    # Use schema file with 1 row
    schema_file = "../data/dataset/testTFTestAllTypes/datasetSchema1Row.json"
    data1 = ds.TFRecordDataset(DATA_DIR, schema_file)
    data1 = data1.batch(batch_size)

    assert sum([1 for _ in data1]) == 1
    filename = "batch_11_result.npz"
    save_and_check_dict(data1, filename, generate_golden=GENERATE_GOLDEN)


def test_batch_12():
    """
    Test batch: batch_size boolean value True, treated as valid value 1
    """
    logger.info("test_batch_12")
    # define parameters
    batch_size = True

    # apply dataset operations
    data1 = ds.TFRecordDataset(DATA_DIR, shuffle=ds.Shuffle.FILES)
    data1 = data1.batch(batch_size=batch_size)

    assert sum([1 for _ in data1]) == 12
    filename = "batch_12_result.npz"
    save_and_check_dict(data1, filename, generate_golden=GENERATE_GOLDEN)


def test_batch_exception_01():
    """
    Test batch exception: num_parallel_workers=0
    """
    logger.info("test_batch_exception_01")

    # apply dataset operations
    data1 = ds.TFRecordDataset(DATA_DIR, shuffle=ds.Shuffle.FILES)
    try:
        data1 = data1.batch(batch_size=2, drop_remainder=True, num_parallel_workers=0)
        sum([1 for _ in data1])

    except Exception as e:
        logger.info("Got an exception in DE: {}".format(str(e)))
        assert "num_parallel_workers" in str(e)


def test_batch_exception_02():
    """
    Test batch exception: num_parallel_workers<0
    """
    logger.info("test_batch_exception_02")

    # apply dataset operations
    data1 = ds.TFRecordDataset(DATA_DIR, shuffle=ds.Shuffle.FILES)
    try:
        data1 = data1.batch(3, drop_remainder=True, num_parallel_workers=-1)
        sum([1 for _ in data1])

    except Exception as e:
        logger.info("Got an exception in DE: {}".format(str(e)))
        assert "num_parallel_workers" in str(e)


def test_batch_exception_03():
    """
    Test batch exception: batch_size=0
    """
    logger.info("test_batch_exception_03")

    # apply dataset operations
    data1 = ds.TFRecordDataset(DATA_DIR, shuffle=ds.Shuffle.FILES)
    try:
        data1 = data1.batch(batch_size=0)
        sum([1 for _ in data1])

    except Exception as e:
        logger.info("Got an exception in DE: {}".format(str(e)))
        assert "batch_size" in str(e)


def test_batch_exception_04():
    """
    Test batch exception: batch_size<0
    """
    logger.info("test_batch_exception_04")

    # apply dataset operations
    data1 = ds.TFRecordDataset(DATA_DIR, shuffle=ds.Shuffle.FILES)
    try:
        data1 = data1.batch(batch_size=-1)
        sum([1 for _ in data1])

    except Exception as e:
        logger.info("Got an exception in DE: {}".format(str(e)))
        assert "batch_size" in str(e)


def test_batch_exception_05():
    """
    Test batch exception: batch_size boolean value False, treated as invalid value 0
    """
    logger.info("test_batch_exception_05")

    # apply dataset operations
    data1 = ds.TFRecordDataset(DATA_DIR, shuffle=ds.Shuffle.FILES)
    try:
        data1 = data1.batch(batch_size=False)
        sum([1 for _ in data1])

    except Exception as e:
        logger.info("Got an exception in DE: {}".format(str(e)))
        assert "batch_size" in str(e)


def test_batch_exception_07():
    """
    Test batch exception: drop_remainder wrong type
    """
    logger.info("test_batch_exception_07")

    # apply dataset operations
    data1 = ds.TFRecordDataset(DATA_DIR, shuffle=ds.Shuffle.FILES)
    try:
        data1 = data1.batch(3, drop_remainder=0)
        sum([1 for _ in data1])

    except Exception as e:
        logger.info("Got an exception in DE: {}".format(str(e)))
        assert "drop_remainder" in str(e)


def test_batch_exception_08():
    """
    Test batch exception: num_parallel_workers wrong type
    """
    logger.info("test_batch_exception_08")

    # apply dataset operations
    data1 = ds.TFRecordDataset(DATA_DIR, shuffle=ds.Shuffle.FILES)
    try:
        data1 = data1.batch(3, drop_remainder=True, num_parallel_workers=False)
        sum([1 for _ in data1])

    except Exception as e:
        logger.info("Got an exception in DE: {}".format(str(e)))
        assert "num_parallel_workers" in str(e)


def test_batch_exception_09():
    """
    Test batch exception: Missing mandatory batch_size
    """
    logger.info("test_batch_exception_09")

    # apply dataset operations
    data1 = ds.TFRecordDataset(DATA_DIR, shuffle=ds.Shuffle.FILES)
    try:
        data1 = data1.batch(drop_remainder=True, num_parallel_workers=4)
        sum([1 for _ in data1])

    except Exception as e:
        logger.info("Got an exception in DE: {}".format(str(e)))
        assert "batch_size" in str(e)


def test_batch_exception_10():
    """
    Test batch exception: num_parallel_workers>>1
    """
    logger.info("test_batch_exception_10")

    # apply dataset operations
    data1 = ds.TFRecordDataset(DATA_DIR, shuffle=ds.Shuffle.FILES)
    try:
        data1 = data1.batch(batch_size=4, num_parallel_workers=8192)
        sum([1 for _ in data1])

    except Exception as e:
        logger.info("Got an exception in DE: {}".format(str(e)))
        assert "num_parallel_workers" in str(e)


def test_batch_exception_11():
    """
    Test batch exception: wrong input order, num_parallel_workers wrongly used as drop_remainder
    """
    logger.info("test_batch_exception_11")
    # define parameters
    batch_size = 6
    num_parallel_workers = 1

    # apply dataset operations
    data1 = ds.TFRecordDataset(DATA_DIR)
    try:
        data1 = data1.batch(batch_size, num_parallel_workers)
        sum([1 for _ in data1])

    except Exception as e:
        logger.info("Got an exception in DE: {}".format(str(e)))
        assert "drop_remainder" in str(e)


def test_batch_exception_12():
    """
    Test batch exception: wrong input order, drop_remainder wrongly used as batch_size
    """
    logger.info("test_batch_exception_12")
    # define parameters
    batch_size = 1
    drop_remainder = True

    # apply dataset operations
    data1 = ds.TFRecordDataset(DATA_DIR)
    try:
        data1 = data1.batch(drop_remainder, batch_size)
        sum([1 for _ in data1])

    except Exception as e:
        logger.info("Got an exception in DE: {}".format(str(e)))
        assert "drop_remainder" in str(e)


def test_batch_exception_13():
    """
    Test batch exception: invalid input parameter
    """
    logger.info("test_batch_exception_13")
    # define parameters
    batch_size = 4

    # apply dataset operations
    data1 = ds.TFRecordDataset(DATA_DIR)
    try:
        data1 = data1.batch(batch_size, shard_id=1)
        sum([1 for _ in data1])

    except Exception as e:
        logger.info("Got an exception in DE: {}".format(str(e)))
        assert "shard_id" in str(e)


def test_batch_exception_14():
    """
    Test per_batch_map and input column name
    """
    logger.info("test_batch_exception_14")
    batch_size = 2
    input_columns = ["num"]
    data1 = ds.TFRecordDataset(DATA_DIR)
    try:
        _ = data1.batch(batch_size=batch_size, input_columns=input_columns)
    except ValueError as e:
        assert "per_batch_map and input_columns need to be passed in together." in str(e)


def test_batch_exception_15():
    """
    Test batch_size = int32 max value + 1
    """
    logger.info("test_batch_exception_15")
    batch_size = 2147483647 + 1
    input_columns = ["num"]
    data1 = ds.TFRecordDataset(DATA_DIR)
    err_msg = ""
    try:
        _ = data1.batch(batch_size=batch_size, input_columns=input_columns)
    except ValueError as e:
        err_msg = str(e)
    assert "batch_size is not within the required interval of [1, 2147483647]" in err_msg


if __name__ == '__main__':
    test_batch_01()
    test_batch_02()
    test_batch_03()
    test_batch_04()
    test_batch_05()
    test_batch_06()
    test_batch_07()
    test_batch_08()
    test_batch_09()
    test_batch_10()
    test_batch_11()
    test_batch_12()
    test_batch_exception_01()
    test_batch_exception_02()
    test_batch_exception_03()
    test_batch_exception_04()
    test_batch_exception_05()
    test_batch_exception_07()
    test_batch_exception_08()
    test_batch_exception_09()
    test_batch_exception_10()
    test_batch_exception_11()
    test_batch_exception_12()
    test_batch_exception_13()
    test_batch_exception_14()
    test_batch_exception_15()
    logger.info('\n')
