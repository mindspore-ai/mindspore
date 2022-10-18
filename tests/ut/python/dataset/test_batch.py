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
from util import save_and_check_dict

# Note: Number of rows in test.data dataset:  12
DATA_DIR = ["../data/dataset/testTFTestAllTypes/test.data"]
GENERATE_GOLDEN = False


def test_batch_01():
    """
    Feature: Batch op
    Description: Test Batch op with batch_size>1, drop_remainder=True, and no remainder exists
    Expectation: The dataset is processed as expected
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
    Feature: Batch op
    Description: Test Batch op with batch_size>1, drop_remainder=True, and remainder exists
    Expectation: The dataset is processed as expected
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
    Feature: Batch op
    Description: Test Batch op with batch_size>1, drop_remainder=False, and no remainder exists
    Expectation: The dataset is processed as expected
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
    Feature: Batch op
    Description: Test Batch op with batch_size>1, drop_remainder=False, and remainder exists
    Expectation: The dataset is processed as expected
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
    Feature: Batch op
    Description: Test Batch op with batch_size=1 (minimum valid size), drop_remainder default
    Expectation: The dataset is processed as expected
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
    Feature: Batch op
    Description: Test Batch op with batch_size = number-of-rows-in-dataset, drop_remainder=True, reorder parameters
    Expectation: The dataset is processed as expected
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
    Feature: Batch op
    Description: Test Batch op with num_parallel_workers>1, drop_remainder=False, reorder parameters
    Expectation: The dataset is processed as expected
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
    Feature: Batch op
    Description: Test Batch op with num_parallel_workers=1, drop_remainder default
    Expectation: The dataset is processed as expected
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
    Feature: Batch op
    Description: Test Batch op with batch_size > number-of-rows-in-dataset, drop_remainder=False
    Expectation: The dataset is processed as expected
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
    Feature: Batch op
    Description: Test Batch op with batch_size > number-of-rows-in-dataset, drop_remainder=True
    Expectation: The dataset is processed as expected
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
    Feature: Batch op
    Description: Test Batch op with batch_size=1 and dataset-size=1
    Expectation: The dataset is processed as expected
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
    Feature: Batch op
    Description: Test Batch op with batch_size boolean value True, treated as valid value 1
    Expectation: The dataset is processed as expected
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


def test_batch_13():
    """
    Feature: Batch op
    Description: Test python_multiprocessing is True with per_batch_map is None
    Expectation: python_multiprocessing is True is ignored when per_batch_map is None
    """
    logger.info("test_batch_13")
    # define parameters
    batch_size = True

    # apply dataset operations
    data1 = ds.TFRecordDataset(DATA_DIR, shuffle=ds.Shuffle.FILES)
    data1 = data1.batch(batch_size=batch_size, python_multiprocessing=True)

    assert sum([1 for _ in data1]) == 12
    filename = "batch_12_result.npz"
    save_and_check_dict(data1, filename, generate_golden=GENERATE_GOLDEN)


def test_batch_exception_01():
    """
    Feature: Batch op
    Description: Test Batch op with num_parallel_workers=0
    Expectation: Exception is raised as expected
    """
    logger.info("test_batch_exception_01")

    # apply dataset operations
    data1 = ds.TFRecordDataset(DATA_DIR, shuffle=ds.Shuffle.FILES)
    try:
        data1 = data1.batch(
            batch_size=2, drop_remainder=True, num_parallel_workers=0)
        sum([1 for _ in data1])

    except Exception as e:
        logger.info("Got an exception in DE: {}".format(str(e)))
        assert "num_parallel_workers" in str(e)


def test_batch_exception_02():
    """
    Feature: Batch op
    Description: Test Batch op with num_parallel_workers<0
    Expectation: Exception is raised as expected
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
    Feature: Batch op
    Description: Test Batch op with batch_size=0
    Expectation: Exception is raised as expected
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
    Feature: Batch op
    Description: Test Batch op with batch_size<0
    Expectation: Exception is raised as expected
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
    Feature: Batch op
    Description: Test Batch op boolean value False, treated as invalid value 0
    Expectation: Exception is raised as expected
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
    Feature: Batch op
    Description: Test Batch op with drop_remainder wrong type
    Expectation: Exception is raised as expected
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
    Feature: Batch op
    Description: Test Batch op with num_parallel_workers wrong type
    Expectation: Exception is raised as expected
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
    Feature: Batch op
    Description: Test Batch op with missing mandatory batch_size
    Expectation: Exception is raised as expected
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
    Feature: Batch op
    Description: Test Batch op with num_parallel_workers>>1
    Expectation: Exception is raised as expected
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
    Feature: Batch op
    Description: Test Batch op with wrong input order, num_parallel_workers wrongly used as drop_remainder
    Expectation: Exception is raised as expected
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
    Feature: Batch op
    Description: Test Batch op with wrong input order, drop_remainder wrongly used as batch_size
    Expectation: Exception is raised as expected
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
    Feature: Batch op
    Description: Test Batch op with invalid input parameter
    Expectation: Exception is raised as expected
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
    Feature: Batch op
    Description: Test per_batch_map and input column name
    Expectation: Error is raised as expected
    """
    logger.info("test_batch_exception_14")
    batch_size = 2
    input_columns = ["num"]
    data1 = ds.TFRecordDataset(DATA_DIR)
    try:
        _ = data1.batch(batch_size=batch_size, input_columns=input_columns)
    except ValueError as e:
        assert "input_columns can be specified only when per_batch_map is set." in str(e)


def test_batch_exception_15():
    """
    Feature: Batch op
    Description: Test Batch op with batch_size = int32 max value + 1
    Expectation: Error is raised as expected
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


def test_batch_exception_16():
    """
    Feature: Batch op
    Description: Test Batch op with mismatched batch type
    Expectation: Error is raised as expected
    """
    def gen(num):
        for i in range(num):
            if i % 2 == 0:
                yield (np.array([i]), np.array([i + (1 + i) * 0.01]))
            else:
                yield (np.array([(i + 1) * 0.01 + i]), np.array([i]))

    def swap_col(col1, col2, batch_info):
        return ([np.copy(a) for a in col2], [np.copy(b) for b in col1])

    logger.info("test_batch_exception_16")

    batch_size = 4
    input_columns = ["num1", "num2"]
    data1 = ds.GeneratorDataset((lambda: gen(20)), input_columns)
    with pytest.raises(RuntimeError) as raise_info:
        result = data1.batch(batch_size=batch_size, per_batch_map=swap_col)
        for _ in result.create_dict_iterator(num_epochs=1, output_numpy=True):
            pass
    assert "Inconsistent batch type, batch operation expects same type for each data row" in str(raise_info.value)


def test_batch_exception_17():
    """
    Feature: Batch op
    Description: Test Batch op with mismatched batch size
    Expectation: Error is raised as expected
    """
    def gen(num):
        for i in range(1, num + 1):
            yield np.array([i] * i)

    logger.info("test_batch_exception_17")

    batch_size = 4
    input_columns = ["num1"]
    data1 = ds.GeneratorDataset((lambda: gen(20)), input_columns)
    with pytest.raises(RuntimeError) as raise_info:
        result = data1.batch(batch_size=batch_size)
        for _ in result.create_dict_iterator(num_epochs=1, output_numpy=True):
            pass
    assert "Inconsistent batch shapes, batch operation expects same shape for each data row" in str(raise_info.value)


def test_no_input_columns_01():
    """
    Feature: Batch op
    Description: Test with per_batch_map has value but input_columns has no value
    Expectation: Output is equal to the expected output
    """
    def gen_2_cols(num):
        for i in range(1, 1 + num):
            yield (np.array([i]), np.array([i ** 2]))

    def swap_col(col1, col2, batch_info):
        return ([np.copy(a) for a in col2], [np.copy(b) for b in col1])

    def batch_map_config(num, s, f, col_order=None):
        try:
            dst = ds.GeneratorDataset((lambda: gen_2_cols(num)), ["col1", "col2"])
            dst = dst.batch(batch_size=s, per_batch_map=f)
            res = []
            for row in dst.create_dict_iterator(num_epochs=1, output_numpy=True):
                res.append(row)
            return res
        except (ValueError, RuntimeError, TypeError) as e:
            return str(e)

    res = batch_map_config(3, 3, swap_col)[0]
    assert np.array_equal(res["col1"], [[1], [4], [9]]) and np.array_equal(res["col2"], [[1], [2], [3]])


def test_no_input_columns_02():
    """
    Feature: Batch op
    Description: Test per_batch_map has value but input_columns has no value and given output_columns parameter
    Expectation: Output is equal to the expected output
    """
    def gen_2_cols(num):
        for i in range(1, 1 + num):
            yield (np.array([i]), np.array([i ** 2]))

    def split_col(col1, col2, batch_info):
        return (col1, [np.copy(arr) for arr in col2], [np.copy(-arr) for arr in col2])

    def batch_map_config(num, s, f, out_nms, col_order=None):
        try:
            dst = ds.GeneratorDataset((lambda: gen_2_cols(num)), ["col1", "col2"])
            dst = dst.batch(batch_size=s, per_batch_map=f, output_columns=out_nms)
            res = []
            for row in dst.create_dict_iterator(num_epochs=1, output_numpy=True):
                res.append(row)
            return res
        except (ValueError, RuntimeError, TypeError) as e:
            return str(e)

    # split 2 col into 3 cols
    res = batch_map_config(3, 3, split_col, ["col1", "col_x2", "col_y2"])[0]
    assert np.array_equal(res["col1"], [[1], [2], [3]])
    assert np.array_equal(res["col_x2"], [[1], [4], [9]]) and np.array_equal(res["col_y2"], [[-1], [-4], [-9]])


def test_batch_exception_18():
    """
    Feature: Batch op
    Description: Test batch with parameter column_order
    Expectation: Output is equal to the expected output
    """
    def gen(num):
        for i in range(num):
            if i % 2 == 0:
                yield (np.array([i]), np.array([i + (1 + i) * 0.01]))
            else:
                yield (np.array([(i + 1) * 0.01 + i]), np.array([i]))

    def swap_col(col1, col2, batch_info):
        return ([np.copy(a) for a in col2], [np.copy(b) for b in col1])

    logger.info("test_batch_exception_18")

    batch_size = 4
    input_columns = ["num1", "num2"]
    data1 = ds.GeneratorDataset((lambda: gen(20)), input_columns)
    with pytest.raises(TypeError) as raise_info:
        result = data1.batch(batch_size=batch_size, per_batch_map=swap_col, column_order=input_columns)
        for _ in result.create_dict_iterator(num_epochs=1, output_numpy=True):
            pass
    assert "got an unexpected keyword argument 'column_order'" in str(raise_info.value)


def test_batch_exception_19():
    """
    Feature: Batch op
    Description: Test batch with parameter pad_info
    Expectation: Output is equal to the expected output
    """
    data_dir_coco = "../data/dataset/testCOCO/train/"
    annotation_file_coco = "../data/dataset/testCOCO/annotations/train.json"
    data1 = ds.CocoDataset(data_dir_coco, annotation_file=annotation_file_coco, task="Detection", decode=True)
    data1 = data1.shuffle(10)
    with pytest.raises(TypeError) as raise_info:
        data1 = data1.batch(3, pad_info={})
        num_iter = 0
        for _ in data1.create_dict_iterator(num_epochs=1):
            num_iter += 1
        assert num_iter == 2
    assert "got an unexpected keyword argument 'pad_info'" in str(raise_info.value)


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
    test_batch_13()
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
    test_batch_exception_16()
    test_batch_exception_17()
    test_no_input_columns_01()
    test_no_input_columns_02()
    test_batch_exception_18()
    test_batch_exception_19()
    logger.info('\n')
