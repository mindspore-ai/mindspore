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
import numpy as np
import pytest

import mindspore.dataset as ds
from mindspore import log as logger

DATA_DIR = ["../data/dataset/testPyfuncMap/data.data"]
SCHEMA_DIR = "../data/dataset/testPyfuncMap/schema.json"
COLUMNS = ["col0", "col1", "col2"]
GENERATE_GOLDEN = False


def test_case_0():
    """
    Test PyFunc
    """
    logger.info("Test 1-1 PyFunc : lambda x : x + x")

    # apply dataset operations
    data1 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, shuffle=False)

    data1 = data1.map(input_columns="col0", output_columns="out", operations=(lambda x: x + x))

    i = 0
    for item in data1.create_dict_iterator():  # each data is a dictionary
        # In this test, the dataset is 2x2 sequential tensors
        golden = np.array([[i * 2, (i + 1) * 2], [(i + 2) * 2, (i + 3) * 2]])
        assert np.array_equal(item["out"], golden)
        i = i + 4


def test_case_1():
    """
    Test PyFunc
    """
    logger.info("Test 1-n PyFunc : lambda x : (x , x + x) ")

    col = "col0"

    # apply dataset operations
    data1 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, shuffle=False)
    data1 = data1.map(input_columns=col, output_columns=["out0", "out1"], operations=(lambda x: (x, x + x)),
                  columns_order=["out0", "out1"])

    i = 0
    for item in data1.create_dict_iterator():  # each data is a dictionary
        # In this test, the dataset is 2x2 sequential tensors
        golden = np.array([[i, i + 1], [i + 2, i + 3]])
        assert np.array_equal(item["out0"], golden)
        golden = np.array([[i * 2, (i + 1) * 2], [(i + 2) * 2, (i + 3) * 2]])
        assert np.array_equal(item["out1"], golden)
        i = i + 4


def test_case_2():
    """
    Test PyFunc
    """
    logger.info("Test n-1 PyFunc : lambda x, y : x + y ")

    col = ["col0", "col1"]

    # apply dataset operations
    data1 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, shuffle=False)

    data1 = data1.map(input_columns=col, output_columns="out", operations=(lambda x, y: x + y),
                  columns_order=["out"])

    i = 0
    for item in data1.create_dict_iterator():  # each data is a dictionary
        # In this test, the dataset is 2x2 sequential tensors
        golden = np.array([[i * 2, (i + 1) * 2], [(i + 2) * 2, (i + 3) * 2]])
        assert np.array_equal(item["out"], golden)
        i = i + 4


def test_case_3():
    """
    Test PyFunc
    """
    logger.info("Test n-m PyFunc : lambda x, y : (x , x + 1, x + y)")

    col = ["col0", "col1"]

    # apply dataset operations
    data1 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, shuffle=False)

    data1 = data1.map(input_columns=col, output_columns=["out0", "out1", "out2"],
                  operations=(lambda x, y: (x, x + y, x + y + 1)), columns_order=["out0", "out1", "out2"])

    i = 0
    for item in data1.create_dict_iterator():  # each data is a dictionary
        # In this test, the dataset is 2x2 sequential tensors
        golden = np.array([[i, i + 1], [i + 2, i + 3]])
        assert np.array_equal(item["out0"], golden)
        golden = np.array([[i * 2, (i + 1) * 2], [(i + 2) * 2, (i + 3) * 2]])
        assert np.array_equal(item["out1"], golden)
        golden = np.array([[i * 2 + 1, (i + 1) * 2 + 1], [(i + 2) * 2 + 1, (i + 3) * 2 + 1]])
        assert np.array_equal(item["out2"], golden)
        i = i + 4


def test_case_4():
    """
    Test PyFunc
    """
    logger.info("Test Parallel n-m PyFunc : lambda x, y : (x , x + 1, x + y)")

    col = ["col0", "col1"]

    # apply dataset operations
    data1 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, shuffle=False)

    data1 = data1.map(input_columns=col, output_columns=["out0", "out1", "out2"], num_parallel_workers=4,
                  operations=(lambda x, y: (x, x + y, x + y + 1)), columns_order=["out0", "out1", "out2"])

    i = 0
    for item in data1.create_dict_iterator():  # each data is a dictionary
        # In this test, the dataset is 2x2 sequential tensors
        golden = np.array([[i, i + 1], [i + 2, i + 3]])
        assert np.array_equal(item["out0"], golden)
        golden = np.array([[i * 2, (i + 1) * 2], [(i + 2) * 2, (i + 3) * 2]])
        assert np.array_equal(item["out1"], golden)
        golden = np.array([[i * 2 + 1, (i + 1) * 2 + 1], [(i + 2) * 2 + 1, (i + 3) * 2 + 1]])
        assert np.array_equal(item["out2"], golden)
        i = i + 4


# The execution of this function will acquire GIL
def func_5(x):
    return np.ones(x.shape, dtype=x.dtype)


def test_case_5():
    """
    Test PyFunc
    """
    logger.info("Test 1-1 PyFunc : lambda x: np.ones(x.shape)")

    # apply dataset operations
    data1 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, shuffle=False)

    data1 = data1.map(input_columns="col0", output_columns="out", operations=func_5)

    for item in data1.create_dict_iterator():  # each data is a dictionary
        # In this test, the dataset is 2x2 sequential tensors
        golden = np.array([[1, 1], [1, 1]])
        assert np.array_equal(item["out"], golden)


def test_case_6():
    """
    Test PyFunc
    """
    logger.info("Test PyFunc ComposeOp : (lambda x : x + x), (lambda x : x + x)")

    # apply dataset operations
    data1 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, shuffle=False)

    data1 = data1.map(input_columns="col0", output_columns="out",
                  operations=[(lambda x: x + x), (lambda x: x + x)])

    i = 0
    for item in data1.create_dict_iterator():  # each data is a dictionary
        # In this test, the dataset is 2x2 sequential tensors
        golden = np.array([[i * 4, (i + 1) * 4], [(i + 2) * 4, (i + 3) * 4]])
        assert np.array_equal(item["out"], golden)
        i = i + 4


def test_case_7():
    """
    Test PyFunc
    """
    logger.info("Test 1-1 PyFunc Multiprocess: lambda x : x + x")

    # apply dataset operations
    data1 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, shuffle=False)

    data1 = data1.map(input_columns="col0", output_columns="out", operations=(lambda x: x + x),
                      num_parallel_workers=4, python_multiprocessing = True)

    i = 0
    for item in data1.create_dict_iterator():  # each data is a dictionary
        # In this test, the dataset is 2x2 sequential tensors
        golden = np.array([[i * 2, (i + 1) * 2], [(i + 2) * 2, (i + 3) * 2]])
        assert np.array_equal(item["out"], golden)
        i = i + 4


def test_case_8():
    """
    Test PyFunc
    """
    logger.info("Test Multiprocess n-m PyFunc : lambda x, y : (x , x + 1, x + y)")

    col = ["col0", "col1"]

    # apply dataset operations
    data1 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, shuffle=False)

    data1 = data1.map(input_columns=col, output_columns=["out0", "out1", "out2"], num_parallel_workers=4,
                      operations=(lambda x, y: (x, x + y, x + y + 1)), columns_order=["out0", "out1", "out2"],
                      python_multiprocessing=True)

    i = 0
    for item in data1.create_dict_iterator():  # each data is a dictionary
        # In this test, the dataset is 2x2 sequential tensors
        golden = np.array([[i, i + 1], [i + 2, i + 3]])
        assert np.array_equal(item["out0"], golden)
        golden = np.array([[i * 2, (i + 1) * 2], [(i + 2) * 2, (i + 3) * 2]])
        assert np.array_equal(item["out1"], golden)
        golden = np.array([[i * 2 + 1, (i + 1) * 2 + 1], [(i + 2) * 2 + 1, (i + 3) * 2 + 1]])
        assert np.array_equal(item["out2"], golden)
        i = i + 4


def test_case_9():
    """
    Test PyFunc
    """
    logger.info("Test multiple 1-1 PyFunc Multiprocess: lambda x : x + x")

    # apply dataset operations
    data1 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, shuffle=False)

    data1 = data1.map(input_columns="col0", output_columns="out", operations=[(lambda x: x + x), (lambda x: x + 1),
                                                                              (lambda x: x + 2)],
                      num_parallel_workers=4, python_multiprocessing=True)

    i = 0
    for item in data1.create_dict_iterator():  # each data is a dictionary
        # In this test, the dataset is 2x2 sequential tensors
        golden = np.array([[i * 2 + 3, (i + 1) * 2 + 3], [(i + 2) * 2 + 3, (i + 3) * 2 + 3]])
        assert np.array_equal(item["out"], golden)
        i = i + 4


def test_pyfunc_execption():
    logger.info("Test PyFunc Execption Throw: lambda x : raise Execption()")

    def pyfunc(x):
        raise Exception("Pyfunc Throw")

    with pytest.raises(RuntimeError) as info:
        # apply dataset operations
        data1 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, shuffle=False)
        data1 = data1.map(input_columns="col0", output_columns="out", operations= pyfunc,
                          num_parallel_workers=4)
        for _ in data1:
            pass
        assert "Pyfunc Throw" in str(info.value)


def test_pyfunc_execption_multiprocess():
    logger.info("Test Multiprocess PyFunc Execption Throw: lambda x : raise Execption()")

    def pyfunc(x):
        raise Exception("MP Pyfunc Throw")

    with pytest.raises(RuntimeError) as info:
        # apply dataset operations
        data1 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, shuffle=False)
        data1 = data1.map(input_columns="col0", output_columns="out", operations= pyfunc,
                          num_parallel_workers=4, python_multiprocessing = True)
        for _ in data1:
            pass
        assert "MP Pyfunc Throw" in str(info.value)


if __name__ == "__main__":
    test_case_0()
    test_case_1()
    test_case_2()
    test_case_3()
    test_case_4()
    test_case_5()
    test_case_6()
    test_case_7()
    test_case_8()
    test_case_9()
    test_pyfunc_execption()
    test_pyfunc_execption_multiprocess()
