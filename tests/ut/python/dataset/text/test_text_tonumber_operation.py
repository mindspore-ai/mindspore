# Copyright 2025 Huawei Technologies Co., Ltd
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
"""text transform - tonumber"""

import os
import numpy as np
import pytest
import mindspore.dataset as ds
import mindspore.dataset.text as nlp
from mindspore.common import dtype as mstype


TEST_DATA_DATASET_FUNC ="../data/dataset/"


DATA_FILE = os.path.join(TEST_DATA_DATASET_FUNC, "text_data/testTextFile/textfile/testToNumber/number.txt")
NEGATIVE_DATA_FILE = os.path.join(TEST_DATA_DATASET_FUNC,
                                  "text_data/testTextFile/textfile/testToNumber/negative_number.txt")


def test_tonumber_operation_01():
    """
    Feature: ToNumber op
    Description: Test ToNumber op with different data types (float, int) and input sources
    Expectation: Successfully convert string numbers to specified numeric types
    """
    # file: DATA_FILE
    dataset = ds.TextFileDataset(DATA_FILE, shuffle=False)
    op_tonumber = nlp.ToNumber(mstype.float32)
    dataset = dataset.map(operations=op_tonumber)
    num_iter = 0
    for _ in dataset.create_dict_iterator(output_numpy=True):
        num_iter += 1
        # print(data)
    assert num_iter == 6

    # file: DATA_FILE
    dataset = ds.TextFileDataset(DATA_FILE, shuffle=False)
    op_tonumber = nlp.ToNumber(mstype.float64)
    dataset = dataset.map(operations=op_tonumber)
    num_iter = 0
    for _ in dataset.create_dict_iterator(output_numpy=True):
        num_iter += 1
        # print(data)
    assert num_iter == 6

    # test:  -1
    dataset = ds.TextFileDataset(NEGATIVE_DATA_FILE, shuffle=False)
    op_tonumber = nlp.ToNumber(mstype.int8)
    dataset = dataset.map(operations=op_tonumber)
    num_iter = 0
    for _ in dataset.create_dict_iterator(output_numpy=True):
        num_iter += 1
        # print(data)
    assert num_iter == 4

    # file: NEGATIVE_DATA_FILE
    dataset = ds.TextFileDataset(NEGATIVE_DATA_FILE, shuffle=False)
    op_tonumber = nlp.ToNumber(mstype.int32)
    dataset = dataset.map(operations=op_tonumber)
    num_iter = 0
    for _ in dataset.create_dict_iterator(output_numpy=True):
        num_iter += 1
        # print(data)
    assert num_iter == 4

    # file: DATA_FILE
    dataset = ds.TextFileDataset(DATA_FILE, shuffle=False)
    op_tonumber = nlp.ToNumber(mstype.uint32)
    dataset = dataset.map(operations=op_tonumber)
    num_iter = 0
    for _ in dataset.create_dict_iterator(output_numpy=True):
        num_iter += 1
        # print(data)
    assert num_iter == 6

    # data: "123"
    data = "123"
    op_tonumber = nlp.ToNumber(mstype.uint32)
    res = op_tonumber(data)
    assert res == np.array(123, dtype=np.uint32)

    # data: "0.0015"
    data = "0.0015"
    op_tonumber = nlp.ToNumber(mstype.float32)
    res = op_tonumber(data)
    assert res == np.array(0.0015, dtype=np.float32)

    # data: "1234"
    data = "1234"
    op_tonumber = nlp.ToNumber(mstype.float16)
    res = op_tonumber(data)
    assert res == np.array(1234, dtype=np.float16)

    # data: "1234"
    data = "1234"
    op_tonumber = nlp.ToNumber(mstype.float64)
    res = op_tonumber(data)
    assert res == np.array(1234, dtype=np.float64)

    # data: "1234"
    data = "1234"
    op_tonumber = nlp.ToNumber(mstype.uint64)
    res = op_tonumber(data)
    assert res == np.array(1234, dtype=np.uint64)

    # data: "123"
    data = "123"
    op_tonumber = nlp.ToNumber(mstype.int8)
    res = op_tonumber(data)
    assert res == np.array(123, dtype=np.int8)

    # data: ["123","456"]
    data = ["123", "456"]
    result = []
    for i in data:
        op = nlp.ToNumber(mstype.int64)
        result.append(op(i))
    assert result == [np.array(123, dtype=np.int64), np.array(456, dtype=np.int64)]


def test_tonumber_operation_02():
    """
    Feature: ToNumber op
    Description: Test ToNumber op with list and multi-dimensional array inputs
    Expectation: Successfully convert string arrays to numeric arrays
    """
    # data: ["123","456"]
    input_strings = ["123", "234", "345"]
    result = []
    for i in input_strings:
        op = nlp.ToNumber(mstype.float32)
        result.append(op(i))
    assert result == [np.array(123., dtype=np.float32), np.array(234., dtype=np.float32),
                      np.array(345., dtype=np.float32)]

    # data: ["123","456"]
    data = ["123", "456"]
    op = nlp.ToNumber(mstype.int64)
    res = op(data)
    assert np.array_equal(res, np.array(["123", "456"], dtype=np.int64))

    # data: [["1", "2", "3"], ["4", "5", "6"]]
    data = [["1", "2", "3"], ["4", "5", "6"]]
    op = nlp.ToNumber(mstype.int8)
    res = op(data)
    assert np.array_equal(res, np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int8))


def test_tonumber_exception_01():
    """
    Feature: ToNumber op
    Description: Test ToNumber op with out-of-range values and invalid parameters
    Expectation: Raise expected exceptions for values exceeding data type limits
    """
    # file: DATA_FILE
    dataset = ds.TextFileDataset(DATA_FILE, shuffle=False)
    op_tonumber = nlp.ToNumber(mstype.float16)
    dataset = dataset.map(operations=op_tonumber)
    num_iter = 0
    with pytest.raises(RuntimeError, match=r"outside of valid float16 range \[65504.000000, -65504.000000\]"):
        for _ in dataset.create_dict_iterator(output_numpy=True):
            num_iter += 1
            # print(data)
        assert num_iter == 6

    # test: 3456789 out of range
    dataset = ds.TextFileDataset(DATA_FILE, shuffle=False)
    op_tonumber = nlp.ToNumber(mstype.int8)
    dataset = dataset.map(operations=op_tonumber)
    with pytest.raises(RuntimeError, match=r"out of bounds if cast to int8. The valid range is: \[-128, 127\]"):
        for data in dataset.create_dict_iterator(output_numpy=True):
            print(data)

    # file: NEGATIVE_DATA_FILE
    dataset = ds.TextFileDataset(NEGATIVE_DATA_FILE, shuffle=False)
    op_tonumber = nlp.ToNumber(mstype.uint16)
    dataset = dataset.map(operations=op_tonumber)
    with pytest.raises(RuntimeError, match=r"out of bounds if cast to uint16. The valid range is: \[0, 65535\]"):
        for data in dataset.create_dict_iterator(output_numpy=True):
            print(data)

    # file: DATA_FILE
    with pytest.raises(TypeError, match="missing a required argument: 'data_type'"):
        dataset = ds.TextFileDataset(DATA_FILE, shuffle=False)
        op_tonumber = nlp.ToNumber()
        dataset = dataset.map(operations=op_tonumber)
        for data in dataset.create_dict_iterator(output_numpy=True):
            print(data)

    # file: DATA_FILE
    with pytest.raises(TypeError, match="Argument data_type with value \\<class 'int'\\> is not of type"):
        dataset = ds.TextFileDataset(DATA_FILE, shuffle=False)
        op_tonumber = nlp.ToNumber(int)
        dataset = dataset.map(operations=op_tonumber)
        for data in dataset.create_dict_iterator(output_numpy=True):
            print(data)

    # data: "1234"
    data = "1234"
    op_tonumber = nlp.ToNumber(mstype.int8)
    with pytest.raises(RuntimeError, match=r"string input 1234 will be out of bounds if cast to int8. The valid "
                                           r"range is: \[-128, 127\]"):
        _ = op_tonumber(data)

    # data: "123"
    with pytest.raises(TypeError, match="data_type: Bool is not numeric data type"):
        _ = nlp.ToNumber(mstype.bool_)
