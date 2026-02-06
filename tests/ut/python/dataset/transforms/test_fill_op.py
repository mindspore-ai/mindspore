# Copyright 2020-2022 Huawei Technologies Co., Ltd
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
Testing fill op
"""
import numpy as np
import pytest
import mindspore.dataset as ds
import mindspore.dataset.transforms as data_trans


def test_fillop_basic():
    """
    Feature: Fill op
    Description: Test Fill op basic usage (positive int onto an array of uint8)
    Expectation: Output is equal to the expected output
    """

    def gen():
        yield (np.array([4, 5, 6, 7], dtype=np.uint8),)

    data = ds.GeneratorDataset(gen, column_names=["col"])
    fill_op = data_trans.Fill(3)

    data = data.map(operations=fill_op, input_columns=["col"])
    expected = np.array([3, 3, 3, 3], dtype=np.uint8)
    for data_row in data:
        np.testing.assert_array_equal(data_row[0].asnumpy(), expected)


def test_fillop_down_type_cast():
    """
    Feature: Fill op
    Description: Test Fill op with a negative number onto an array of unsigned int8
    Expectation: Output is equal to the expected output
    """

    def gen():
        yield (np.array([4, 5, 6, 7], dtype=np.uint8),)

    data = ds.GeneratorDataset(gen, column_names=["col"])
    fill_op = data_trans.Fill(-3)

    data = data.map(operations=fill_op, input_columns=["col"])
    expected = np.array([253, 253, 253, 253], dtype=np.uint8)
    for data_row in data:
        np.testing.assert_array_equal(data_row[0].asnumpy(), expected)


def test_fillop_up_type_cast():
    """
    Feature: Fill op
    Description: Test Fill op with a int onto an array of floats
    Expectation: Output is equal to the expected output
    """

    def gen():
        yield (np.array([4, 5, 6, 7], dtype=float),)

    data = ds.GeneratorDataset(gen, column_names=["col"])
    fill_op = data_trans.Fill(3)

    data = data.map(operations=fill_op, input_columns=["col"])
    expected = np.array([3., 3., 3., 3.], dtype=float)
    for data_row in data:
        np.testing.assert_array_equal(data_row[0].asnumpy(), expected)


def test_fillop_string():
    """
    Feature: Fill op
    Description: Test Fill op with a string onto an array of strings
    Expectation: Output is equal to the expected output
    """

    def gen():
        yield (np.array(["45555", "45555"], dtype=np.str_),)

    data = ds.GeneratorDataset(gen, column_names=["col"])
    fill_op = data_trans.Fill("error")

    data = data.map(operations=fill_op, input_columns=["col"])
    expected = np.array(['error', 'error'], dtype=np.str_)
    for data_row in data.create_tuple_iterator(num_epochs=1, output_numpy=True):
        np.testing.assert_array_equal(data_row[0], expected)


def test_fillop_bytes():
    """
    Feature: Fill op
    Description: Test Fill op with bytes onto an array of strings
    Expectation: Output is equal to the expected output
    """

    def gen():
        yield (np.array(["A", "B", "C"], dtype=np.bytes_),)

    data = ds.GeneratorDataset(gen, column_names=["col"])
    fill_op = data_trans.Fill(b'abc')

    data = data.map(operations=fill_op, input_columns=["col"])
    expected = np.array([b'abc', b'abc', b'abc'], dtype=np.bytes_)
    for data_row in data.create_tuple_iterator(num_epochs=1, output_numpy=True):
        np.testing.assert_array_equal(data_row[0], expected)


def test_fillop_error_handling():
    """
    Feature: Fill op
    Description: Test Fill op with a mismatch data type (string onto an array of ints)
    Expectation: Error is raised as expected
    """

    def gen():
        yield (np.array([4, 4, 4, 4]),)

    data = ds.GeneratorDataset(gen, column_names=["col"])
    fill_op = data_trans.Fill("words")
    data = data.map(operations=fill_op, input_columns=["col"])

    with pytest.raises(RuntimeError) as error_info:
        for _ in data:
            pass
    assert "fill_value and the input tensor must be of the same data type" in str(error_info.value)


if __name__ == "__main__":
    test_fillop_basic()
    test_fillop_up_type_cast()
    test_fillop_down_type_cast()
    test_fillop_string()
    test_fillop_bytes()
    test_fillop_error_handling()
