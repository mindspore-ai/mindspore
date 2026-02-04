# Copyright 2021-2025 Huawei Technologies Co., Ltd
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
"""Test Magphase."""

import numpy as np
import pytest

import mindspore.dataset as ds
from mindspore.dataset import audio


def test_magphase_pipeline():
    """
    Feature: Magphase
    Description: Test Magphase in pipeline mode
    Expectation: Output is equal to the expected output
    """
    data = [[[3.0, -4.0], [-5.0, 12.0]]]
    expected = [5, 13, -0.927295, 1.965587]
    dataset = ds.NumpySlicesDataset(data, column_names=["col1"], shuffle=False)
    magphase_window = audio.Magphase(power=1.0)
    dataset = dataset.map(operations=magphase_window, input_columns=["col1"],
                          output_columns=["mag", "phase"])
    for column1, column2 in dataset.create_tuple_iterator(num_epochs=1, output_numpy=True):
        assert abs(column1[0] - expected[0]) < 0.00001
        assert abs(column1[1] - expected[1]) < 0.00001
        assert abs(column2[0] - expected[2]) < 0.00001
        assert abs(column2[1] - expected[3]) < 0.00001


def test_magphase_eager():
    """
    Feature: Magphase
    Description: Test Magphase in eager mode
    Expectation: Output is equal to the expected output
    """
    data = np.array([41, 67, 34, 0, 69, 24, 78, 58]).reshape((2, 2, 2)).astype("double")
    mag = np.array([78.54934755, 34., 73.05477397, 97.20082304]).reshape((2, 2)).astype("double")
    phase = np.array([1.02164342, 0, 0.33473684, 0.63938591]).reshape((2, 2)).astype("double")
    magphase_window = audio.Magphase()
    data1, data2 = magphase_window(data)
    assert (abs(data1 - mag) < 0.00001).all()
    assert (abs(data2 - phase) < 0.00001).all()


def test_magphase_exception():
    """
    Feature: Magphase
    Description: Test Magphase with invalid input
    Expectation: Correct error is raised as expected
    """
    try:
        data = np.array([1, 2, 3, 4]).reshape(4, ).astype("double")
        magphase_window = audio.Magphase(power=2.0)
        magphase_window(data)
    except RuntimeError as error:
        assert "the shape of input tensor does not match the requirement of operator" in str(error)
    try:
        data = np.array([1, 2, 3, 4]).reshape(1, 4).astype("double")
        magphase_window = audio.Magphase(power=2.0)
        magphase_window(data)
    except RuntimeError as error:
        assert "the shape of input tensor does not match the requirement of operator" in str(error)
    try:
        data = np.array(['test', 'test']).reshape(1, 2)
        magphase_window = audio.Magphase(power=2.0)
        magphase_window(data)
    except RuntimeError as error:
        assert "the data type of input tensor does not match the requirement of operator" in str(error)
    try:
        data = np.array([1, 2, 3, 4]).reshape(2, 2).astype("double")
        magphase_window = audio.Magphase(power=-1.0)
        magphase_window(data)
    except ValueError as error:
        assert "Input power is not within the required interval of [0, 16777216]." in str(error)


def test_magphase_transform():
    """
    Feature: Magphase
    Description: Test Magphase with various valid input parameters and data types
    Expectation: The operation completes successfully
    """

    data = np.array([41, 67, 34, 0, 69, 24, 78, 58]).reshape((2, 2, 2)).astype("float")
    magphase = audio.Magphase()
    mag, phase = magphase(data)
    assert np.shape(mag) == (2, 2)
    assert np.shape(phase) == (2, 2)

    # test
    data = np.array([41, 67, 34, 0, 69, 24, 78, 58]).reshape((2, 2, 2)).astype("float")
    magphase = audio.Magphase(2)
    mag, phase = magphase(data)
    assert np.shape(mag) == (2, 2)
    assert np.shape(phase) == (2, 2)


def test_magphase_param_check():
    """
    Feature: Magphase
    Description: Test Magphase with invalid input data types and parameters
    Expectation: Correct error types and messages are raised as expected
    """

    # Test with input tensor shape mismatch (1D array)
    magphase = audio.Magphase()
    with pytest.raises(RuntimeError, match=".*Magphase: the shape of input tensor does not match"
                                           " the requirement of operator. Expecting tensor in "
                                           "shape of <..., complex=2>."):
        data = np.array([1, 2, 3, 4]).reshape(4, ).astype("double")
        magphase(data)

    # Test with input tensor shape mismatch (last dimension is not 2)
    magphase = audio.Magphase()
    data = np.array([1, 2, 3, 4]).reshape(1, 4).astype("double")
    with pytest.raises(RuntimeError, match=".*Magphase: the shape of input tensor does not match"
                                           " the requirement of operator. Expecting tensor in "
                                           "shape of <..., complex=2>."):
        magphase(data)

    # Test with invalid input data type (string type)
    data = np.array(['test', 'test']).reshape(1, 2)
    magphase = audio.Magphase()
    with pytest.raises(RuntimeError, match=".*Magphase: the data type of input tensor does not match "
                                           "the requirement of operator. Expecting tensor in type "
                                           "of .*int, float, double.*. But got type string."):
        magphase(data)

    # Test with negative power parameter
    with pytest.raises(ValueError, match=r"Input power is not within the required interval of \[0, 16777216\]"):
        audio.Magphase(-1)

    # Test with invalid power parameter type (string type)
    with pytest.raises(TypeError, match=r"Argument power with value 1 is not of type "
                                        r"\[<class 'int'>, <class 'float'>\], but got <class 'str'>"):
        audio.Magphase("1")

    # Test with power parameter out of valid range (greater than maximum value)
    with pytest.raises(ValueError, match=r"Input power is not within the required interval of \[0, 16777216\]"):
        audio.Magphase(100000000)


if __name__ == "__main__":
    test_magphase_pipeline()
    test_magphase_eager()
    test_magphase_exception()
    test_magphase_transform()
    test_magphase_param_check()
