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
"""Test Angle."""

import numpy as np
import pytest

import mindspore.dataset as ds
from mindspore.dataset import audio
from . import count_unequal_element


def test_angle_eager():
    """
    Feature: Angle
    Description: Test Angle in eager mode
    Expectation: Output is equal to the expected value
    """
    arr = np.array([[73.04, -13.00], [57.49, 13.20], [-57.64, 6.51], [-52.25, 30.67], [-30.11, -18.34],
                    [-63.32, 99.33], [95.82, -24.76]], dtype=np.double)
    expected = np.array([-0.17614017, 0.22569334, 3.02912684, 2.6107975, -2.59450886, 2.13831337, -0.25286988],
                        dtype=np.double)
    angle = audio.Angle()
    output = angle(arr)
    count_unequal_element(expected, output, 0.0001, 0.0001)


def test_angle_pipeline():
    """
    Feature: Angle
    Description: Test Angle in pipeline mode
    Expectation: Output is equal to the expected value
    """
    np.random.seed(6)
    arr = np.array([[[84.25, -85.92], [-92.23, 23.06], [-7.33, -44.17], [-62.95, -14.73]],
                    [[93.09, 38.18], [-81.94, 71.34], [71.33, -39.00], [95.25, -32.94]]], dtype=np.double)
    expected = np.array([[-0.79521156, 2.89658848, -1.73524737, -2.91173309],
                         [0.3892177, 2.42523905, -0.50034807, -0.33295219]], dtype=np.double)
    label = np.random.sample((2, 4, 1))
    data = (arr, label)
    dataset = ds.NumpySlicesDataset(
        data, column_names=["col1", "col2"], shuffle=False)
    angle = audio.Angle()
    dataset = dataset.map(operations=angle, input_columns=["col1"])
    for item1, item2 in zip(dataset.create_dict_iterator(num_epochs=1, output_numpy=True), expected):
        count_unequal_element(item2, item1['col1'], 0.0001, 0.0001)


def test_angle_exception():
    """
    Feature: Angle
    Description: Test Angle in pipeline mode with invalid input data type
    Expectation: Error is raised as expected
    """
    np.random.seed(78)
    arr = np.array([["11", "22"], ["33", "44"], ["55", "66"], ["77", "88"]])
    label = np.random.sample((4, 1))
    data = (arr, label)
    dataset = ds.NumpySlicesDataset(
        data, column_names=["col1", 'col2'], shuffle=False)
    angle = audio.Angle()
    dataset = dataset.map(operations=angle, input_columns=["col1"])
    num_itr = 0
    with pytest.raises(RuntimeError, match="the data type of input tensor does not match the requirement of operator"):
        for _ in dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
            num_itr += 1


def test_angle_transform():
    """
    Feature: Angle
    Description: Test Angle with various valid input data types and shapes
    Expectation: The operation completes successfully and output shapes are correct
    """
    # test angle normal with type is float
    arr = np.random.uniform(-100, 100, size=(7, 2)).astype(np.double)
    label = np.random.sample((7, 1))
    data = (arr, label)
    dataset = ds.NumpySlicesDataset(data, column_names=["col1", "col2"],
                                    shuffle=False)
    angle = audio.Angle()
    dataset = dataset.map(operations=angle, input_columns=["col1"])
    for _ in dataset.create_dict_iterator(output_numpy=True):
        pass

    # test angle normal with type is double
    arr = np.random.uniform(-100, 100, size=(4, 2)).astype(np.double)
    label = np.random.sample((4, 1))
    data = (arr, label)
    dataset = ds.NumpySlicesDataset(data, column_names=["col1", 'col2'],
                                    shuffle=False)
    angle = audio.Angle()
    dataset = dataset.map(operations=angle, input_columns=["col1"])
    for _ in dataset.create_dict_iterator(output_numpy=True):
        pass

    # test angle normal with type is string
    arr = np.array([["11", "22"], ["33", "44"], ["55", "66"], ["77", "88"]])
    label = np.random.sample((4, 1))
    data = (arr, label)
    dataset = ds.NumpySlicesDataset(data, column_names=["col1", 'col2'],
                                    shuffle=False)
    with pytest.raises(RuntimeError,
                       match=r"Expecting tensor in type of \[int, float, double\]. But got type string."):
        angle = audio.Angle()
        dataset = dataset.map(operations=angle, input_columns=["col1"])
        for _ in dataset.create_dict_iterator(output_numpy=True):
            pass

    # test error angle, The last dimension is 1
    arr = np.array([[1], [2]], dtype=np.double)
    label = np.random.sample((2, 1))
    data = (arr, label)
    dataset = ds.NumpySlicesDataset(data, column_names=["col1", 'col2'],
                                    shuffle=False)
    with pytest.raises(RuntimeError,
                       match="Expecting tensor in shape of <..., complex=2>"):
        angle = audio.Angle()
        dataset = dataset.map(operations=angle, input_columns=["col1"])
        for _ in dataset.create_dict_iterator(output_numpy=True):
            pass

    # Test angle input data is empty
    arr = np.random.uniform(-100, 100, size=(0, 2)).astype(np.float64)
    angle = audio.Angle()
    with pytest.raises(RuntimeError, match="Input Tensor has no data."):
        angle(arr)

    # Test angle input data only one data
    arr = np.random.uniform(-100, 100, size=(1, 2)).astype(np.float64)
    angle = audio.Angle()
    out = angle(arr)
    num_data = 0
    for _ in out:
        num_data += 1
    assert num_data == 1

    # Test angle input data input data has multiple data.
    arr = np.random.uniform(-100, 100, size=(50, 2)).astype(np.float64)
    angle = audio.Angle()
    out = angle(arr)
    assert out.shape == (50,)

    # Test angle input data type is float32.
    arr = np.random.uniform(-100, 100, size=(30, 2)).astype(np.float32)
    angle = audio.Angle()
    out = angle(arr)
    assert out.shape == (30,)

    # Test angle input data type is bool.
    arr = np.random.uniform(-100, 100, size=(30, 2)).astype(np.bool_)
    angle = audio.Angle()
    out = angle(arr)
    assert out.shape == (30,)

    # Test angle input data type is int32.
    arr = np.random.uniform(-100, 100, size=(30, 2)).astype(np.int32)
    angle = audio.Angle()
    out = angle(arr)
    assert out.shape == (30,)

    # Test angle input data type is int16.
    arr = np.random.uniform(-100, 100, size=(30, 2)).astype(np.int16)
    angle = audio.Angle()
    out = angle(arr)
    assert out.shape == (30,)

    # Test angle input data type is float.
    arr = np.random.uniform(-100, 100, size=(30, 2)).astype(np.double)
    angle = audio.Angle()
    out = angle(arr)
    assert out.shape == (30,)

    # The input data is the correct shape value.
    arr = np.random.randn(4, 8, 2)
    angle = audio.Angle()
    out = angle(arr)
    assert out.shape == (4, 8)

    # The input data is abnormal.
    arr = np.random.randn(4, 8, 1)
    angle = audio.Angle()
    with pytest.raises(RuntimeError,
                       match="Expecting tensor in shape of <..., complex=2>"):
        angle(arr)

    # The input data is abnormal.
    arr = np.array(["a", "b", "c"])
    angle = audio.Angle()
    with pytest.raises(RuntimeError,
                       match="Expecting tensor in shape of <..., complex=2>"):
        angle(arr)

    # The input data is abnormal.
    arr = np.random.randn(2, 9, 5)
    angle = audio.Angle()
    with pytest.raises(RuntimeError,
                       match="Expecting tensor in shape of <..., complex=2>"):
        angle(arr)

    # The input data is one dimension.
    arr = np.random.uniform(-100, 100, size=2).astype(np.float64)
    label = np.random.random_sample(2)
    data = (arr, label)
    dataset = ds.NumpySlicesDataset(data, column_names=["col1", "col2"],
                                    shuffle=False)
    with pytest.raises(RuntimeError,
                       match="Expecting tensor in shape of <..., complex=2>"):
        angle = audio.Angle()
        dataset = dataset.map(operations=angle, input_columns=["col1"])
        for _ in dataset.create_dict_iterator(output_numpy=True):
            pass


if __name__ == "__main__":
    test_angle_eager()
    test_angle_pipeline()
    test_angle_exception()
    test_angle_transform()
