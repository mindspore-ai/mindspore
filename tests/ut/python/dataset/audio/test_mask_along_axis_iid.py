# Copyright 2022-2025 Huawei Technologies Co., Ltd
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
"""Test MaskAlongAxisIID."""

import copy
import numpy as np
import pytest

import mindspore.dataset as ds
from mindspore.dataset import audio
from . import count_unequal_element

BATCH = 2
CHANNEL = 2
FREQ = 10
TIME = 10


def allclose_nparray(data_expected, data_me, rtol, atol, equal_nan=True):
    """
    Precision calculation formula
    """
    if np.any(np.isnan(data_expected)):
        assert np.allclose(data_me, data_expected, rtol, atol, equal_nan=equal_nan)
    elif not np.allclose(data_me, data_expected, rtol, atol, equal_nan=equal_nan):
        count_unequal_element(data_expected, data_me, rtol, atol)


def gen(shape):
    np.random.seed(0)
    data = np.random.random(shape)
    yield (np.array(data, dtype=np.float32),)


def test_mask_along_axis_iid_eager():
    """
    Feature: MaskAlongAxisIID
    Description: Mindspore eager mode with normal testcase
    Expectation: The returned result is as expected
    """
    spectrogram = next(gen((BATCH, CHANNEL, FREQ, TIME)))[0]
    output = audio.MaskAlongAxisIID(mask_param=8, mask_value=5.0, axis=1)(spectrogram)
    assert output.shape == (BATCH, CHANNEL, FREQ, TIME)

    spectrogram = next(gen((BATCH, CHANNEL, FREQ, TIME)))[0]
    expect_output = copy.deepcopy(spectrogram)
    output = audio.MaskAlongAxisIID(mask_param=0, mask_value=5.0, axis=1)(spectrogram)
    allclose_nparray(output, expect_output, 0.0001, 0.0001)


def test_mask_along_axis_iid_pipeline():
    """
    Feature: MaskAlongAxisIID
    Description: Mindspore pipeline mode with normal testcase
    Expectation: The returned result is as expected
    """
    generator = gen([BATCH, CHANNEL, FREQ, TIME])
    dataset = ds.GeneratorDataset(source=generator, column_names=["multi_dimensional_data"])

    transforms = [audio.MaskAlongAxisIID(mask_param=8, mask_value=5.0, axis=2)]
    dataset = dataset.map(operations=transforms, input_columns=["multi_dimensional_data"])

    for item in dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
        output = item["multi_dimensional_data"]
    assert output.shape == (BATCH, CHANNEL, FREQ, TIME)


def test_mask_along_axis_iid_invalid_input():
    """
    Feature: MaskAlongAxisIID
    Description: Mindspore eager mode with invalid input
    Expectation: The returned result is as expected
    """

    def test_invalid_param(mask_param, mask_value, axis, error, error_msg):
        """
        a function used for checking correct error and message
        """
        with pytest.raises(error) as error_info:
            audio.MaskAlongAxisIID(mask_param, mask_value, axis)
        assert error_msg in str(error_info.value)

    test_invalid_param(1.0, 1.0, 1, TypeError,
                       "Argument mask_param with value 1.0 is not of type [<class 'int'>], but got <class 'float'>.")
    test_invalid_param(-1, 1.0, 1, ValueError,
                       "Input mask_param is not within the required interval of [0, 2147483647].")
    test_invalid_param(5, 1.0, 5.0, TypeError,
                       "Argument axis with value 5.0 is not of type [<class 'int'>], but got <class 'float'>.")
    test_invalid_param(5, 1.0, 0, ValueError,
                       "Input axis is not within the required interval of [1, 2].")
    test_invalid_param(5, 1.0, 3, ValueError,
                       "Input axis is not within the required interval of [1, 2].")


def test_mask_along_axis_iid_transform():
    """
    Feature: MaskAlongAxisIid
    Description: Test MaskAlongAxisIid with various valid input parameters and data types
    Expectation: The operation completes successfully
    """

    waveform = np.random.random((6, 5, 8)).astype(np.int16)
    dataset = ds.NumpySlicesDataset(data=waveform, column_names=["audio"])
    transforms = [audio.MaskAlongAxisIID(5, 0.5, 1)]
    dataset = dataset.map(operations=transforms, input_columns=["audio"])
    for _ in dataset.create_dict_iterator(output_numpy=True):
        break

    # MaskAlongAxisIID: test eager
    data = np.random.random((25, 8)).astype(np.float16)
    mask_along_axis_iid = audio.MaskAlongAxisIID(5, 2, 2)
    mask_along_axis_iid(data)

    # MaskAlongAxisIID: test mask_param is 1
    data = np.random.random((1, 1))
    mask_along_axis_iid = audio.MaskAlongAxisIID(1, 0.5, 2)
    mask_along_axis_iid(data)

    # MaskAlongAxisIID: test mask_param is equal to the number of columns
    data = np.random.random((1, 20))
    mask_along_axis_iid = audio.MaskAlongAxisIID(20, 0.5, 2)
    mask_along_axis_iid(data)

    # MaskAlongAxisIID: test mask_param is 0
    data = np.random.random((1, 1))
    mask_along_axis_iid = audio.MaskAlongAxisIID(0, 0.5, 2)
    mask_along_axis_iid(data)

    # MaskAlongAxisIID: test mask_value is negative integer
    data = np.random.random([6, 25, 20])
    mask_along_axis_iid = audio.MaskAlongAxisIID(5, -1, 2)
    mask_along_axis_iid(data)

    # MaskAlongAxisIID: test input tensor is 2d numpy data
    data = np.random.random((5, 8)).astype(np.float64)
    mask_along_axis_iid = audio.MaskAlongAxisIID(5, 2, 2)
    mask_along_axis_iid(data)

    # MaskAlongAxisIID: test input tensor is double type numpy data
    data = np.random.random((25, 20)).astype(np.double)
    mask_along_axis_iid = audio.MaskAlongAxisIID(5, 2, 2)
    mask_along_axis_iid(data)

    # MaskAlongAxisIID: test input tensor is 3d numpy data
    data = np.random.random((25, 20, 8)).astype(np.float64)
    mask_along_axis_iid = audio.MaskAlongAxisIID(5, 2, 2)
    mask_along_axis_iid(data)

    # MaskAlongAxisIID: test input tensor is 4d numpy data
    data = np.random.random((3, 2, 5, 8)).astype(np.float64)
    mask_along_axis_iid = audio.MaskAlongAxisIID(5, 2, 2)
    mask_along_axis_iid(data)

    # MaskAlongAxisIID: test input tensor is 5d numpy data
    data = np.random.random((3, 4, 2, 5, 8)).astype(np.float64)
    mask_along_axis_iid = audio.MaskAlongAxisIID(5, 2, 2)
    mask_along_axis_iid(data)

    # MaskAlongAxisIID: test input tensor is 6d numpy data
    data = np.random.random((2, 2, 2, 2, 2, 8)).astype(np.float64)
    mask_along_axis_iid = audio.MaskAlongAxisIID(5, 2, 2)
    mask_along_axis_iid(data)

    # MaskAlongAxisIID: test input tensor is 7d numpy data
    data = np.random.random((2, 3, 2, 2, 2, 2, 8)).astype(np.float64)
    mask_along_axis_iid = audio.MaskAlongAxisIID(5, 2, 2)
    mask_along_axis_iid(data)

    # MaskAlongAxisIID: test input tensor is 8d numpy data
    data = np.random.random((2, 2, 3, 3, 2, 2, 2, 8)).astype(np.float64)
    mask_along_axis_iid = audio.MaskAlongAxisIID(5, 2, 2)
    mask_along_axis_iid(data)


def test_mask_along_axis_iid_param_check():
    """
    Feature: MaskAlongAxisIid
    Description: Test MaskAlongAxisIid with invalid input data types and parameters
    Expectation: Correct error types and messages are raised as expected
    """

    data = np.random.random([4, 25, 20])
    with pytest.raises(RuntimeError, match="MaskAlongAxisIID: mask_param should be less than or "
                                           "equal to the length of time dimension."):
        audio.MaskAlongAxisIID(21, 0.5, 2)(data)

    # MaskAlongAxisIID: test mask_param is -1
    data = np.random.random([6, 25, 20])
    with pytest.raises(ValueError, match="Input mask_param is not within the required interval "
                                         "of \\[0, 2147483647\\]."):
        audio.MaskAlongAxisIID(-1, 0.5, 2)(data)

    # MaskAlongAxisIID: test mask_param is 2147483648
    data = np.random.random([8, 25, 20])
    with pytest.raises(ValueError, match="Input mask_param is not within the required interval "
                                         "of \\[0, 2147483647\\]."):
        audio.MaskAlongAxisIID(2147483648, 0.5, 2)(data)

    # MaskAlongAxisIID: test mask_param is float
    data = np.random.random([9, 25, 20])
    with pytest.raises(TypeError, match="Argument mask_param with value 5.0 is not of type \\[<class 'int'>\\], "
                                        "but got <class 'float'>."):
        audio.MaskAlongAxisIID(5.0, 0.5, 2)(data)

    # MaskAlongAxisIID: test mask_param is str
    data = np.random.random([2, 25, 20])
    with pytest.raises(TypeError, match="Argument mask_param with value test is not of type \\[<class 'int'>\\], "
                                        "but got <class 'str'>."):
        audio.MaskAlongAxisIID('test', 0.5, 2)(data)

    # MaskAlongAxisIID: test mask_param is bool
    data = np.random.random([3, 25, 20])
    with pytest.raises(TypeError, match="Argument mask_param with value True is not of type \\(<class 'int'>,\\), "
                                        "but got <class 'bool'>."):
        audio.MaskAlongAxisIID(True, 0.5, 2)(data)

    # MaskAlongAxisIID: test mask_param is None
    data = np.random.random([4, 25, 20])
    with pytest.raises(TypeError, match="Argument mask_param with value None is not of type \\[<class 'int'>\\], "
                                        "but got <class 'NoneType'>."):
        audio.MaskAlongAxisIID(None, 0.5, 2)(data)

    # MaskAlongAxisIID: test mask_value is str
    data = np.random.random([7, 25, 20])
    with pytest.raises(TypeError, match="Argument mask_value with value test is not of type "
                                        "\\[<class 'int'>, <class 'float'>\\], but got <class 'str'>."):
        audio.MaskAlongAxisIID(5, "test", 2)(data)

    # MaskAlongAxisIID: test mask_value is list
    data = np.random.random([8, 25, 20])
    with pytest.raises(TypeError, match="Argument mask_value with value \\[1, 2\\] is not of type "
                                        "\\[<class 'int'>, <class 'float'>\\], but got <class 'list'>."):
        audio.MaskAlongAxisIID(5, [1, 2], 2)(data)

    # MaskAlongAxisIID: test mask_value is tuple
    data = np.random.random([9, 25, 20])
    with pytest.raises(TypeError, match="Argument mask_value with value \\(1, 2\\) is not of type \\[<class 'int'>, "
                                        "<class 'float'>\\], but got <class 'tuple'>."):
        audio.MaskAlongAxisIID(5, (1, 2), 2)(data)

    # MaskAlongAxisIID: test mask_value is None
    data = np.random.random([10, 25, 20])
    with pytest.raises(TypeError, match="Argument mask_value with value None is not of type \\[<class 'int'>, "
                                        "<class 'float'>\\], but got <class 'NoneType'>."):
        audio.MaskAlongAxisIID(5, None, 2)(data)

    # MaskAlongAxisIID: test mask_value is bool
    data = np.random.random([11, 25, 20])
    with pytest.raises(TypeError, match="Argument mask_value with value True is not of type \\(<class 'int'>, "
                                        "<class 'float'>\\), but got <class 'bool'>."):
        audio.MaskAlongAxisIID(5, True, 2)(data)

    # MaskAlongAxisIID: test axis is an integer that is not 1 or 2.
    data = np.random.random([13, 25, 20])
    with pytest.raises(ValueError, match="Input axis is not within the required interval of \\[1, 2\\]."):
        audio.MaskAlongAxisIID(5, 1, 3)(data)

    # MaskAlongAxisIID: test axis is float
    data = np.random.random([14, 25, 20])
    with pytest.raises(TypeError, match="Argument axis with value 1.5 is not of type \\[<class 'int'>\\], "
                                        "but got <class 'float'>."):
        audio.MaskAlongAxisIID(5, 1, 1.5)(data)

    # MaskAlongAxisIID: test axis is str
    data = np.random.random([15, 25, 20])
    with pytest.raises(TypeError, match="Argument axis with value test is not of type \\[<class 'int'>\\], "
                                        "but got <class 'str'>."):
        audio.MaskAlongAxisIID(5, 1, "test")(data)

    # MaskAlongAxisIID: test axis is None
    data = np.random.random([16, 25, 20])
    with pytest.raises(TypeError, match="Argument axis with value None is not of type \\[<class 'int'>\\], "
                                        "but got <class 'NoneType'>."):
        audio.MaskAlongAxisIID(5, 1, None)(data)

    # MaskAlongAxisIID: test axis is bool
    data = np.random.random([17, 25, 20])
    with pytest.raises(TypeError, match="Argument axis with value True is not of type \\(<class 'int'>,\\), "
                                        "but got <class 'bool'>."):
        audio.MaskAlongAxisIID(5, 1, True)(data)

    # MaskAlongAxisIID: test input tensor is 1d numpy data
    data = np.random.randn(8, )
    with pytest.raises(RuntimeError, match="MaskAlongAxisIID: the shape of input tensor does not match .*"):
        audio.MaskAlongAxisIID(5, 2, 1)(data)

    # MaskAlongAxisIID: test input tensor is not transferred
    with pytest.raises(RuntimeError, match="Input Tensor is not valid."):
        audio.MaskAlongAxisIID(5, 2, 1)()

    # MaskAlongAxisIID: test input tensor is str numpy data
    data = np.random.random((25, 20)).astype(np.str_)
    with pytest.raises(RuntimeError, match="MaskAlongAxisIID: the data type ."):
        audio.MaskAlongAxisIID(5, 2, 2)(data)

    # MaskAlongAxisIID: test more parameter
    data = np.random.random((25, 20)).astype(np.double)
    with pytest.raises(TypeError, match="got an unexpected keyword argument 'test'"):
        audio.MaskAlongAxisIID(5, 2, 2, test="test")(data)


if __name__ == "__main__":
    test_mask_along_axis_iid_eager()
    test_mask_along_axis_iid_invalid_input()
    test_mask_along_axis_iid_pipeline()
    test_mask_along_axis_iid_transform()
    test_mask_along_axis_iid_param_check()
