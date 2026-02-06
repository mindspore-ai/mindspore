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
"""Test MaskAlongAxis."""

import copy
import numpy as np
import pytest

import mindspore.dataset as ds
from mindspore.dataset import audio
from . import count_unequal_element

CHANNEL = 1
FREQ = 5
TIME = 5


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


def test_mask_along_axis_eager_random_input():
    """
    Feature: MaskAlongAxis
    Description: Mindspore eager mode normal testcase with random input tensor
    Expectation: The returned result is as expected
    """
    spectrogram = next(gen((CHANNEL, FREQ, TIME)))[0]
    expect_output = copy.deepcopy(spectrogram)
    output = audio.MaskAlongAxis(mask_start=0, mask_width=1, mask_value=5.0, axis=2)(
        spectrogram
    )
    for item in expect_output[0]:
        item[0] = 5.0
    assert output.shape == (CHANNEL, FREQ, TIME)
    allclose_nparray(output, expect_output, 0.0001, 0.0001)


def test_mask_along_axis_eager_precision():
    """
    Feature: MaskAlongAxis
    Description: Mindspore eager mode checking precision
    Expectation: The returned result is as expected
    """
    spectrogram0 = np.array(
        [
            [
                [-0.0635, -0.6903],
                [-1.7175, -0.0815],
                [0.7981, -0.8297],
                [-0.4589, -0.7506],
            ],
            [[0.6189, 1.1874], [0.1856, -0.5536], [1.0620, 0.2071], [-0.3874, 0.0664]],
        ]
    ).astype(np.float32)
    output_0 = audio.MaskAlongAxis(mask_start=0, mask_width=1, mask_value=2.0, axis=2)(
        spectrogram0
    )
    spectrogram1 = np.array(
        [
            [
                [-0.0635, -0.6903],
                [-1.7175, -0.0815],
                [0.7981, -0.8297],
                [-0.4589, -0.7506],
            ],
            [[0.6189, 1.1874], [0.1856, -0.5536], [1.0620, 0.2071], [-0.3874, 0.0664]],
        ]
    ).astype(np.float64)
    output_1 = audio.MaskAlongAxis(mask_start=0, mask_width=1, mask_value=2.0, axis=2)(
        spectrogram1
    )
    out_benchmark = np.array(
        [
            [
                [2.0000, -0.6903],
                [2.0000, -0.0815],
                [2.0000, -0.8297],
                [2.0000, -0.7506],
            ],
            [[2.0000, 1.1874], [2.0000, -0.5536], [2.0000, 0.2071], [2.0000, 0.0664]],
        ]
    ).astype(np.float32)
    allclose_nparray(output_0, out_benchmark, 0.0001, 0.0001)
    allclose_nparray(output_1, out_benchmark, 0.0001, 0.0001)


def test_mask_along_axis_pipeline():
    """
    Feature: MaskAlongAxis
    Description: Mindspore pipeline mode normal testcase
    Expectation: The returned result is as expected
    """
    generator = gen((CHANNEL, FREQ, TIME))
    expect_output = copy.deepcopy(next(gen((CHANNEL, FREQ, TIME)))[0])
    dataset = ds.GeneratorDataset(
        source=generator, column_names=["multi_dimensional_data"]
    )
    transforms = [
        audio.MaskAlongAxis(mask_start=2, mask_width=2, mask_value=2.0, axis=2)
    ]
    dataset = dataset.map(operations=transforms, input_columns=["multi_dimensional_data"])

    for item in dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
        output = item["multi_dimensional_data"]

    for item in expect_output[0]:
        item[2] = 2.0
        item[3] = 2.0
    assert output.shape == (CHANNEL, FREQ, TIME)
    allclose_nparray(output, expect_output, 0.0001, 0.0001)


def test_mask_along_axis_invalid_input():
    """
    Feature: MaskAlongAxis
    Description: Mindspore eager mode with invalid input tensor
    Expectation: Throw correct error and message
    """

    def test_invalid_param(mask_start, mask_width, mask_value, axis, error, error_msg):
        """
        a function used for checking correct error and message with various input
        """
        with pytest.raises(error) as error_info:
            audio.MaskAlongAxis(mask_start, mask_width, mask_value, axis)
        assert error_msg in str(error_info.value)

    test_invalid_param(
        -1,
        10,
        1.0,
        1,
        ValueError,
        "Input mask_start is not within the required interval of [0, 2147483647].",
    )
    test_invalid_param(
        0,
        -1,
        1.0,
        1,
        ValueError,
        "Input mask_width is not within the required interval of [1, 2147483647].",
    )
    test_invalid_param(
        0,
        10,
        1.0,
        1.0,
        TypeError,
        "Argument axis with value 1.0 is not of type [<class 'int'>], but got <class 'float'>.",
    )
    test_invalid_param(
        0,
        10,
        1.0,
        0,
        ValueError,
        "Input axis is not within the required interval of [1, 2].",
    )
    test_invalid_param(
        0,
        10,
        1.0,
        3,
        ValueError,
        "Input axis is not within the required interval of [1, 2].",
    )
    test_invalid_param(
        0,
        10,
        1.0,
        -1,
        ValueError,
        "Input axis is not within the required interval of [1, 2].",
    )


def test_mask_along_axis_transform():
    """
    Feature: MaskAlongAxis
    Description: Test MaskAlongAxis with various valid input parameters and data types
    Expectation: The operation completes successfully
    """

    waveform = np.random.randn(4, 4).astype(np.float16)
    mask_along_axis = audio.MaskAlongAxis(0, 1, 3, 1)
    mask_along_axis(waveform)

    # Normal case, input random number size is 4 x 4 x 4, element type is float32;
    waveform = np.random.randn(4, 4, 4).astype(np.float32)
    mask_along_axis = audio.MaskAlongAxis(1, 1, 1.0, 2)
    mask_along_axis(waveform)

    # Normal case, input random number size is 4 x 4 x 8 x 4, element type is float64;
    waveform = np.random.randn(4, 4, 8, 4).astype(np.float32)
    mask_along_axis = audio.MaskAlongAxis(6, 1, 1.0, 1)
    mask_along_axis(waveform)

    # Normal case, input custom ndarray, element type is int16;
    waveform = np.array(
        [[[[1, 2, 3, 4, 5], [3, 4, 5, 6, 7]], [[5, 6, 7, 8, 9], [7, 8, 9, 10, 11]]]]
    ).astype(np.int16)
    mask_along_axis = audio.MaskAlongAxis(3, 1, 1.0, 2)
    mask_along_axis(waveform)

    # Normal case, input element type is int32, mask_start=0
    waveform = np.array([[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]]).astype(np.int32)
    mask_along_axis = audio.MaskAlongAxis(0, 1, 1.0, 1)
    mask_along_axis(waveform)

    # Normal case, input element type is int64;
    waveform = np.array(
        [[[[1, 2], [3, 4], [5, 6], [7, 8]], [[5, 6], [7, 8], [9, 10], [11, 12]]]]
    ).astype(np.int64)
    mask_along_axis = audio.MaskAlongAxis(3, 1, 1.0, 1)
    out = mask_along_axis(waveform)
    assert out.shape == waveform.shape

    # Normal case, input element type is uint8
    waveform = np.array([[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]]).astype(np.uint8)
    mask_along_axis = audio.MaskAlongAxis(1, 1, 1.0, 2)
    mask_along_axis(waveform)

    # Pipeline normal case
    spectrogram = np.random.randn(10, 1, 2, 3, 4)
    dataset = ds.NumpySlicesDataset(spectrogram, ["spectrogram"], shuffle=False)
    mask = audio.MaskAlongAxis(1, 2, 1.0, 1)
    dataset = dataset.map(
        input_columns=["spectrogram"], operations=mask, num_parallel_workers=8
    )
    for _ in dataset.create_dict_iterator(output_numpy=True):
        pass


def test_mask_along_axis_param_check():
    """
    Feature: MaskAlongAxis
    Description: Test MaskAlongAxis with invalid input data types and parameters
    Expectation: Correct error types and messages are raised as expected
    """

    waveform = np.random.randn(4)
    mask_along_axis = audio.MaskAlongAxis(1, 1, 1.0, 1)
    with pytest.raises(
        RuntimeError,
        match="MaskAlongAxis: the shape of input tensor does not match "
        "the requirement of operator. Expecting tensor in shape "
        "of <..., freq, time>. But got tensor with dimension 1.",
    ):
        mask_along_axis(waveform)

    # Exception case, input element type is string
    waveform = np.array([[[["a", "b"], ["a", "a"]], [["a", "a"], ["a", "a"]]]])
    mask_along_axis = audio.MaskAlongAxis(0, 1, 1.0, 1)
    with pytest.raises(
        RuntimeError,
        match="MaskAlongAxis: the data type of input tensor "
        "does not match the requirement of operator. Expecting tensor "
        "in type of .*int, float, double.*. But got type string.",
    ):
        mask_along_axis(waveform)

    # Exception case, input data type is list
    waveform = [1, 2, 3, 4, 5, 6, 7, 8]
    mask_along_axis = audio.MaskAlongAxis(0, 1, 1.0, 1)
    with pytest.raises(
        TypeError, match="Input should be NumPy audio, got <class 'list'>."
    ):
        mask_along_axis(waveform)

    # Exception case, input data type is bool
    waveform = True
    mask_along_axis = audio.MaskAlongAxis(0, 1, 1.0, 1)
    with pytest.raises(
        TypeError, match="Input should be NumPy audio, got <class 'bool'>."
    ):
        mask_along_axis(waveform)

    # Exception case, input data type is str
    waveform = "12345678"
    mask_along_axis = audio.MaskAlongAxis(0, 1, 1.0, 1)
    with pytest.raises(
        TypeError, match="Input should be NumPy audio, got <class 'str'>."
    ):
        mask_along_axis(waveform)

    # Exception case, input data type is float
    waveform = 12.34
    mask_along_axis = audio.MaskAlongAxis(0, 1, 1.0, 1)
    with pytest.raises(
        TypeError, match="Input should be NumPy audio, got <class 'float'>."
    ):
        mask_along_axis(waveform)

    # Exception case, mask_start data type is bool
    with pytest.raises(
        TypeError, match=".*mask_start.*is not of type.*'int'.*, but got.*'bool'.*"
    ):
        audio.MaskAlongAxis(True, 2, 0, 1)

    # Exception case, mask_start < 0
    with pytest.raises(
        ValueError, match=".*mask_start is not within.*[0, 2147483647].*"
    ):
        audio.MaskAlongAxis(-1, 2, 0, 1)

    # Exception case, mask_start out of [0, 2147483647] range
    with pytest.raises(
        ValueError, match=".*mask_start is not within.*[0, 2147483647].*"
    ):
        audio.MaskAlongAxis(2147483648, 2, 0, 2)

    # Exception case, when axis=1, mask_start exceeds number of rows
    waveform = np.random.randn(4, 4, 4)
    mask_along_axis = audio.MaskAlongAxis(5, 1, 1.0, 1)
    with pytest.raises(
        RuntimeError,
        match=".*MaskAlongAxis.*'mask_start' should be "
        "less than the length of the masked dimension.*",
    ):
        mask_along_axis(waveform)

    # Exception case, when axis=2, mask_start exceeds number of columns
    waveform = np.random.randn(4, 4, 4)
    mask_along_axis = audio.MaskAlongAxis(5, 1, 1.0, 2)
    with pytest.raises(
        RuntimeError,
        match=".*MaskAlongAxis.*'mask_start' should be "
        "less than the length of the masked dimension.*",
    ):
        mask_along_axis(waveform)

    # Exception case, mask_width data type is str
    with pytest.raises(
        TypeError, match=".*mask_width.*is not of type.*'int'.*, but got.*'str'.*"
    ):
        audio.MaskAlongAxis(2, "sss", 0, 1)

    # Exception case, mask_width=0
    with pytest.raises(
        ValueError, match=".*mask_width is not within.*[1, 2147483647].*"
    ):
        audio.MaskAlongAxis(2, 0, 0, 1)

    # Exception case, mask_width < 0
    with pytest.raises(
        ValueError, match=".*mask_width is not within.*[1, 2147483647].*"
    ):
        audio.MaskAlongAxis(2, -5, 0, 1)

    # Exception case, mask_width out of [1, 2147483647] range
    with pytest.raises(
        ValueError, match=".*mask_width is not within.*[1, 2147483647].*"
    ):
        audio.MaskAlongAxis(2, 2147483648, 0, 1)

    # Exception case, when axis=1, mask_start+mask_width exceeds number of rows
    waveform = np.random.randn(4, 4, 4)
    mask_along_axis = audio.MaskAlongAxis(2, 4, 1.0, 1)
    with pytest.raises(
        RuntimeError,
        match=".*MaskAlongAxis.*the sum of 'mask_start' and"
        " 'mask_width' should be no more than.*masked dimension.*",
    ):
        mask_along_axis(waveform)

    # Exception case, when axis=2, mask_start+mask_width exceeds number of columns
    waveform = np.random.randn(4, 4, 4)
    mask_along_axis = audio.MaskAlongAxis(2, 3, 1.0, 2)
    with pytest.raises(
        RuntimeError,
        match=".*MaskAlongAxis.*the sum of 'mask_start' and"
        " 'mask_width' should be no more than.*masked dimension.*",
    ):
        mask_along_axis(waveform)

    # Exception case, mask_value data type is list
    with pytest.raises(
        TypeError, match=".*mask_value.*not of type.*'int'.*'float'.*but got.*'list'.*"
    ):
        audio.MaskAlongAxis(1, 1, [5], 1)

    # Exception case, mask_value exceeds maximum value
    with pytest.raises(
        ValueError, match=".*mask_value is not within.*[-16777216, 16777216].*"
    ):
        audio.MaskAlongAxis(2, 5, 16777217, 1)

    # Exception case, mask_value less than minimum value
    with pytest.raises(
        ValueError, match=".*mask_value is not within.*[-16777216, 16777216].*"
    ):
        audio.MaskAlongAxis(2, 5, -16777217, 1)

    # Exception case, axis data type is str
    with pytest.raises(
        TypeError, match=".*axis.*is not of type.*'int'.*but got.*'str'.*"
    ):
        audio.MaskAlongAxis(1, 1, 1.0, "sss")

    # Exception case, axis data type is bool
    with pytest.raises(
        TypeError, match=".*axis.*is not of type.*'int'.*but got.*'bool'.*"
    ):
        audio.MaskAlongAxis(1, 1, 1.0, True)

    # Exception case, axis not in valid range {1, 2}
    with pytest.raises(ValueError, match=".*axis is not within.*[1, 2].*"):
        audio.MaskAlongAxis(1, 1, 1.0, 5)

    # Exception case, MaskAlongAxis no input data provided
    mask_along_axis = audio.MaskAlongAxis(5, 5, 0, 1)
    with pytest.raises(RuntimeError, match="Input Tensor is not valid."):
        mask_along_axis()

    # Pipeline exception case, mask_start parameter exceeds the range of specified dimension of input tensor
    spectrogram = np.random.randn(10, 1, 2, 3, 4)
    dataset = ds.NumpySlicesDataset(spectrogram, ["spectrogram"], shuffle=False)
    mask = audio.MaskAlongAxis(10, 2, 1.0, 1)
    dataset = dataset.map(
        input_columns=["spectrogram"], operations=mask, num_parallel_workers=8
    )
    with pytest.raises(
        RuntimeError,
        match=".*MaskAlongAxis: .*'mask_start' should be less than.*masked dimension.*",
    ):
        i = 0
        for _ in dataset.create_dict_iterator(output_numpy=True):
            i += 1

    # Pipeline exception case, mask_start+mask_width parameter exceeds the range of specified dimension of input tensor
    spectrogram = np.random.randn(10, 1, 2, 3, 3)
    dataset = ds.NumpySlicesDataset(spectrogram, ["spectrogram"], shuffle=False)
    mask = audio.MaskAlongAxis(1, 5, 1.0, 1)
    dataset = dataset.map(
        input_columns=["spectrogram"], operations=mask, num_parallel_workers=8
    )
    with pytest.raises(
        RuntimeError,
        match=".*MaskAlongAxis: .*the sum of 'mask_start' and "
        "'mask_width' should be no more than.*masked dimension.*",
    ):
        i = 0
        for _ in dataset.create_dict_iterator(output_numpy=True):
            i += 1

    # Input Tensor type is string in pipeline (exception)
    spectrogram = np.array([[[["aa", "b"], ["ss", "a"]], [["ss", "a"], ["ss", "a"]]]])
    dataset = ds.NumpySlicesDataset(spectrogram, ["spectrogram"], shuffle=False)
    mask = audio.MaskAlongAxis(1, 1, 1.0, 1)
    dataset = dataset.map(
        input_columns=["spectrogram"], operations=mask, num_parallel_workers=8
    )
    with pytest.raises(
        RuntimeError,
        match="map operation: .*MaskAlongAxis.* failed. "
        "MaskAlongAxis: the data type of input tensor does not match "
        "the requirement of operator. Expecting tensor in type of "
        ".*int, float, double.*. But got type string.",
    ):
        i = 0
        for _ in dataset.create_dict_iterator(output_numpy=True):
            i += 1


if __name__ == "__main__":
    test_mask_along_axis_eager_random_input()
    test_mask_along_axis_eager_precision()
    test_mask_along_axis_pipeline()
    test_mask_along_axis_invalid_input()
    test_mask_along_axis_transform()
    test_mask_along_axis_param_check()
