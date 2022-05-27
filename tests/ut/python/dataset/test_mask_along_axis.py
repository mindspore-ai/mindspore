# Copyright 2022 Huawei Technologies Co., Ltd
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
import copy
import numpy as np
import pytest

import mindspore.dataset as ds
import mindspore.dataset.audio as atf
from mindspore import log as logger

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


def count_unequal_element(data_expected, data_me, rtol, atol):
    """
    Precision calculation func
    """
    assert data_expected.shape == data_me.shape
    total_count = len(data_expected.flatten())
    error = np.abs(data_expected - data_me)
    greater = np.greater(error, atol + np.abs(data_expected) * rtol)
    loss_count = np.count_nonzero(greater)
    assert (loss_count / total_count) < rtol, "\ndata_expected_std:{0}\ndata_me_error:{1}\nloss:{2}".format(
        data_expected[greater], data_me[greater], error[greater])


def gen(shape):
    np.random.seed(0)
    data = np.random.random(shape)
    yield(np.array(data, dtype=np.float32),)


def test_mask_along_axis_eager_random_input():
    """
    Feature: MaskAlongAxis
    Description: Mindspore eager mode normal testcase with random input tensor
    Expectation: The returned result is as expected
    """
    logger.info("test Mask_Along_axis op")
    spectrogram = next(gen((CHANNEL, FREQ, TIME)))[0]
    expect_output = copy.deepcopy(spectrogram)
    out_put = atf.MaskAlongAxis(mask_start=0, mask_width=1, mask_value=5.0, axis=2)(spectrogram)
    for item in expect_output[0]:
        item[0] = 5.0
    assert out_put.shape == (CHANNEL, FREQ, TIME)
    allclose_nparray(out_put, expect_output, 0.0001, 0.0001)


def test_mask_along_axis_eager_precision():
    """
    Feature: MaskAlongAxis
    Description: Mindspore eager mode checking precision
    Expectation: The returned result is as expected
    """
    logger.info("test MaskAlongAxis op, checking precision")
    spectrogram_0 = np.array([[[-0.0635, -0.6903],
                               [-1.7175, -0.0815],
                               [0.7981, -0.8297],
                               [-0.4589, -0.7506]],
                              [[0.6189, 1.1874],
                               [0.1856, -0.5536],
                               [1.0620, 0.2071],
                               [-0.3874, 0.0664]]]).astype(np.float32)
    out_ms_0 = atf.MaskAlongAxis(mask_start=0, mask_width=1, mask_value=2.0, axis=2)(spectrogram_0)
    spectrogram_1 = np.array([[[-0.0635, -0.6903],
                               [-1.7175, -0.0815],
                               [0.7981, -0.8297],
                               [-0.4589, -0.7506]],
                              [[0.6189, 1.1874],
                               [0.1856, -0.5536],
                               [1.0620, 0.2071],
                               [-0.3874, 0.0664]]]).astype(np.float64)
    out_ms_1 = atf.MaskAlongAxis(mask_start=0, mask_width=1, mask_value=2.0, axis=2)(spectrogram_1)
    out_benchmark = np.array([[[2.0000, -0.6903],
                               [2.0000, -0.0815],
                               [2.0000, -0.8297],
                               [2.0000, -0.7506]],
                              [[2.0000, 1.1874],
                               [2.0000, -0.5536],
                               [2.0000, 0.2071],
                               [2.0000, 0.0664]]]).astype(np.float32)
    allclose_nparray(out_ms_0, out_benchmark, 0.0001, 0.0001)
    allclose_nparray(out_ms_1, out_benchmark, 0.0001, 0.0001)


def test_mask_along_axis_pipeline():
    """
    Feature: MaskAlongAxis
    Description: Mindspore pipeline mode normal testcase
    Expectation: The returned result is as expected
    """
    logger.info("test MaskAlongAxis op, pipeline")

    generator = gen((CHANNEL, FREQ, TIME))
    expect_output = copy.deepcopy(next(gen((CHANNEL, FREQ, TIME)))[0])
    data1 = ds.GeneratorDataset(source=generator, column_names=["multi_dimensional_data"])
    transforms = [atf.MaskAlongAxis(mask_start=2, mask_width=2, mask_value=2.0, axis=2)]
    data1 = data1.map(operations=transforms, input_columns=["multi_dimensional_data"])

    for item in data1.create_dict_iterator(num_epochs=1, output_numpy=True):
        out_put = item["multi_dimensional_data"]

    for item in expect_output[0]:
        item[2] = 2.0
        item[3] = 2.0
    assert out_put.shape == (CHANNEL, FREQ, TIME)
    allclose_nparray(out_put, expect_output, 0.0001, 0.0001)


def test_mask_along_axis_invalid_input():
    """
    Feature: MaskAlongAxis
    Description: Mindspore eager mode with invalid input tensor
    Expectation: Throw correct error and message
    """
    def test_invalid_param(test_name, mask_start, mask_width, mask_value, axis, error, error_msg):
        """
        a function used for checking correct error and message with various input
        """
        logger.info("Test MaskAlongAxis with wrong params: {0}".format(test_name))
        with pytest.raises(error) as error_info:
            atf.MaskAlongAxis(mask_start, mask_width, mask_value, axis)
        assert error_msg in str(error_info.value)

    test_invalid_param("invalid mask_start", -1, 10, 1.0, 1, ValueError,
                       "Input mask_start is not within the required interval of [0, 2147483647].")
    test_invalid_param("invalid mask_width", 0, -1, 1.0, 1, ValueError,
                       "Input mask_width is not within the required interval of [1, 2147483647].")
    test_invalid_param("invalid axis", 0, 10, 1.0, 1.0, TypeError,
                       "Argument axis with value 1.0 is not of type [<class 'int'>], but got <class 'float'>.")
    test_invalid_param("invalid axis", 0, 10, 1.0, 0, ValueError,
                       "Input axis is not within the required interval of [1, 2].")
    test_invalid_param("invalid axis", 0, 10, 1.0, 3, ValueError,
                       "Input axis is not within the required interval of [1, 2].")
    test_invalid_param("invalid axis", 0, 10, 1.0, -1, ValueError,
                       "Input axis is not within the required interval of [1, 2].")


if __name__ == "__main__":
    test_mask_along_axis_eager_random_input()
    test_mask_along_axis_eager_precision()
    test_mask_along_axis_pipeline()
    test_mask_along_axis_invalid_input()
