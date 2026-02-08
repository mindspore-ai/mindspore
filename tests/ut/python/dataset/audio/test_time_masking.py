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
"""Test TimeMasking."""

import numpy as np
import pytest

import mindspore.dataset as ds
from mindspore.dataset import audio
from . import count_unequal_element

CHANNEL = 2
FREQ = 20
TIME = 30


def gen(shape):
    np.random.seed(0)
    data = np.random.random(shape)
    yield (np.array(data, dtype=np.float32),)


def allclose_nparray(data_expected, data_me, rtol, atol, equal_nan=True):
    """ Precision calculation formula  """
    if np.any(np.isnan(data_expected)):
        assert np.allclose(data_me, data_expected, rtol, atol, equal_nan=equal_nan)
    elif not np.allclose(data_me, data_expected, rtol, atol, equal_nan=equal_nan):
        count_unequal_element(data_expected, data_me, rtol, atol)


def test_time_masking_eager_random_input():
    """
    Feature: TimeMasking
    Description: Test TimeMasking in eager mode under normal test case
    Expectation: Output's shape is the same as expected output's shape
    """
    spectrogram = next(gen((CHANNEL, FREQ, TIME)))[0]
    out_put = audio.TimeMasking(False, 3, 1, 10)(spectrogram)
    assert out_put.shape == (CHANNEL, FREQ, TIME)


def test_time_masking_eager_precision():
    """
    Feature: TimeMasking
    Description: Test TimeMasking in eager mode by comparing precision
    Expectation: Output is the same as expected output
    """
    spectrogram = np.array([[[0.17274511, 0.85174704, 0.07162686, -0.45436913],
                             [-1.045921, -1.8204843, 0.62333095, -0.09532598],
                             [1.8175547, -0.25779432, -0.58152324, -0.00221091]],
                            [[-1.205032, 0.18922766, -0.5277673, -1.3090396],
                             [1.8914849, -0.97001046, -0.23726775, 0.00525892],
                             [-1.0271876, 0.33526883, 1.7413973, 0.12313101]]]).astype(np.float32)
    output = audio.TimeMasking(False, 2, 0, 0)(spectrogram)
    out_benchmark = np.array([[[0., 0., 0.07162686, -0.45436913],
                               [0., 0., 0.62333095, -0.09532598],
                               [0., 0., -0.58152324, -0.00221091]],
                              [[0., 0., -0.5277673, -1.3090396],
                               [0., 0., -0.23726775, 0.00525892],
                               [0., 0., 1.7413973, 0.12313101]]]).astype(np.float32)
    allclose_nparray(output, out_benchmark, 0.0001, 0.0001)


def test_time_masking_pipeline():
    """
    Feature: TimeMasking
    Description: Test TimeMasking in pipeline mode under normal test case
    Expectation: Output's shape is the same as expected output's shape
    """
    generator = gen([CHANNEL, FREQ, TIME])
    dataset = ds.GeneratorDataset(source=generator, column_names=["multi_dimensional_data"])

    transforms = [audio.TimeMasking(True, 8)]
    dataset = dataset.map(operations=transforms, input_columns=["multi_dimensional_data"])

    for item in dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
        output = item["multi_dimensional_data"]
    assert output.shape == (CHANNEL, FREQ, TIME)


def test_time_masking_invalid_input():
    """
    Feature: TimeMasking
    Description: Test TimeMasking with invalid input
    Expectation: Correct error is raised as expected
    """

    def test_invalid_param(iid_masks, time_mask_param, mask_start, error, error_msg):
        with pytest.raises(error) as error_info:
            audio.TimeMasking(iid_masks, time_mask_param, mask_start)
        assert error_msg in str(error_info.value)

    def test_invalid_input(iid_masks, time_mask_param, mask_start, error, error_msg):
        with pytest.raises(error) as error_info:
            spectrogram = next(gen((CHANNEL, FREQ, TIME)))[0]
            audio.TimeMasking(iid_masks, time_mask_param, mask_start)(spectrogram)
        assert error_msg in str(error_info.value)

    test_invalid_param(True, 2, -10, ValueError,
                       "Input mask_start is not within the required interval of [0, 16777216].")
    test_invalid_param(True, -2, 10, ValueError,
                       "Input mask_param is not within the required interval of [0, 16777216].")
    test_invalid_param("True", 2, 10, TypeError,
                       "Argument iid_masks with value True is not of type [<class 'bool'>], but got <class 'str'>.")

    test_invalid_input(False, 2, 100, RuntimeError,
                       "'mask_start' should be less than the length of the masked dimension")
    test_invalid_input(False, 200, 2, RuntimeError,
                       "'time_mask_param' should be less than or equal to the length of time dimension")


def test_time_masking_transform():
    """
    Feature: TimeMasking
    Description: Test TimeMasking with various valid input parameters and data types
    Expectation: The operation completes successfully
    """

    spectrum = np.array([[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]])
    time_masking = audio.TimeMasking(False, 2, 3, 0)
    output = time_masking(spectrum)
    assert (np.array(output) == np.array([[[1, 2, 3, 0, 0, 6, 7, 8, 9, 10], [1, 2, 3, 0, 0, 6, 7, 8, 9, 10]]])).all()

    # test time_masking is normal
    spectrum = np.array([[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]])
    time_masking = audio.TimeMasking(False, 1, 3, 0)
    output = time_masking(spectrum)
    assert (np.array(output) == np.array([[[1, 2, 3, 0, 5, 6, 7, 8, 9, 10], [1, 2, 3, 0, 5, 6, 7, 8, 9, 10]]])).all()

    # test time_masking is normal
    spectrum = np.random.randn(20, 30, 14, 28)
    time_masking = audio.TimeMasking(True, 0, 0, 10)
    output = time_masking(spectrum)
    assert (np.array(output) == spectrum).all()

    # test time_masking is normal
    spectrum = np.random.randint(0, 250, (28, 32))
    time_masking = audio.TimeMasking(True, 20, 4, 0.26)
    time_masking(spectrum)

    # test time_masking is normal
    spectrum = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    time_masking = audio.TimeMasking(True, 10, 3, 30)
    time_masking(spectrum)

    # test time_masking is normal
    spectrum = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    time_masking = audio.TimeMasking(False, 3, 6, 8.0)
    time_masking(spectrum)


def test_time_masking_param_check():
    """
    Feature: TimeMasking
    Description: Test TimeMasking with invalid input data types and parameters
    Expectation: Correct error types and messages are raised as expected
    """

    spectrum = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    with pytest.raises(RuntimeError, match=".*MaskAlongAxis: invalid parameter, the sum of 'mask_start' and"
                                           " 'mask_width' should be no more than the length of the masked dimension"):
        time_masking = audio.TimeMasking(False, 10, 3, 30)
        time_masking(spectrum)

    # test Input type is abnormal
    spectrum = 10
    time_masking = audio.TimeMasking(True, 0, 0, 30)
    with pytest.raises(TypeError, match="Input should be NumPy audio, got <class 'int'>."):
        time_masking(spectrum)

    # test input is abnormal
    spectrum = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    time_masking = audio.TimeMasking(True, 4, 3, 30)
    with pytest.raises(RuntimeError, match=".*TimeMasking: the shape of input tensor does not "
                                           "match the requirement of operator. Expecting tensor "
                                           "in shape of <..., freq, time>. But got tensor with dimension 1."):
        time_masking(spectrum)

    # test Input type is abnormal
    spectrum = list(np.random.randn(10, 20))
    time_masking = audio.TimeMasking(True, 4, 3, 30)
    with pytest.raises(TypeError, match="Input should be NumPy audio, got <class 'list'>."):
        time_masking(spectrum)

    # test Input type is abnormal
    spectrum = np.array([["1", "2", "3", "4", "5"], ["6", "7", "8", "9", "10"]])
    time_masking = audio.TimeMasking(True, 2, 1, 30)
    with pytest.raises(RuntimeError, match=".*TimeMasking: the data type of input tensor does not"
                                           " match the requirement of operator. Expecting tensor "
                                           "in type of .*int, float, double.*But got type string."):
        time_masking(spectrum)

    # test iid_masks type is abnormal
    with pytest.raises(TypeError, match="Argument iid_masks with value True is not of "
                                        "type \\[<class 'bool'>\\], but got <class 'str'>."):
        audio.TimeMasking("True", 4, 3, 30)

    # test iid_masks type is abnormal
    with pytest.raises(TypeError, match="Argument iid_masks with value \\[1\\] is not of type \\[<class 'bool'>\\], "
                                        "but got <class 'list'>."):
        audio.TimeMasking([1], 4, 3, 30)

    # test iid_masks type is abnormal
    with pytest.raises(TypeError, match="Argument iid_masks with value \\(1, 2\\) is not of type \\[<class 'bool'>\\], "
                                        "but got <class 'tuple'>."):
        audio.TimeMasking((1, 2), 4, 3, 30)

    # test iid_masks type is abnormal
    with pytest.raises(TypeError, match="Argument iid_masks with value 3.0 is not of type \\[<class 'bool'>\\], "
                                        "but got <class 'float'>."):
        audio.TimeMasking(3.0, -1, 3, 30)

    # test mask_start is abnormal
    with pytest.raises(ValueError,
                       match="Input mask_start is not within the required interval of \\[0, 16777216\\]."):
        audio.TimeMasking(True, 2, -10)

    # test mask_param is abnormal
    with pytest.raises(ValueError,
                       match="Input mask_param is not within the required interval of \\[0, 16777216\\]."):
        audio.TimeMasking(True, -2, 10)


if __name__ == "__main__":
    test_time_masking_eager_random_input()
    test_time_masking_eager_precision()
    test_time_masking_pipeline()
    test_time_masking_invalid_input()
    test_time_masking_transform()
    test_time_masking_param_check()
