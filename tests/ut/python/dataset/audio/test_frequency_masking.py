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
"""Test FrequencyMasking."""

import numpy as np
import pytest

import mindspore.dataset as ds
from mindspore.dataset import audio
from . import count_unequal_element

CHANNEL = 2
FREQ = 30
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


def test_frequency_masking_eager_random_input():
    """
    Feature: FrequencyMasking
    Description: Test FrequencyMasking in eager mode under normal test case with random input
    Expectation: Output's shape is equal to the expected output's shape
    """
    spectrogram = next(gen((CHANNEL, FREQ, TIME)))[0]
    out_put = audio.FrequencyMasking(False, 3, 1, 10)(spectrogram)
    assert out_put.shape == (CHANNEL, FREQ, TIME)


def test_frequency_masking_eager_precision():
    """
    Feature: FrequencyMasking
    Description: Test FrequencyMasking precision in eager mode under normal test case
    Expectation: Output is equal to the expected output
    """
    spectrogram = np.array([[[0.17274511, 0.85174704, 0.07162686, -0.45436913],
                             [-1.045921, -1.8204843, 0.62333095, -0.09532598],
                             [1.8175547, -0.25779432, -0.58152324, -0.00221091]],
                            [[-1.205032, 0.18922766, -0.5277673, -1.3090396],
                             [1.8914849, -0.97001046, -0.23726775, 0.00525892],
                             [-1.0271876, 0.33526883, 1.7413973, 0.12313101]]]).astype(np.float32)
    out_ms = audio.FrequencyMasking(False, 2, 0, 0)(spectrogram)
    out_benchmark = np.array([[[0.0, 0.0, 0.0, 0.0],
                               [0.0, 0.0, 0.0, 0.0],
                               [1.8175547, -0.25779432, -0.58152324, -0.00221091]],
                              [[0.0, 0.0, 0.0, 0.0],
                               [0.0, 0.0, 0.0, 0.0],
                               [-1.0271876, 0.33526883, 1.7413973, 0.12313101]]]).astype(np.float32)
    allclose_nparray(out_ms, out_benchmark, 0.0001, 0.0001)


def test_frequency_masking_pipeline():
    """
    Feature: FrequencyMasking
    Description: Test FrequencyMasking in pipeline mode under normal test case
    Expectation: Output's shape is equal to the expected output's shape
    """
    generator = gen([CHANNEL, FREQ, TIME])
    data1 = ds.GeneratorDataset(source=generator, column_names=["multi_dimensional_data"])

    transforms = [audio.FrequencyMasking(True, 8)]
    data1 = data1.map(operations=transforms, input_columns=["multi_dimensional_data"])

    for item in data1.create_dict_iterator(num_epochs=1, output_numpy=True):
        out_put = item["multi_dimensional_data"]
    assert out_put.shape == (CHANNEL, FREQ, TIME)


def test_frequency_masking_invalid_input():
    """
    Feature: FrequencyMasking
    Description: Test FrequencyMasking with invalid input
    Expectation: Error is raised as expected
    """

    def test_invalid_param(iid_masks, frequency_mask_param, mask_start, error, error_msg):
        with pytest.raises(error) as error_info:
            audio.FrequencyMasking(iid_masks, frequency_mask_param, mask_start)
        assert error_msg in str(error_info.value)

    def test_invalid_input(iid_masks, frequency_mask_param, mask_start, error, error_msg):
        with pytest.raises(error) as error_info:
            spectrogram = next(gen((CHANNEL, FREQ, TIME)))[0]
            audio.FrequencyMasking(iid_masks, frequency_mask_param, mask_start)(spectrogram)
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
                       "'frequency_mask_param' should be less than or equal to the length of frequency dimension")


def test_frequency_masking_transform():
    """
    Feature: FrequencyMasking
    Description: Test FrequencyMasking with various valid input parameters and data types
    Expectation: The operation completes successfully
    """
    waveform = np.random.randn(64, 40)
    freq_masking = audio.FrequencyMasking(False, 2, 0, 0)
    freq_masking(waveform)

    # Test with various parameter combinations
    waveform = np.random.randn(64, 40, 20)
    freq_masking = audio.FrequencyMasking(False, 2, 0, 0)
    freq_masking(waveform)

    # Test of float16 type
    waveform = np.random.randn(64, 40, 20, 10)
    freq_masking = audio.FrequencyMasking(False, 2, 0, 0)
    freq_masking(waveform)

    # Test of float32 type
    waveform = np.random.randn(64, 40).astype(np.float16)
    freq_masking = audio.FrequencyMasking(False, 2, 0, 0)
    freq_masking(waveform)

    # Test of float64 type
    waveform = np.random.randn(64, 40).astype(np.float32)
    freq_masking = audio.FrequencyMasking(False, 2, 0, 0)
    freq_masking(waveform)

    # Test of float64 type
    waveform = np.random.randn(64, 40).astype(np.float64)
    freq_masking = audio.FrequencyMasking(False, 2, 0, 0)
    freq_masking(waveform)

    # Test of int32 type
    waveform = np.random.randn(64, 40).astype(np.uint8)
    freq_masking = audio.FrequencyMasking(False, 2, 0, 0)
    freq_masking(waveform)

    # Test of int32 type
    waveform = np.random.randn(64, 40).astype(np.int32)
    freq_masking = audio.FrequencyMasking(False, 2, 0, 0)
    freq_masking(waveform)


def test_frequency_masking_param_check():
    """
    Feature: FrequencyMasking
    Description: Test FrequencyMasking with invalid input data types and parameters
    Expectation: Correct error types and messages are raised as expected
    """
    waveform = np.random.randn(64, 40).tolist()
    freq_masking = audio.FrequencyMasking(False, 2, 0, 0)
    with pytest.raises(TypeError, match="Input should be NumPy audio, got <class 'list'>."):
        freq_masking(waveform)

    # Test with invalid value parameter (ValueError expected)
    waveform = np.random.randn(64, 40).astype(np.float16)
    with pytest.raises(ValueError, match="Input mask_start is not within the required interval of \\[0, 16777216\\]."):
        freq_masking = audio.FrequencyMasking(True, 2, -10)

    # Test with invalid type parameter (TypeError expected)
    waveform = np.random.randn(64, 40).astype(np.float16)
    with pytest.raises(ValueError, match="Input mask_param is not within the required interval of \\[0, 16777216\\]."):
        freq_masking = audio.FrequencyMasking(True, -2, 10)

    # Test with invalid type parameter (TypeError expected)
    waveform = np.random.randn(64, 40).astype(np.float16)
    with pytest.raises(TypeError, match="Argument iid_masks with value True is not of type \\[<class 'bool'>\\], "
                                        "but got <class 'str'>."):
        freq_masking = audio.FrequencyMasking("True", 2, 10)

    # Test with invalid type parameter (TypeError expected)
    waveform = np.random.randn(64, 40).astype(np.float16)
    with pytest.raises(TypeError, match="Argument mask_param with value 2 is not of type \\[<class 'int'>\\], "
                                        "but got <class 'str'>."):
        freq_masking = audio.FrequencyMasking(True, "2", 10)

    # Test with invalid type parameter (TypeError expected)
    waveform = np.random.randn(64, 40).astype(np.float16)
    with pytest.raises(TypeError, match="Argument mask_start with value 10 is not of type \\[<class 'int'>\\], "
                                        "but got <class 'str'>."):
        freq_masking = audio.FrequencyMasking(True, 2, "10")


if __name__ == "__main__":
    test_frequency_masking_eager_random_input()
    test_frequency_masking_eager_precision()
    test_frequency_masking_pipeline()
    test_frequency_masking_invalid_input()
    test_frequency_masking_transform()
    test_frequency_masking_param_check()
