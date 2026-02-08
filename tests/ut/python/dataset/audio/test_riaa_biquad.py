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
"""Test RiaaBiquad."""

import numpy as np
import pytest

import mindspore.dataset as ds
from mindspore.dataset import audio
from . import count_unequal_element


def test_riaa_biquad_eager():
    """
    Feature: RiaaBiquad
    Description: Test RiaaBiquad in eager mode under normal test case
    Expectation: Output is equal to the expected output
    """
    # Original waveform
    waveform = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64)
    # Expect waveform
    expect_waveform = np.array([[0.23806122, 0.70914434, 1.],
                                [0.95224489, 1., 1.]], dtype=np.float64)
    riaa_biquad = audio.RiaaBiquad(44100)
    # Filtered waveform by RiaaBiquad
    output = riaa_biquad(waveform)
    count_unequal_element(expect_waveform, output, 0.0001, 0.0001)


def test_riaa_biquad_pipeline():
    """
    Feature: RiaaBiquad
    Description: Test RiaaBiquad in pipeline mode under normal test case
    Expectation: Output is equal to the expected output
    """
    # Original waveform
    waveform = np.array([[1.47, 4.722, 5.863], [0.492, 0.235, 0.56]], dtype=np.float32)
    # Expect waveform
    expect_waveform = np.array([[0.18626465, 0.7859906, 1.],
                                [0.06234163, 0.09258664, 0.15710703]], dtype=np.float64)
    dataset = ds.NumpySlicesDataset(waveform, ["waveform"], shuffle=False)
    riaa_biquad = audio.RiaaBiquad(88200)
    # Filtered waveform by RiaaBiquad
    dataset = dataset.map(input_columns=["waveform"], operations=riaa_biquad)
    i = 0
    for item in dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
        count_unequal_element(expect_waveform[i, :], item['waveform'], 0.0001, 0.0001)
        i += 1


def test_riaa_biquad_invalid_parameter():
    """
    Feature: RiaaBiquad
    Description: Test RiaaBiquad with invalid parameter
    Expectation: Error is raised as expected
    """
    def test_invalid_input(sample_rate, error, error_msg):
        with pytest.raises(error) as error_info:
            audio.RiaaBiquad(sample_rate)
        assert error_msg in str(error_info.value)

    test_invalid_input(44100.5, TypeError,
                       "Argument sample_rate with value 44100.5 is not of type [<class 'int'>],"
                       " but got <class 'float'>.")
    test_invalid_input("44100", TypeError,
                       "Argument sample_rate with value 44100 is not of type [<class 'int'>],"
                       + " but got <class 'str'>.")
    test_invalid_input(45670, ValueError,
                       "sample_rate should be one of [44100, 48000, 88200, 96000], but got 45670.")


def test_riaa_biquad_transform():
    """
    Feature: RiaaBiquad
    Description: Test RiaaBiquad with various valid input parameters and data types
    Expectation: The operation completes successfully
    """

    waveform = np.array([[1.234, 0.1873, 0.6], [-0.73, 3.886, 0.666]], dtype=np.float16)
    actual = audio.RiaaBiquad(48000)(waveform)
    assert np.shape(actual) == (2, 3)


def test_riaa_biquad_param_check():
    """
    Feature: RiaaBiquad
    Description: Test RiaaBiquad with invalid input data types and parameters
    Expectation: Correct error types and messages are raised as expected
    """

    with pytest.raises(TypeError) as error_info:
        error_msg = "Argument sample_rate with value 44100.5 is not of type [<class 'int'>], but got <class 'float'>."
        audio.RiaaBiquad(44100.5)
    assert error_msg in str(error_info.value)

    # Test with invalid sample_rate parameter type (TypeError expected)
    with pytest.raises(TypeError) as error_info:
        error_msg = "Argument sample_rate with value 44100 is not of type [<class 'int'>], but got <class 'str'>."
        audio.RiaaBiquad("44100")
    assert error_msg in str(error_info.value)

    # Test with invalid sample_rate parameter value (ValueError expected)
    with pytest.raises(ValueError) as error_info:
        error_msg = "sample_rate should be one of [44100, 48000, 88200, 96000], but got 45670."
        audio.RiaaBiquad(45670)
    assert error_msg in str(error_info.value)

    # Test with invalid sample_rate parameter type (TypeError expected)
    with pytest.raises(TypeError) as error_info:
        error_msg = "Argument sample_rate with value [44100] is not of type [<class 'int'>], but got <class 'list'>."
        audio.RiaaBiquad([44100])
    assert error_msg in str(error_info.value)

    # Test with invalid sample_rate parameter value (ValueError expected)
    with pytest.raises(ValueError) as error_info:
        error_msg = "ample_rate should be one of [44100, 48000, 88200, 96000], but got 0."
        audio.RiaaBiquad(0)
    assert error_msg in str(error_info.value)


if __name__ == "__main__":
    test_riaa_biquad_eager()
    test_riaa_biquad_pipeline()
    test_riaa_biquad_invalid_parameter()
    test_riaa_biquad_transform()
    test_riaa_biquad_param_check()
