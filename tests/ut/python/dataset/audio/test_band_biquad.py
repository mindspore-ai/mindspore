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
"""Test BandBiquad."""

import numpy as np
import pytest

import mindspore.dataset as ds
from mindspore.dataset import audio
from . import count_unequal_element


def test_band_biquad_eager():
    """
    Feature: BandBiquad
    Description: Test BandBiquad in eager mode with valid input
    Expectation: Output is equal to the expected output
    """

    # Original waveform
    waveform = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64)
    # Expect waveform
    expect_waveform = np.array([[0.00137832, 0.00545664, 0.01350014],
                                [0.00551329, 0.01769161, 0.03763063]], dtype=np.float64)
    band_biquad = audio.BandBiquad(44100, 200.0, 0.707, False)
    # Filtered waveform by bandbiquad
    output = band_biquad(waveform)
    count_unequal_element(expect_waveform, output, 0.0001, 0.0001)


def test_band_biquad_pipeline():
    """
    Feature: BandBiquad
    Description: Test BandBiquad in pipeline mode with valid input
    Expectation: Output is equal to the expected output
    """

    # Original waveform
    waveform = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64)
    # Expect waveform
    expect_waveform = np.array([[0.00137832, 0.00545664, 0.01350014],
                                [0.00551329, 0.01769161, 0.03763063]], dtype=np.float64)
    label = np.random.random_sample((2, 1))
    data = (waveform, label)
    dataset = ds.NumpySlicesDataset(data, ["channel", "sample"], shuffle=False)
    band_biquad = audio.BandBiquad(44100, 200.0)
    # Filtered waveform by bandbiquad
    dataset = dataset.map(
        input_columns=["channel"], operations=band_biquad, num_parallel_workers=8)
    i = 0
    for item in dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
        count_unequal_element(expect_waveform[i, :],
                              item['channel'], 0.0001, 0.0001)
        i += 1


def test_band_biquad_invalid_input():
    """
    Feature: BandBiquad
    Description: Test BandBiquad with invalid input
    Expectation: Correct error and message are thrown as expected
    """
    def test_invalid_input(sample_rate, central_freq, Q, noise, error, error_msg):
        with pytest.raises(error) as error_info:
            audio.BandBiquad(sample_rate, central_freq, Q, noise)
        assert error_msg in str(error_info.value)

    test_invalid_input(44100.5, 200, 0.707, True, TypeError,
                       "Argument sample_rate with value 44100.5 is not of type [<class 'int'>],"
                       " but got <class 'float'>.")
    test_invalid_input("44100", 200, 0.707, True, TypeError,
                       "Argument sample_rate with value 44100 is not of type [<class 'int'>], but got <class 'str'>.")
    test_invalid_input(44100, "200", 0.707, True, TypeError,
                       "Argument central_freq with value 200 is not of type [<class 'float'>, <class 'int'>],"
                       " but got <class 'str'>.")
    test_invalid_input(0, 200, 0.707, True, ValueError,
                       "Input sample_rate is not within the required interval of [-2147483648, 0) and (0, 2147483647].")
    test_invalid_input(44100, 32434324324234321, 0.707, True, ValueError,
                       "Input central_freq is not within the required interval of [-16777216, 16777216].")
    test_invalid_input(44100, 200, "0.707", True, TypeError,
                       "Argument Q with value 0.707 is not of type [<class 'float'>, <class 'int'>],"
                       " but got <class 'str'>.")
    test_invalid_input(44100, 200, 1.707, True, ValueError,
                       "Input Q is not within the required interval of (0, 1].")
    test_invalid_input(44100, 200, 0, True, ValueError,
                       "Input Q is not within the required interval of (0, 1].")
    test_invalid_input(441324343243242342345300, 200, 0.707, True, ValueError,
                       "Input sample_rate is not within the required interval of [-2147483648, 0) and (0, 2147483647].")
    test_invalid_input(None, 200, 0.707, True, TypeError,
                       "Argument sample_rate with value None is not of type [<class 'int'>],"
                       " but got <class 'NoneType'>.")
    test_invalid_input(44100, None, 0.707, True, TypeError,
                       "Argument central_freq with value None is not of type [<class 'float'>, <class 'int'>],"
                       " but got <class 'NoneType'>.")
    test_invalid_input(44100, 200, 0.707, "False", TypeError,
                       "Argument noise with value False is not of type [<class 'bool'>], but got <class 'str'>.")


if __name__ == "__main__":
    test_band_biquad_eager()
    test_band_biquad_pipeline()
    test_band_biquad_invalid_input()
