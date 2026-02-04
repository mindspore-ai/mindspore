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
"""Test DeemphBiquad."""

import numpy as np
import pytest

import mindspore.dataset as ds
from mindspore.dataset import audio
from . import count_unequal_element


def test_deemph_biquad_eager():
    """
    Feature: DeemphBiquad
    Description: Test DeemphBiquad in eager mode with valid input
    Expectation: Output is equal to the expected output
    """
    # Original waveform
    waveform = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=np.float64)
    # Expect waveform
    expect_waveform = np.array([[0.04603508, 0.11216372, 0.19070681],
                                [0.18414031, 0.31054966, 0.42633607]], dtype=np.float64)
    deemph_biquad = audio.DeemphBiquad(44100)
    # Filtered waveform by DeemphBiquad
    output = deemph_biquad(waveform)
    count_unequal_element(expect_waveform, output, 0.0001, 0.0001)


def test_deemph_biquad_pipeline():
    """
    Feature: DeemphBiquad
    Description: Test DeemphBiquad in pipeline mode with valid input
    Expectation: Output is equal to the expected output
    """
    # Original waveform
    waveform = np.array([[0.2, 0.2, 0.3], [0.4, 0.5, 0.7]], dtype=np.float64)
    # Expect waveform
    expect_waveform = np.array([[0.0895, 0.1279, 0.1972],
                                [0.1791, 0.3006, 0.4583]], dtype=np.float64)
    dataset = ds.NumpySlicesDataset(waveform, ["audio"], shuffle=False)
    deemph_biquad = audio.DeemphBiquad(48000)
    # Filtered waveform by DeemphBiquad
    dataset = dataset.map(
        input_columns=["audio"], operations=deemph_biquad, num_parallel_workers=8)
    i = 0
    for data in dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
        count_unequal_element(
            expect_waveform[i, :], data['audio'], 0.0001, 0.0001)
        i += 1


def test_invalid_input_all():
    """
    Feature: DeemphBiquad
    Description: Test DeemphBiquad with invalid input
    Expectation: Correct error and message are thrown as expected
    """
    waveform = np.random.rand(2, 1000)

    def test_invalid_input(sample_rate, error, error_msg):
        with pytest.raises(error) as error_info:
            audio.DeemphBiquad(sample_rate)(waveform)
        assert error_msg in str(error_info.value)

    test_invalid_input(44100.5, TypeError,
                       "Argument sample_rate with value 44100.5 is not of type [<class 'int'>],"
                       + " but got <class 'float'>.")
    test_invalid_input("44100", TypeError,
                       "Argument sample_rate with value 44100 is not of type [<class 'int'>],"
                       + " but got <class 'str'>.")
    test_invalid_input(45000, ValueError,
                       "Argument sample_rate should be 44100 or 48000, but got 45000.")


def test_deemph_biquad_transform():
    """
    Feature: DeemphBiquad
    Description: Test DeemphBiquad with various valid input parameters and data types
    Expectation: The operation completes successfully and output values are within valid range
    """
    # test DeemphBiquad is normal
    waveform = np.random.randn(20, 20, 20, 3).astype(np.float64)
    dataset = ds.NumpySlicesDataset(waveform, column_names=["column1"])
    allpass_biquad = audio.DeemphBiquad(48000)
    dataset = dataset.map(operations=allpass_biquad)
    for data in dataset.create_dict_iterator(output_numpy=True):
        assert (np.abs(data["column1"]) <= 1).all()

    # mindspore eager mode acc testcase:deemph_biquad
    waveform = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float16)
    deemph_biquad = audio.DeemphBiquad(44100)
    deemph_biquad(waveform)

    # mindspore eager mode acc testcase:deemph_biquad
    waveform = np.random.randint(-10000, 10000, (10, 10, 10, 10)).astype(np.double)
    deemph_biquad = audio.DeemphBiquad(44100)
    deemph_biquad(waveform)


def test_deemph_biquad_param_check():
    """
    Feature: DeemphBiquad
    Description: Test DeemphBiquad with invalid input data types and parameters
    Expectation: Correct error types and messages are raised as expected
    """
    # mindspore eager mode acc testcase:deemph_biquad
    waveform = list(np.random.randn(10, 10))
    deemph_biquad = audio.DeemphBiquad(44100)
    with pytest.raises(TypeError, match="Input should be NumPy audio, got <class 'list'>."):
        deemph_biquad(waveform)

    # mindspore eager mode acc testcase:deemph_biquad
    waveform = 10
    deemph_biquad = audio.DeemphBiquad(44100)
    with pytest.raises(TypeError, match="Input should be NumPy audio, got <class 'int'>."):
        deemph_biquad(waveform)

    # mindspore eager mode acc testcase:deemph_biquad
    waveform = np.array(0)
    deemph_biquad = audio.DeemphBiquad(44100)
    with pytest.raises(RuntimeError, match=".*DeemphBiquad:.*Expecting tensor in shape "
                                           "of <..., time>. But got tensor with dimension 0."):
        deemph_biquad(waveform)

    # mindspore eager mode acc testcase:deemph_biquad
    waveform = np.array([1, 2, 3, 6, 5, 4])
    deemph_biquad = audio.DeemphBiquad(44100)
    with pytest.raises(RuntimeError, match=".*DeemphBiquad:.*Expecting tensor in type of"
                                           ".*float, double.*But got type int*."):
        deemph_biquad(waveform)

    # mindspore eager mode acc testcase:deemph_biquad
    with pytest.raises(ValueError, match="Argument sample_rate should be 44100 or 48000, but got 44101."):
        audio.DeemphBiquad(44101)

    # mindspore eager mode acc testcase:deemph_biquad
    with pytest.raises(TypeError, match="Argument sample_rate with value 44100.0 is not of "
                                        "type \\[<class 'int'>\\], but got <class 'float'>."):
        audio.DeemphBiquad(44100.0)

    # mindspore eager mode acc testcase:deemph_biquad
    with pytest.raises(TypeError, match="Argument sample_rate with value 44100 is not of "
                                        "type \\[<class 'int'>\\], but got <class 'str'>."):
        audio.DeemphBiquad("44100")

    # mindspore eager mode acc testcase:deemph_biquad
    with pytest.raises(TypeError, match="Argument sample_rate with value \\[44100\\] is not of "
                                        "type \\[<class 'int'>\\], but got <class 'list'>."):
        audio.DeemphBiquad([44100])


if __name__ == '__main__':
    test_deemph_biquad_eager()
    test_deemph_biquad_pipeline()
    test_invalid_input_all()
    test_deemph_biquad_transform()
    test_deemph_biquad_param_check()
