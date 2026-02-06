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
"""Test EqualizerBiquad."""

import numpy as np
import pytest

import mindspore.dataset as ds
from mindspore.dataset import audio
from . import count_unequal_element


def test_equalizer_biquad_eager():
    """
    Feature: EqualizerBiquad
    Description: Test EqualizerBiquad in eager mode with valid input
    Expectation: Output is equal to the expected output
    """
    # Original waveform
    waveform = np.array([[0.8236, 0.2049, 0.3335], [0.5933, 0.9911, 0.2482],
                         [0.3007, 0.9054, 0.7598], [0.5394, 0.2842, 0.5634], [0.6363, 0.2226, 0.2288]])
    # Expect waveform
    expect_waveform = np.array([[1.0000, 0.2532, 0.1273], [0.7333, 1.0000, 0.1015],
                                [0.3717, 1.0000, 0.8351], [0.6667, 0.3513, 0.5098], [0.7864, 0.2751, 0.0627]])
    equalizer_biquad = audio.EqualizerBiquad(4000, 1000.0, 5.5, 1)
    # Filtered waveform by highpass_biquad
    output = equalizer_biquad(waveform)
    count_unequal_element(expect_waveform, output, 0.0001, 0.0001)


def test_equalizer_biquad_pipeline():
    """
    Feature: EqualizerBiquad
    Description: Test EqualizerBiquad in pipeline mode with valid input
    Expectation: Output is equal to the expected output
    """
    # Original waveform
    waveform = np.array([[0.4063, 0.7729, 0.2325], [0.2687, 0.1426, 0.8987],
                         [0.6914, 0.6681, 0.1783], [0.2704, 0.2680, 0.7975], [0.5880, 0.1776, 0.6323]])
    # Expect waveform
    expect_waveform = np.array([[0.5022, 0.9553, 0.1468], [0.3321, 0.1762, 1.0000],
                                [0.8545, 0.8257, -0.0188], [0.3342, 0.3312, 0.8921], [0.7267, 0.2195, 0.5781]])
    dataset = ds.NumpySlicesDataset(waveform, ["col1"], shuffle=False)
    equalizer_biquad = audio.EqualizerBiquad(4000, 1000.0, 5.5, 1)
    # Filtered waveform by equalizer_biquad
    dataset = dataset.map(
        input_columns=["col1"], operations=equalizer_biquad, num_parallel_workers=4)
    i = 0
    for item in dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
        count_unequal_element(expect_waveform[i, :],
                              item["col1"], 0.0001, 0.0001)
        i += 1


def test_equalizer_biquad_invalid_input():
    """
    Feature: EqualizerBiquad
    Description: Test EqualizerBiquad with invalid input
    Expectation: Correct error and message are thrown as expected
    """

    def test_invalid_input(sample_rate, center_freq, gain, Q, error, error_msg):
        with pytest.raises(error) as error_info:
            audio.EqualizerBiquad(sample_rate, center_freq, gain, Q)
        assert error_msg in str(error_info.value)

    test_invalid_input(44100.5, 1000, 5.5, 0.707, TypeError,
                       "Argument sample_rate with value 44100.5 is not of type [<class 'int'>],"
                       " but got <class 'float'>.")
    test_invalid_input("44100", 1000, 5.5, 0.707, TypeError,
                       "Argument sample_rate with value 44100 is not of type [<class 'int'>],"
                       " but got <class 'str'>.")
    test_invalid_input(44100, "1000", 5.5, 0.707, TypeError,
                       "Argument central_freq with value 1000 is not of type [<class 'float'>, <class 'int'>],"
                       " but got <class 'str'>.")
    test_invalid_input(44100, 1000, "5.5", 0.707, TypeError,
                       "Argument gain with value 5.5 is not of type [<class 'float'>, <class 'int'>],"
                       " but got <class 'str'>.")
    test_invalid_input(44100, 1000, 5.5, "0.707", TypeError,
                       "Argument Q with value 0.707 is not of type [<class 'float'>, <class 'int'>],"
                       " but got <class 'str'>.")

    test_invalid_input(441324343243242342345300, 1000, 5.5, 0.707, ValueError,
                       "Input sample_rate is not within the required interval of [-2147483648, 0) and (0, 2147483647].")
    test_invalid_input(44100, 3243432434, 5.5, 0.707, ValueError,
                       "Input central_freq is not within the required interval of [-16777216, 16777216].")
    test_invalid_input(0, 1000, 5.5, 0.707, ValueError,
                       "Input sample_rate is not within the required interval of [-2147483648, 0) and (0, 2147483647].")
    test_invalid_input(44100, 1000, 5.5, 0, ValueError,
                       "Input Q is not within the required interval of (0, 1].")

    test_invalid_input(None, 1000, 5.5, 0.707, TypeError,
                       "Argument sample_rate with value None is not of type [<class 'int'>], "
                       "but got <class 'NoneType'>.")
    test_invalid_input(44100, None, 5.5, 0.707, TypeError,
                       "Argument central_freq with value None is not of type [<class 'float'>, <class 'int'>],"
                       " but got <class 'NoneType'>.")
    test_invalid_input(44100, 200, None, 0.707, TypeError,
                       "Argument gain with value None is not of type [<class 'float'>, <class 'int'>], "
                       "but got <class 'NoneType'>.")


def test_equalizer_biquad_transform():
    """
    Feature: EqualizerBiquad
    Description: Test EqualizerBiquad with various valid input parameters and data types
    Expectation: The operation completes successfully and output values are within valid range
    """
    # test EqualizerBiquad normal
    waveform = np.random.randn(10, 10, 10, 10).astype(dtype=np.float32)
    dataset = ds.NumpySlicesDataset(waveform, column_names=["column1"])
    biquad = audio.EqualizerBiquad(1100, 0.85, 1024, 1)
    dataset = dataset.map(operations=biquad)
    for data in dataset.create_dict_iterator(output_numpy=True):
        assert (np.abs(data["column1"]) <= 1).all()

    # test EqualizerBiquad normal
    waveform = np.random.randn(20, 20)
    biquad = audio.EqualizerBiquad(-12000, 0.1, 0, 0.01)
    biquad(waveform)

    # test EqualizerBiquad normal
    waveform = np.random.randn(20, 20, 20)
    biquad = audio.EqualizerBiquad(-2147483648, 0.1, 0, 0.01)
    biquad(waveform)

    # test EqualizerBiquad normal
    waveform = np.random.randn(20, 20, 20)
    biquad = audio.EqualizerBiquad(2147483647, 0, -16777216, 0.01)
    biquad(waveform)

    # test EqualizerBiquad normal
    waveform = np.random.randn(20, 20, 20)
    biquad = audio.EqualizerBiquad(44000, 16777216, 0.01)
    biquad(waveform)

    # test EqualizerBiquad normal
    waveform = np.random.randn(20, 20, 20)
    biquad = audio.EqualizerBiquad(44000, -16777216, 16777216, 0.01)
    biquad(waveform)


def test_equalizer_biquad_param_check():
    """
    Feature: EqualizerBiquad
    Description: Test EqualizerBiquad with invalid input data types and parameters
    Expectation: Correct error types and messages are raised as expected
    """
    # test data type is abnormal.
    waveform = np.array([20, 20, 20]).astype(np.int64)
    biquad = audio.EqualizerBiquad(44000, 217.0, 6.6, 0.01)
    with pytest.raises(RuntimeError, match=".*EqualizerBiquad:.*Expecting tensor in type "
                                           "of.*float, double.*But got type int64."):
        biquad(waveform)

    # test data shape is abnormal.
    waveform = np.array(20.0)
    biquad = audio.EqualizerBiquad(44000, 217.0, 6.6, 0.01)
    with pytest.raises(RuntimeError, match=".*EqualizerBiquad:.*Expecting tensor in shape of "
                                           "<..., time>. But got tensor with dimension 0."):
        biquad(waveform)

    # test data type is abnormal.
    waveform = [10.0, 10.0, 10.0]
    biquad = audio.EqualizerBiquad(44000, 217.0, 6.6, 0.01)
    with pytest.raises(TypeError, match="Input should be NumPy audio, got <class 'list'>."):
        biquad(waveform)

    # test data type is abnormal.
    waveform = 0.8
    biquad = audio.EqualizerBiquad(44000, 217.0, 6.6, 0.01)
    with pytest.raises(TypeError, match="Input should be NumPy audio, got <class 'float'>."):
        biquad(waveform)

    # test data type is abnormal.
    waveform = np.array(["20.0", "20.0"])
    biquad = audio.EqualizerBiquad(44000, 217.0, 6.6, 0.01)
    with pytest.raises(RuntimeError, match=".*EqualizerBiquad:.*Expecting tensor in type "
                                           "of.*float, double.*But got type string."):
        biquad(waveform)

    # test sample_rate  is abnormal.
    with pytest.raises(ValueError, match="Input sample_rate is not within the required interval"
                                         " of \\[-2147483648, 0\\) and \\(0, 2147483647\\]."):
        audio.EqualizerBiquad(0, -116, 1.0, 0.45)

    # test sample_rate type is abnormal.
    with pytest.raises(TypeError, match="Argument sample_rate with value 10.0 is not of "
                                        "type \\[<class 'int'>\\], but got <class 'float'>."):
        audio.EqualizerBiquad(10.0, -116, 1.0, 0.45)

    # test sample_rate type is abnormal.
    with pytest.raises(TypeError, match="Argument sample_rate with value True is not of"
                                        " type \\(<class 'int'>,\\), but got <class 'bool'>."):
        audio.EqualizerBiquad(True, -116, 1.0, 0.45)

    # test sample_rate type is abnormal.
    with pytest.raises(TypeError, match="Argument sample_rate with value \\[1000\\] is not of"
                                        " type \\[<class 'int'>\\], but got <class 'list'>."):
        audio.EqualizerBiquad([1000], -116, 1.0, 0.45)

    # test central_freq is abnormal.
    with pytest.raises(ValueError,
                       match="Input central_freq is not within the required interval of \\[-16777216, 16777216\\]."):
        audio.EqualizerBiquad(1000, -16777216.1, 1.0, 0.45)

    # test central_freq is abnormal.
    with pytest.raises(ValueError,
                       match="Input central_freq is not within the required interval of \\[-16777216, 16777216\\]."):
        audio.EqualizerBiquad(1000, 16777216.1, 1.0, 0.45)

    # test central_freq type is abnormal.
    with pytest.raises(TypeError, match="Argument central_freq with value True is not of type"
                                        " \\(<class 'float'>, <class 'int'>\\), but got <class 'bool'>."):
        audio.EqualizerBiquad(1000, True, 1.0, 0.45)

    # test central_freq type is abnormal.
    with pytest.raises(TypeError, match="Argument central_freq with value None is not of type \\[<class"
                                        " 'float'>, <class 'int'>\\], but got <class 'NoneType'>."):
        audio.EqualizerBiquad(1000, None, 1.0, 0.45)

    # test central_freq type is abnormal.
    with pytest.raises(TypeError, match="Argument sample_rate with value None is not of "
                                        "type \\[<class 'int'>\\], but got <class 'NoneType'>."):
        audio.EqualizerBiquad(None, 10.0, 1.0, 0.45)

    # test gain is abnormal.
    with pytest.raises(ValueError,
                       match="Input gain is not within the required interval of \\[-16777216, 16777216\\]."):
        audio.EqualizerBiquad(10000, 10.0, 16777216.1, 0.45)

    # test gain is abnormal.
    with pytest.raises(ValueError,
                       match="Input gain is not within the required interval of \\[-16777216, 16777216\\]."):
        audio.EqualizerBiquad(10000, 10.0, -16777216.1, 0.45)

    # test gain type is abnormal.
    with pytest.raises(TypeError, match="Argument gain with value 1.0 is not of type \\[<class"
                                        " 'float'>, <class 'int'>\\], but got <class 'str'>."):
        audio.EqualizerBiquad(10000, 10.0, "1.0", 0.45)

    # test gain type is abnormal.
    with pytest.raises(TypeError, match="Argument gain with value \\(2.5,\\) is not of type"
                                        " \\[<class 'float'>, <class 'int'>\\], but got <class 'tuple'>."):
        audio.EqualizerBiquad(10000, 10.0, (2.5,), 0.45)

    # test Q is abnormal.
    with pytest.raises(ValueError, match="Input Q is not within the required interval of \\(0, 1\\]."):
        audio.EqualizerBiquad(10000, 10.0, 2.5, 0)

    # test Q is abnormal.
    with pytest.raises(ValueError, match="Input Q is not within the required interval of \\(0, 1\\]."):
        audio.EqualizerBiquad(10000, 10.0, 2.5, 1.05)

    # test Q type is abnormal.
    with pytest.raises(TypeError, match="Argument Q with value True is not of type \\(<class"
                                        " 'float'>, <class 'int'>\\), but got <class 'bool'>."):
        audio.EqualizerBiquad(10000, 10.0, 2.5, True)

    # test Q type is abnormal.
    with pytest.raises(TypeError, match="Argument Q with value 0.65 is not of type \\[<class"
                                        " 'float'>, <class 'int'>\\], but got <class 'str'>."):
        audio.EqualizerBiquad(10000, 10.0, 2.5, "0.65")


if __name__ == "__main__":
    test_equalizer_biquad_eager()
    test_equalizer_biquad_pipeline()
    test_equalizer_biquad_invalid_input()
    test_equalizer_biquad_transform()
    test_equalizer_biquad_param_check()
