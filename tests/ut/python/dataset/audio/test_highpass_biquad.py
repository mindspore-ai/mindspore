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
"""Test HighpassBiquad."""

import numpy as np
import pytest

import mindspore.dataset as ds
from mindspore.dataset import audio
from . import count_unequal_element


def test_highpass_biquad_eager():
    """
    Feature: HighpassBiquad
    Description: Test HighpassBiquad in eager mode with valid input
    Expectation: Output is equal to the expected output
    """
    # Original waveform
    waveform = np.array([[0.8236, 0.2049, 0.3335], [0.5933, 0.9911, 0.2482],
                         [0.3007, 0.9054, 0.7598], [0.5394, 0.2842, 0.5634], [0.6363, 0.2226, 0.2288]])
    # Expect waveform
    expect_waveform = np.array([[0.2745, -0.4808, 0.1576], [0.1978, -0.0652, -0.4462],
                                [0.1002, 0.1013, -0.2835], [0.1798, -0.2649, 0.1182], [0.2121, -0.3500, 0.0693]])
    highpass_biquad = audio.HighpassBiquad(4000, 1000.0, 1)
    # Filtered waveform by HighpassBiquad
    output = highpass_biquad(waveform)
    count_unequal_element(expect_waveform, output, 0.0001, 0.0001)


def test_highpass_biquad_pipeline():
    """
    Feature: HighpassBiquad
    Description: Test HighpassBiquad in pipeline mode with valid input
    Expectation: Output is equal to the expected output
    """
    # Original waveform
    waveform = np.array([[0.4063, 0.7729, 0.2325], [0.2687, 0.1426, 0.8987],
                         [0.6914, 0.6681, 0.1783], [0.2704, 0.2680, 0.7975], [0.5880, 0.1776, 0.6323]])
    # Expect waveform
    expect_waveform = np.array([[0.1354, -0.0133, -0.3474], [0.0896, -0.1316, 0.2642],
                                [0.2305, -0.2382, -0.2323], [0.0901, -0.0909, 0.1473], [0.1960, -0.3328, 0.2230]])
    dataset = ds.NumpySlicesDataset(waveform, ["col1"], shuffle=False)
    highpass_biquad = audio.HighpassBiquad(4000, 1000.0, 1)
    # Filtered waveform by HighpassBiquad
    dataset = dataset.map(
        input_columns=["col1"], operations=highpass_biquad, num_parallel_workers=4)
    i = 0
    for item in dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
        count_unequal_element(expect_waveform[i, :],
                              item["col1"], 0.0001, 0.0001)
        i += 1


def test_highpass_biquad_invalid_input():
    """
    Feature: HighpassBiquad
    Description: Test HighpassBiquad with invalid input
    Expectation: Correct error and message are thrown as expected
    """

    def test_invalid_input(sample_rate, cutoff_freq, Q, error, error_msg):
        with pytest.raises(error) as error_info:
            audio.HighpassBiquad(sample_rate, cutoff_freq, Q)
        assert error_msg in str(error_info.value)

    test_invalid_input(44100.5, 1000, 0.707, TypeError,
                       "Argument sample_rate with value 44100.5 is not of type [<class 'int'>],"
                       " but got <class 'float'>.")
    test_invalid_input("44100", 1000, 0.707, TypeError,
                       "Argument sample_rate with value 44100 is not of type [<class 'int'>],"
                       " but got <class 'str'>.")
    test_invalid_input(44100, "1000", 0.707, TypeError,
                       "Argument cutoff_freq with value 1000 is not of type [<class 'float'>, <class 'int'>],"
                       " but got <class 'str'>.")
    test_invalid_input(44100, 1000, "0.707", TypeError,
                       "Argument Q with value 0.707 is not of type [<class 'float'>, <class 'int'>],"
                       " but got <class 'str'>.")

    test_invalid_input(441324343243242342345300, 1000, 0.707, ValueError,
                       "Input sample_rate is not within the required interval of [-2147483648, 0) and (0, 2147483647].")
    test_invalid_input(44100, 32434324324234321, 0.707, ValueError,
                       "Input cutoff_freq is not within the required interval of [-16777216, 16777216].")

    test_invalid_input(None, 1000, 0.707, TypeError,
                       "Argument sample_rate with value None is not of type [<class 'int'>], "
                       "but got <class 'NoneType'>.")
    test_invalid_input(44100, None, 0.707, TypeError,
                       "Argument cutoff_freq with value None is not of type [<class 'float'>, <class 'int'>],"
                       " but got <class 'NoneType'>.")

    test_invalid_input(0, 1000, 0.707, ValueError,
                       "Input sample_rate is not within the required interval of [-2147483648, 0) and (0, 2147483647].")
    test_invalid_input(44100, 1000, 0, ValueError,
                       "Input Q is not within the required interval of (0, 1].")


def test_highpass_biquad_transform():
    """
    Feature: HighpassBiquad
    Description: Test HighpassBiquad with various valid input parameters and data types
    Expectation: The operation completes successfully
    """
    # test HighpassBiquad normal
    waveform = np.array([[0.53059434, 0.32745672, 0.77121041], [0.2471812, 0.16274778, 0.01163962],
                         [0.67923531, 0.78052533, 0.90926096], [0.33950221, 0.50732238, 0.38346966],
                         [0.98423111, 0.54079969, 0.76682591]])
    highpass_biquad = audio.HighpassBiquad(44100, 5000.5, 0.707)
    dataset = ds.NumpySlicesDataset(waveform, ["audio"], shuffle=False)
    dataset = dataset.map(input_columns=["audio"], operations=highpass_biquad)
    for _ in dataset.create_dict_iterator(output_numpy=True):
        pass

    # test HighpassBiquad normal
    waveform = np.array([[0.53059434, 0.32745672, 0.77121041], [0.2471812, 0.16274778, 0.01163962],
                         [0.67923531, 0.78052533, 0.90926096], [0.33950221, 0.50732238, 0.38346966],
                         [0.98423111, 0.54079969, 0.76682591]], dtype=np.float32)

    highpass_biquad = audio.HighpassBiquad(4000, 1000.0, 1)
    output = highpass_biquad(waveform)
    assert output.shape == waveform.shape
    assert (output <= 1).all()
    assert (output >= -1).all()

    # test HighpassBiquad normal
    waveform = np.array([[0.53059434, 0.32745672, 0.77121041], [0.2471812, 0.16274778, 0.01163962],
                         [0.67923531, 0.78052533, 0.90926096], [0.33950221, 0.50732238, 0.38346966],
                         [0.98423111, 0.54079969, 0.76682591]], dtype=np.float64)
    highpass_biquad = audio.HighpassBiquad(4000, 1000.0, 1)
    output = highpass_biquad(waveform)
    assert output.shape == waveform.shape
    assert (output <= 1).all()
    assert (output >= -1).all()


def test_highpass_biquad_param_check():
    """
    Feature: HighpassBiquad
    Description: Test HighpassBiquad with invalid input data types and parameters
    Expectation: Correct error types and messages are raised as expected
    """

    # Verify that passing 'sample_rate' as a float raises a TypeError.
    with pytest.raises(
        TypeError,
        match=r"Argument sample_rate with value 44100.5 is not of type \[<class 'int'>\], but got <class 'float'>",
    ):
        highpass_biquad = audio.HighpassBiquad(44100.5, 5000.5, 0.707)

    # Verify that input data not being NumPy array (passing a bool) raises a TypeError.
    highpass_biquad = audio.HighpassBiquad(44100, 5000.5, 0.707)
    with pytest.raises(
        TypeError,
        match="Input should be NumPy audio, got <class 'bool'>.",
    ):
        highpass_biquad(True)

    # Verify that input waveform with an integer type array will raise a RuntimeError due to unsupported type.
    waveform = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
    highpass_biquad = audio.HighpassBiquad(44100, 5000.5, 0.707)
    with pytest.raises(
        RuntimeError,
        match=".*HighpassBiquad: the data type of input tensor does not match the requirement of operator. "
              "Expecting tensor in type of.*float, double.*But got type int32.",
    ):
        highpass_biquad(waveform)

    # Verify that input waveform with string type will raise a RuntimeError due to unsupported type.
    waveform = np.array(["a", "b", "c"])
    highpass_biquad = audio.HighpassBiquad(44100, 5000.5, 0.707)
    with pytest.raises(
        RuntimeError,
        match=".*HighpassBiquad: the data type of input tensor does not match the requirement of operator. "
              "Expecting tensor in type of.*float, double.*But got type string.",
    ):
        highpass_biquad(waveform)

    # Verify that passing 'sample_rate' as a float again raises a TypeError (repeat for waveform input isolation).
    waveform = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    with pytest.raises(
        TypeError,
        match=r"Argument sample_rate with value 44100.5 is not of type \[<class 'int'>\], but got <class 'float'>",
    ):
        highpass_biquad = audio.HighpassBiquad(44100.5, 5000.5, 0.707)

    # Verify that passing 'sample_rate' as a string raises a TypeError.
    waveform = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    with pytest.raises(
        TypeError,
        match=r"Argument sample_rate with value 44100.5 is not of type \[<class 'int'>\], but got <class 'str'>",
    ):
        highpass_biquad = audio.HighpassBiquad("44100.5", 5000.5, 0.707)

    # Verify that passing 'cutoff_freq' as a string raises a TypeError.
    waveform = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    with pytest.raises(
        TypeError,
        match=r"Argument cutoff_freq with value 5000.5 is not of type \[<class 'float'>, <class 'int'>\], "
              r"but got <class 'str'>",
    ):
        highpass_biquad = audio.HighpassBiquad(44100, "5000.5", 0.707)

    # Verify that passing 'Q' as a string raises a TypeError.
    waveform = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    with pytest.raises(
        TypeError,
        match=r"Argument Q with value 0.707 is not of type \[<class 'float'>, <class 'int'>\], but got <class 'str'>",
    ):
        highpass_biquad = audio.HighpassBiquad(44100, 5000.5, "0.707")

    # Verify that passing an extremely large 'sample_rate' value outside valid range raises a ValueError.
    waveform = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    with pytest.raises(
        ValueError,
        match=r"Input sample_rate is not within the required interval of \[-2147483648, 0\) and \(0, 2147483647\]",
    ):
        highpass_biquad = audio.HighpassBiquad(441324343243242342345300, 5000.5, 0.707)

    # Verify that passing an out-of-bounds 'cutoff_freq' value raises a ValueError.
    waveform = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    with pytest.raises(
        ValueError,
        match=r"Input cutoff_freq is not within the required interval of \[-16777216, 16777216\]",
    ):
        highpass_biquad = audio.HighpassBiquad(44100, 32434324324234321, 0.707)

    # Verify that passing 'sample_rate' as 0 (invalid value) raises a ValueError.
    waveform = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    with pytest.raises(
        ValueError,
        match=r"Input sample_rate is not within the required interval of \[-2147483648, 0\) and \(0, 2147483647\]",
    ):
        highpass_biquad = audio.HighpassBiquad(0, 5000.5, 0.707)

    # Verify that passing 'Q' outside the allowed interval (e.g., 0) raises a ValueError.
    waveform = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    with pytest.raises(
        ValueError,
        match=r"Input Q is not within the required interval of \(0, 1\]",
    ):
        highpass_biquad = audio.HighpassBiquad(44100, 5000.5, 0)


if __name__ == "__main__":
    test_highpass_biquad_eager()
    test_highpass_biquad_pipeline()
    test_highpass_biquad_invalid_input()
    test_highpass_biquad_transform()
    test_highpass_biquad_param_check()
