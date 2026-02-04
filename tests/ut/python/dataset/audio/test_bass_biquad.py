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
"""Test BassBiquad."""

import numpy as np
import pytest

import mindspore.dataset as ds
from mindspore.dataset import audio
from . import count_unequal_element


def test_bass_biquad_eager():
    """
    Feature: BassBiquad
    Description: Test BassBiquad in eager mode with valid input
    Expectation: Output is equal to the expected output
    """

    # Original waveform
    waveform = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=np.float64)
    # Expect waveform
    expect_waveform = np.array(
        [
            [0.10409035359, 0.21652136269, 0.33761211292],
            [0.41636141439, 0.55381438997, 0.70088436361],
        ],
        dtype=np.float64,
    )
    bass_biquad = audio.BassBiquad(44100, 50.0, 100.0, 0.707)
    # Filtered waveform by BassBiquad
    output = bass_biquad(waveform)
    count_unequal_element(expect_waveform, output, 0.0001, 0.0001)


def test_bass_biquad_pipeline():
    """
    Feature: BassBiquad
    Description: Test BassBiquad in pipeline mode with valid input
    Expectation: Output is equal to the expected output
    """

    # Original waveform
    waveform = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=np.float64)
    # Expect waveform
    expect_waveform = np.array(
        [
            [0.10409035359, 0.21652136269, 0.33761211292],
            [0.41636141439, 0.55381438997, 0.70088436361],
        ],
        dtype=np.float64,
    )
    label = np.random.random_sample((2, 1))
    data = (waveform, label)
    dataset = ds.NumpySlicesDataset(data, ["channel", "sample"], shuffle=False)
    bass_biquad = audio.BassBiquad(44100, 50, 100.0, 0.707)
    # Filtered waveform by BassBiquad
    dataset = dataset.map(
        input_columns=["channel"], operations=bass_biquad, num_parallel_workers=8
    )
    i = 0
    for item in dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
        count_unequal_element(expect_waveform[i, :], item["channel"], 0.0001, 0.0001)
        i += 1


def test_invalid_invalid_input():
    """
    Feature: BassBiquad
    Description: Test BassBiquad with invalid input
    Expectation: Correct error and message are thrown as expected
    """

    def test_invalid_input(sample_rate, gain, central_freq, Q, error, error_msg):
        with pytest.raises(error) as error_info:
            audio.BassBiquad(sample_rate, gain, central_freq, Q)
        assert error_msg in str(error_info.value)

    test_invalid_input(44100.5, 50.0, 200, 0.707, TypeError,
                       "Argument sample_rate with value 44100.5 is not of type [<class 'int'>],"
                       " but got <class 'float'>.")
    test_invalid_input("44100", 50.0, 200, 0.707, TypeError,
                       "Argument sample_rate with value 44100 is not of type [<class 'int'>],"
                       " but got <class 'str'>.")
    test_invalid_input(44100, "50.0", 200, 0.707, TypeError,
                       "Argument gain with value 50.0 is not of type [<class 'float'>, <class 'int'>],"
                       " but got <class 'str'>.")
    test_invalid_input(44100, 50.0, "200", 0.707, TypeError,
                       "Argument central_freq with value 200 is not of type [<class 'float'>, <class 'int'>],"
                       " but got <class 'str'>.")
    test_invalid_input(44100, 50.0, 200, "0.707", TypeError,
                       "Argument Q with value 0.707 is not of type [<class 'float'>, <class 'int'>],"
                       " but got <class 'str'>.")
    test_invalid_input(441324343243242342345300, 50.0, 200, 0.707, ValueError,
                       "Input sample_rate is not within the required interval of [-2147483648, 0) and (0, 2147483647].")
    test_invalid_input(44100, 32434324324234321, 200, 0.707, ValueError,
                       "Input gain is not within the required interval of [-16777216, 16777216].")
    test_invalid_input(44100, 50, 32434324324234321, 0.707, ValueError,
                       "Input central_freq is not within the required interval of [-16777216, 16777216].")
    test_invalid_input(None, 50.0, 200, 0.707, TypeError,
                       "Argument sample_rate with value None is not of type [<class 'int'>], "
                       "but got <class 'NoneType'>.")
    test_invalid_input(44100, None, 200, 0.707, TypeError,
                       "Argument gain with value None is not of type [<class 'float'>, <class 'int'>], "
                       "but got <class 'NoneType'>.")
    test_invalid_input(44100, 50.0, None, 0.707, TypeError,
                       "Argument central_freq with value None is not of type [<class 'float'>, <class 'int'>],"
                       " but got <class 'NoneType'>.")
    test_invalid_input(0, 50.0, 200, 0.707, ValueError,
                       "Input sample_rate is not within the required interval of [-2147483648, 0) and (0, 2147483647].")
    test_invalid_input(44100, 50.0, 200, 1.707, ValueError,
                       "Input Q is not within the required interval of (0, 1].")


def test_bass_biquad_transform():
    """
    Feature: BassBiquad
    Description: Test BassBiquad with various valid input parameters and data types
    Expectation: The operation completes successfully and output values are within valid range
    """
    # test bass_biquad normal
    data = np.random.randn(10, 20, 25)
    dataset = ds.NumpySlicesDataset(data, ["col"], shuffle=False)
    bass_biquad = audio.BassBiquad(22050, 100.0, 2.0, 0.3)
    dataset = dataset.map(input_columns=["col"], operations=bass_biquad)
    for _ in dataset.create_dict_iterator():
        pass

    # Test eager
    waveform = np.random.randn(
        1024,
    ).astype(np.float16)
    bass_biquad = audio.BassBiquad(2147483647, 10, 1024.0, 0.88)
    bass_biquad(waveform)

    # Test eager
    waveform = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64)
    bass_biquad = audio.BassBiquad(441, 50.0)
    output = bass_biquad(waveform)
    assert output.shape == waveform.shape
    assert (output <= 1).all()
    assert (output >= -1).all()

    # Test eager
    waveform = np.random.randn(10, 8, 6, 4, 2, 5).astype(np.float32)
    bass_biquad = audio.BassBiquad(441, 50.0, Q=0.02)
    output = bass_biquad(waveform)
    assert output.shape == (10, 8, 6, 4, 2, 5)
    assert (output <= 1).all()
    assert (output >= -1).all()

    # Test eager
    waveform = np.random.randn(6, 5, 4).astype(np.float32)
    bass_biquad = audio.BassBiquad(-3000, -150.0, -0.32, 1)
    bass_biquad(waveform)

    # Test eager
    waveform = np.random.randn(6, 5, 4).astype(np.float32)
    bass_biquad = audio.BassBiquad(3000, 0, 0)
    bass_biquad(waveform)

    # Test eager
    waveform = np.random.randn(5, 8, 6, 4).astype(np.float16)
    bass_biquad = audio.BassBiquad(441, 0, 0, Q=0.02)
    output = bass_biquad(waveform)
    assert output.shape == (5, 8, 6, 4)
    assert (output <= 1).all()
    assert (output >= -1).all()


def test_bass_biquad_param_check():
    """
    Feature: BassBiquad
    Description: Test BassBiquad with invalid input data types and parameters
    Expectation: Correct error types and messages are raised as expected
    """
    # Test eager
    waveform = np.random.randn(10, 8).astype(np.int32)
    bass_biquad = audio.BassBiquad(44100, 50.0, 200, 0.707)
    with pytest.raises(
        RuntimeError,
        match=".*BassBiquad.*Expecting tensor in type"
        " of .*float, double.*But got type int32.*",
    ):
        bass_biquad(waveform)

    # Test eager
    waveform = 10.6
    bass_biquad = audio.BassBiquad(44100, 50.0, 200, 0.707)
    with pytest.raises(
        TypeError, match="Input should be NumPy audio, got <class 'float'>."
    ):
        bass_biquad(waveform)

    # Test eager
    waveform = list(np.random.randn(10, 8))
    bass_biquad = audio.BassBiquad(44100, 50.0, 200, 0.707)
    with pytest.raises(
        TypeError, match="Input should be NumPy audio, got <class 'list'>."
    ):
        bass_biquad(waveform)

    # Test eager
    waveform = np.array([["1", "2"], ["3", "4"]])
    bass_biquad = audio.BassBiquad(44100, 50.0, 200, 0.707)
    with pytest.raises(
        RuntimeError,
        match=".*BassBiquad.*Expecting tensor in type"
        " of .*float, double.*But got type string.*",
    ):
        bass_biquad(waveform)

    # Test eager
    waveform = np.array(1)
    bass_biquad = audio.BassBiquad(44100, 50.0, 200, 0.707)
    with pytest.raises(
        RuntimeError,
        match=".*BassBiquad: .*Expecting tensor in shape "
        "of <..., time>. But got tensor with dimension 0.",
    ):
        bass_biquad(waveform)

    # Test eager
    with pytest.raises(
        TypeError,
        match="Argument sample_rate with value 441050.0 is not of type \\[<class 'int'>\\], but got <class 'float'>.",
    ):
        audio.BassBiquad(441050.0, 50.0, 200, 0.707)

    # Test eager
    with pytest.raises(
        ValueError,
        match="Input sample_rate is not within the required interval"
        " of \\[-2147483648, 0\\) and \\(0, 2147483647\\].",
    ):
        audio.BassBiquad(2147483648, 10, 1024.0, 0.88)

    # Test eager
    with pytest.raises(
        TypeError,
        match="Argument sample_rate with value 44100 is not of type \\[<class 'int'>\\], but got <class 'str'>.",
    ):
        audio.BassBiquad("44100", 50.0, 200, 0.707)

    # Test eager
    with pytest.raises(
        TypeError,
        match="Argument sample_rate with value \\[44100\\] is not of type \\[<class 'int'>\\], but got <class 'list'>.",
    ):
        audio.BassBiquad([44100], 50.0, 200, 0.707)

    # Test eager
    with pytest.raises(
        TypeError,
        match="Argument sample_rate with value \\[44100\\] is not of type \\[<class 'int'>\\], "
        "but got <class 'numpy.ndarray'>.",
    ):
        audio.BassBiquad(np.array([44100]), 50.0, 200, 0.707)

    # Test eager
    with pytest.raises(
        ValueError,
        match="Input sample_rate is not within the required interval"
        " of \\[-2147483648, 0\\) and \\(0, 2147483647\\].",
    ):
        audio.BassBiquad(0, 50.0, 200, 0.707)

    # Test eager
    with pytest.raises(
        TypeError,
        match="Argument gain with value 50.0 is not of type \\[<class"
        " 'float'>, <class 'int'>\\], but got <class 'str'>.",
    ):
        audio.BassBiquad(44100, "50.0", 200, 0.707)

    # Test eager
    with pytest.raises(
        TypeError,
        match="Argument gain with value \\(50.0,\\) is not of type \\[<class"
        " 'float'>, <class 'int'>\\], but got <class 'tuple'>.",
    ):
        audio.BassBiquad(44100, (50.0,), 200, 0.707)

    # Test eager
    with pytest.raises(
        ValueError,
        match="Input gain is not within the required interval of \\[-16777216, 16777216\\].",
    ):
        audio.BassBiquad(44100, 16777216.1, 200, 0.707)

    # Test eager
    with pytest.raises(
        ValueError,
        match="Input gain is not within the required interval of \\[-16777216, 16777216\\].",
    ):
        audio.BassBiquad(44100, -16777216.1, 200, 0.707)

    # Test eager
    with pytest.raises(
        TypeError,
        match="Argument central_freq with value 200 is not of type \\[<class"
        " 'float'>, <class 'int'>\\], but got <class 'str'>.",
    ):
        audio.BassBiquad(44100, 50.0, "200", 0.707)

    # Test eager
    with pytest.raises(
        TypeError,
        match="Argument central_freq with value \\[200\\] is not of type "
        "\\[<class 'float'>, <class 'int'>\\], but got <class 'list'>.",
    ):
        audio.BassBiquad(44100, 50.0, [200], 0.707)

    # Test eager
    with pytest.raises(
        ValueError,
        match="Input central_freq is not within the required interval of \\[-16777216, 16777216\\].",
    ):
        audio.BassBiquad(44100, 50.0, 16777216.1, 0.707)

    # Test eager
    with pytest.raises(
        ValueError,
        match="Input central_freq is not within the required interval of \\[-16777216, 16777216\\].",
    ):
        audio.BassBiquad(44100, 50.0, 916777216.1, 0.707)

    # Test eager
    with pytest.raises(
        TypeError,
        match="Argument central_freq with value True is not of type \\(<class"
        " 'float'>, <class 'int'>\\), but got <class 'bool'>.",
    ):
        audio.BassBiquad(44100, 50.0, True, 0.707)

    # Test eager
    with pytest.raises(
        TypeError,
        match="Argument Q with value 0.707 is not of type \\[<class"
        " 'float'>, <class 'int'>\\], but got <class 'str'>.",
    ):
        audio.BassBiquad(44100, 50.0, 200, "0.707")

    # Test eager
    with pytest.raises(
        TypeError,
        match="Argument Q with value \\[0.707\\] is not of type \\[<class"
        " 'float'>, <class 'int'>\\], but got <class 'numpy.ndarray'>.",
    ):
        audio.BassBiquad(44100, 50.0, 200, np.array([0.707]))

    # Test eager
    with pytest.raises(
        ValueError, match="Input Q is not within the required interval of \\(0, 1\\]."
    ):
        audio.BassBiquad(44100, 50.0, 200, 0)

    # Test eager
    with pytest.raises(
        ValueError, match="Input Q is not within the required interval of \\(0, 1\\]."
    ):
        audio.BassBiquad(44100, 50.0, 200, 1.1)

    # Test eager
    with pytest.raises(
        TypeError,
        match="Argument Q with value False is not of type \\(<class "
        "'float'>, <class 'int'>\\), but got <class 'bool'>.",
    ):
        audio.BassBiquad(44100, 50.0, 200, False)

    # Test eager
    with pytest.raises(
        TypeError,
        match="Argument sample_rate with value None is not of "
        "type \\[<class 'int'>\\], but got <class 'NoneType'>.",
    ):
        audio.BassBiquad(None, 50.0, 200, 0.707)

    # Test eager
    with pytest.raises(
        TypeError,
        match="Argument gain with value None is not of type \\[<class"
        " 'float'>, <class 'int'>\\], but got <class 'NoneType'>.",
    ):
        audio.BassBiquad(44100, None, 200, 0.707)


if __name__ == "__main__":
    test_bass_biquad_eager()
    test_bass_biquad_pipeline()
    test_invalid_invalid_input()
    test_bass_biquad_transform()
    test_bass_biquad_param_check()
