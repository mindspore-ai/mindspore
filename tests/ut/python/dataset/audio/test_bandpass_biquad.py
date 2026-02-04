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
"""Test BandpassBiquad."""

import numpy as np
import pytest

import mindspore.dataset as ds
from mindspore.dataset import audio
from . import count_unequal_element


def test_bandpass_biquad_eager():
    """
    Feature: BandpassBiquad
    Description: Test BandpassBiquad in eager mode with valid input
    Expectation: Output is equal to the expected output
    """

    # Original waveform
    waveform = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64)
    # Expect waveform
    expect_waveform = np.array([[0.01979545, 0.07838227, 0.17417782],
                                [0.07918181, 0.25414270, 0.46156447]], dtype=np.float64)
    bandpass_biquad = audio.BandpassBiquad(44000, 200.0, 0.707, False)
    # Filtered waveform by BandpassBiquad
    output = bandpass_biquad(waveform)
    count_unequal_element(expect_waveform, output, 0.0001, 0.0001)


def test_bandpass_biquad_pipeline():
    """
    Feature: BandpassBiquad
    Description: Test BandpassBiquad in pipeline mode with valid input
    Expectation: Output is equal to the expected output
    """

    # Original waveform
    waveform = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64)
    # Expect waveform
    expect_waveform = np.array([[0.01979545, 0.07838227, 0.17417782],
                                [0.07918181, 0.25414270, 0.46156447]], dtype=np.float64)
    label = np.random.sample((2, 1))
    data = (waveform, label)
    dataset = ds.NumpySlicesDataset(data, ["channel", "sample"], shuffle=False)
    bandpass_biquad = audio.BandpassBiquad(44000, 200.0)
    # Filtered waveform by BandpassBiquad
    dataset = dataset.map(
        input_columns=["channel"], operations=bandpass_biquad, num_parallel_workers=8)
    i = 0
    for item in dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
        count_unequal_element(
            expect_waveform[i, :], item['channel'], 0.0001, 0.0001)
        i += 1


def test_bandpass_biquad_invalid_input():
    """
    Feature: BandpassBiquad
    Description: Test BandpassBiquad with invalid input
    Expectation: Correct error and message are thrown as expected
    """
    def test_invalid_input(sample_rate, central_freq, Q, const_skirt_gain, error, error_msg):
        with pytest.raises(error) as error_info:
            audio.BandpassBiquad(
                sample_rate, central_freq, Q, const_skirt_gain)
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
                       "Argument const_skirt_gain with value False is not of type [<class 'bool'>], " +
                       "but got <class 'str'>.")


def test_bandpass_biquad_transform():
    """
    Feature: BandpassBiquad
    Description: Test BandpassBiquad with various valid input parameters and data types
    Expectation: The operation completes successfully and output values are within valid range
    """
    waveform = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64)
    dataset = ds.NumpySlicesDataset(waveform, column_names=["column1"])
    allpass_biquad = audio.BandpassBiquad(10, 0.6, 0.3, True)
    dataset = dataset.map(operations=allpass_biquad)
    for data in dataset.create_dict_iterator(output_numpy=True):
        assert (data["column1"] <= 1).all()
        assert (data["column1"] >= -1).all()

    # Test of float64 type
    waveform = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64)
    dataset = ds.NumpySlicesDataset(waveform, column_names=["column1"])
    allpass_biquad = audio.BandpassBiquad(2147483647, 500.6, 0.05, True)
    dataset = dataset.map(operations=allpass_biquad)
    for data in dataset.create_dict_iterator(output_numpy=True):
        assert (data["column1"] <= 1).all()
        assert (data["column1"] >= -1).all()

    # Test of float16 type
    waveform = list(np.random.randn(12, 18, 50).astype(np.float16))
    dataset = ds.NumpySlicesDataset(waveform, column_names=["column1"])
    allpass_biquad = audio.BandpassBiquad(1024, 16777216, 1, False)
    dataset = dataset.map(operations=allpass_biquad)
    for data in dataset.create_dict_iterator(output_numpy=True):
        assert (data["column1"] <= 1).all()
        assert (data["column1"] >= -1).all()

    # Test with 4D/5D input
    waveform = list(np.random.randn(4, 2, 3, 3, 4))
    dataset = ds.NumpySlicesDataset(waveform, column_names=["column1"])
    allpass_biquad = audio.BandpassBiquad(100, 0.4, 0.3)
    dataset = dataset.map(operations=allpass_biquad)
    for data in dataset.create_dict_iterator(output_numpy=True):
        assert (data["column1"] <= 1).all()
        assert (data["column1"] >= -1).all()

    # Test with 2D input using custom gain value
    waveform = np.random.randn(4, 4)
    allpass_biquad = audio.BandpassBiquad(44100, 120, const_skirt_gain=True)
    out = allpass_biquad(waveform)
    assert (out <= 1).all()
    assert (out >= -1).all()


def test_bandpass_biquad_param_check():
    """
    Feature: BandpassBiquad
    Description: Test BandpassBiquad with invalid input data types and parameters
    Expectation: Correct error types and messages are raised as expected
    """
    waveform = list(np.random.randn(6, 3, 8))
    allpass_biquad = audio.BandpassBiquad(500, 20.0, 1)
    error_msg = "Input should be NumPy audio, got <class 'list'>."
    with pytest.raises(TypeError, match=error_msg):
        allpass_biquad(waveform)

    # Test with invalid type parameter (TypeError expected)
    waveform = tuple(np.random.randn(20,))
    allpass_biquad = audio.BandpassBiquad(2048, 4096, 0.808, True)
    error_msg = "Input should be NumPy audio, got <class 'tuple'>."
    with pytest.raises(TypeError, match=error_msg):
        allpass_biquad(waveform)

    # Test with invalid type parameter (TypeError expected)
    waveform = 1.5
    allpass_biquad = audio.BandpassBiquad(2048, 4096, 0.808)
    error_msg = "Input should be NumPy audio, got <class 'float'>."
    with pytest.raises(TypeError, match=error_msg):
        allpass_biquad(waveform)

    # Test with invalid parameter (exception expected)
    waveform = np.array([["1.0", "1.1"], ["1.2", "1.3"]])
    allpass_biquad = audio.BandpassBiquad(2048, 40.0, 0.8, False)
    with pytest.raises(RuntimeError, match=r"BandpassBiquad: the data type of input tensor does "
                                           r"not match the requirement of operator. Expecting tensor in type "
                                           r"of \[float, double\]. But got type string."):
        allpass_biquad(waveform)

    # Test with invalid type parameter (TypeError expected)
    waveform = np.random.randint(-10, 10, (5, 5))
    allpass_biquad = audio.BandpassBiquad(48, 4096)
    with pytest.raises(RuntimeError, match=r"BandpassBiquad: the data type of input tensor does "
                                           r"not match the requirement of operator. Expecting tensor in type "
                                           r"of \[float, double\]. But got type int"):
        allpass_biquad(waveform)

    # Test with invalid type parameter (TypeError expected)
    with pytest.raises(ValueError, match=r"Input sample_rate is not within the required interval "
                                         r"of \[-2147483648, 0\) and \(0, 2147483647\]."):
        audio.BandpassBiquad(sample_rate=0, central_freq=20.0, Q=1)

    # Test with invalid type parameter (TypeError expected)
    with pytest.raises(TypeError, match="Argument sample_rate with value 10.6 is not of "
                                        "type \\[<class 'int'>\\], but got <class 'float'>."):
        audio.BandpassBiquad(sample_rate=10.6, central_freq=20.0)

    # Test with invalid type parameter (TypeError expected)
    with pytest.raises(ValueError, match=r"Input sample_rate is not within the required interval "
                                         r"of \[-2147483648, 0\) and \(0, 2147483647\]."):
        audio.BandpassBiquad(sample_rate=2147483648, central_freq=20.0, Q=0.5)

    # Test with invalid type parameter (TypeError expected)
    with pytest.raises(TypeError, match="Argument sample_rate with value \\[100\\] is not of "
                                        "type \\[<class 'int'>\\], but got <class 'list'>."):
        audio.BandpassBiquad(sample_rate=[100], central_freq=20.0, Q=0.5)

    # Test with invalid type parameter (TypeError expected)
    with pytest.raises(TypeError, match="Argument sample_rate with value \\[100\\] is not of "
                                        "type \\[<class 'int'>\\], but got <class 'numpy.ndarray'>."):
        audio.BandpassBiquad(sample_rate=np.array([100]), central_freq=20.0, Q=0.5)

    # Test with invalid type parameter (TypeError expected)
    with pytest.raises(TypeError, match="Argument central_freq with value 20.0 is not of type"
                                        " \\[<class 'float'>, <class 'int'>\\], but got <class 'str'>."):
        audio.BandpassBiquad(sample_rate=100, central_freq="20.0")

    # Test with invalid type parameter (TypeError expected)
    with pytest.raises(TypeError, match="Argument central_freq with value \\[10.5\\] is not of type "
                                        "\\[<class 'float'>, <class 'int'>\\], but got <class 'list'>."):
        audio.BandpassBiquad(sample_rate=100, central_freq=[10.5])

    # Test with invalid type parameter (TypeError expected)
    with pytest.raises(TypeError, match="Argument Q with value 10 is not of type \\[<class "
                                        "'float'>, <class 'int'>\\], but got <class 'str'>."):
        audio.BandpassBiquad(sample_rate=100, central_freq=10.5, Q="10")

    # Test with invalid type parameter (TypeError expected)
    with pytest.raises(TypeError, match="Argument Q with value \\[0.5\\] is not of type \\[<class "
                                        "'float'>, <class 'int'>\\], but got <class 'list'>."):
        audio.BandpassBiquad(sample_rate=100, central_freq=10.5, Q=[0.5])

    # Test with invalid type parameter (TypeError expected)
    with pytest.raises(ValueError, match="Input Q is not within the required interval of \\(0, 1\\]."):
        audio.BandpassBiquad(sample_rate=100, central_freq=10.5, Q=0)

    # Test with invalid type parameter (TypeError expected)
    with pytest.raises(TypeError, match="Argument const_skirt_gain with value True is not of"
                                        " type \\[<class 'bool'>\\], but got <class 'str'>."):
        audio.BandpassBiquad(100, 10.5, 0.6, "True")

    # Test with invalid type parameter (TypeError expected)
    with pytest.raises(TypeError, match="Argument const_skirt_gain with value 1 is not of "
                                        "type \\[<class 'bool'>\\], but got <class 'int'>."):
        audio.BandpassBiquad(100, 10.5, 0.6, 1)

    # Test with invalid value parameter (ValueError expected)
    with pytest.raises(ValueError, match=r"Input sample_rate is not within the required interval "
                                         r"of \[-2147483648, 0\) and \(0, 2147483647\]."):
        audio.BandpassBiquad(sample_rate=0, central_freq=10.5, Q=0.5, const_skirt_gain=True)

    # Test with invalid value parameter (ValueError expected)
    with pytest.raises(ValueError, match="Input central_freq is not within the required "
                                         "interval of \\[-16777216, 16777216\\]."):
        audio.BandpassBiquad(sample_rate=1000, central_freq=-16777217, Q=0.5, const_skirt_gain=True)

    # Test with invalid value parameter (ValueError expected)
    with pytest.raises(ValueError, match="Input central_freq is not within the required "
                                         "interval of \\[-16777216, 16777216\\]."):
        audio.BandpassBiquad(1024, 16777217, 1, False)


if __name__ == "__main__":
    test_bandpass_biquad_eager()
    test_bandpass_biquad_pipeline()
    test_bandpass_biquad_invalid_input()
    test_bandpass_biquad_transform()
    test_bandpass_biquad_param_check()
