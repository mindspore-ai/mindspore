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
"""Test BandrejectBiquad."""

import numpy as np
import pytest

import mindspore.dataset as ds
from mindspore.dataset import audio
from . import count_unequal_element


def test_bandreject_biquad_eager():
    """
    Feature: BandrejectBiquad
    Description: Test BandrejectBiquad in eager mode with valid input
    Expectation: Output is equal to the expected output
    """

    # Original waveform
    waveform = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64)
    # Expect waveform
    expect_waveform = np.array([[9.802485108375549316e-01, 1.000000000000000000e+00, 1.000000000000000000e+00],
                                [1.000000000000000000e+00, 1.000000000000000000e+00, 1.000000000000000000e+00]],
                               dtype=np.float64)
    bandreject_biquad = audio.BandrejectBiquad(44100, 200.0, 0.707)
    # Filtered waveform by BandrejectBiquad
    output = bandreject_biquad(waveform)
    count_unequal_element(expect_waveform, output, 0.0001, 0.0001)


def test_bandreject_biquad_pipeline():
    """
    Feature: BandrejectBiquad
    Description: Test BandrejectBiquad in pipeline mode with valid input
    Expectation: Output is equal to the expected output
    """

    # Original waveform
    waveform = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64)
    # Expect waveform
    expect_waveform = np.array([[9.802485108375549316e-01, 1.000000000000000000e+00, 1.000000000000000000e+00],
                                [1.000000000000000000e+00, 1.000000000000000000e+00, 1.000000000000000000e+00]],
                               dtype=np.float64)
    label = np.random.random_sample((2, 1))
    data = (waveform, label)
    dataset = ds.NumpySlicesDataset(data, ["channel", "sample"], shuffle=False)
    bandreject_biquad = audio.BandrejectBiquad(44100, 200.0)
    # Filtered waveform by BandrejectBiquad
    dataset = dataset.map(
        input_columns=["channel"], operations=bandreject_biquad, num_parallel_workers=8)
    i = 0
    for item in dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
        count_unequal_element(expect_waveform[i, :],
                              item['channel'], 0.0001, 0.0001)
        i += 1


def test_bandreject_biquad_invalid_input():
    """
    Feature: BandrejectBiquad
    Description: Test BandrejectBiquad with invalid input
    Expectation: Correct error and message are thrown as expected
    """

    def test_invalid_input(sample_rate, central_freq, Q, error, error_msg):
        with pytest.raises(error) as error_info:
            audio.BandrejectBiquad(sample_rate, central_freq, Q)
        assert error_msg in str(error_info.value)

    test_invalid_input(44100.5, 200, 0.707, TypeError,
                       "Argument sample_rate with value 44100.5 is not of type [<class 'int'>],"
                       " but got <class 'float'>.")
    test_invalid_input("44100", 200, 0.707, TypeError,
                       "Argument sample_rate with value 44100 is not of type [<class 'int'>], but got <class 'str'>.")
    test_invalid_input(44100, "200", 0.707, TypeError,
                       "Argument central_freq with value 200 is not of type [<class 'float'>, <class 'int'>],"
                       " but got <class 'str'>.")
    test_invalid_input(0, 200, 0.707, ValueError,
                       "Input sample_rate is not within the required interval of [-2147483648, 0) and (0, 2147483647].")
    test_invalid_input(44100, 32434324324234321, 0.707, ValueError,
                       "Input central_freq is not within the required interval of [-16777216, 16777216].")
    test_invalid_input(44100, 200, "0.707", TypeError,
                       "Argument Q with value 0.707 is not of type [<class 'float'>, <class 'int'>],"
                       " but got <class 'str'>.")
    test_invalid_input(44100, 200, 1.707, ValueError,
                       "Input Q is not within the required interval of (0, 1].")
    test_invalid_input(44100, 200, 0, ValueError,
                       "Input Q is not within the required interval of (0, 1].")
    test_invalid_input(441324343243242342345300, 200, 0.707, ValueError,
                       "Input sample_rate is not within the required interval of [-2147483648, 0) and (0, 2147483647].")
    test_invalid_input(None, 200, 0.707, TypeError,
                       "Argument sample_rate with value None is not of type [<class 'int'>],"
                       " but got <class 'NoneType'>.")
    test_invalid_input(44100, None, 0.707, TypeError,
                       "Argument central_freq with value None is not of type [<class 'float'>, <class 'int'>],"
                       " but got <class 'NoneType'>.")


def test_bandreject_biquad_transform():
    """
    Feature: BandrejectBiquad
    Description: Test BandrejectBiquad with various valid input parameters and data types
    Expectation: The operation completes successfully and output values are within valid range
    """
    # test BandrejectBiquad normal
    waveform = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64)
    dataset = ds.NumpySlicesDataset(waveform, column_names=["column1"])
    bandreject_biquad = audio.BandrejectBiquad(10, 0.6)
    dataset = dataset.map(operations=bandreject_biquad)
    for data in dataset.create_dict_iterator(output_numpy=True):
        assert (data["column1"] <= 1).all()
        assert (data["column1"] >= -1).all()

    # test BandrejectBiquad normal
    waveform = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64)
    dataset = ds.NumpySlicesDataset(waveform, column_names=["column1"])
    bandreject_biquad = audio.BandrejectBiquad(2147483647, 500.6, 0.05)
    dataset = dataset.map(operations=bandreject_biquad)
    for data in dataset.create_dict_iterator(output_numpy=True):
        assert (data["column1"] <= 1).all()
        assert (data["column1"] >= -1).all()

    # test BandrejectBiquad normal
    waveform = list(np.random.randn(20, 102, 50).astype(np.float16))
    dataset = ds.NumpySlicesDataset(waveform, column_names=["column1"])
    bandreject_biquad = audio.BandrejectBiquad(1024, 16777216, 1)
    dataset = dataset.map(operations=bandreject_biquad)
    for data in dataset.create_dict_iterator(output_numpy=True):
        assert (data["column1"] <= 1).all()
        assert (data["column1"] >= -1).all()

    # test BandrejectBiquad normal
    waveform = list(np.random.randn(4, 2, 3, 3, 4))
    dataset = ds.NumpySlicesDataset(waveform, column_names=["column1"])
    bandreject_biquad = audio.BandrejectBiquad(100, 0.4, 0.3)
    dataset = dataset.map(operations=bandreject_biquad)
    for data in dataset.create_dict_iterator(output_numpy=True):
        assert (data["column1"] <= 1).all()
        assert (data["column1"] >= -1).all()

    # test BandrejectBiquad normal
    waveform = list(np.random.randn(4, 10))
    dataset = ds.NumpySlicesDataset(waveform, column_names=["column1"])
    bandreject_biquad = audio.BandrejectBiquad(-10, 0.4, 0.3)
    dataset = dataset.map(operations=bandreject_biquad)
    for _ in dataset.create_dict_iterator(output_numpy=True):
        pass

    # test BandrejectBiquad normal
    waveform = list(np.random.randn(4, 4, 3))
    dataset = ds.NumpySlicesDataset(waveform, column_names=["column1"])
    bandreject_biquad = audio.BandrejectBiquad(5000, 0, 0.8)
    dataset = dataset.map(operations=bandreject_biquad)
    for _ in dataset.create_dict_iterator(output_numpy=True):
        pass

    # test BandrejectBiquad normal
    waveform = np.random.randn(4, 4)
    bandreject_biquad = audio.BandrejectBiquad(44100, 120)
    out = bandreject_biquad(waveform)
    assert (out <= 1).all()
    assert (out >= -1).all()

    # test BandrejectBiquad normal
    waveform = np.random.randn(20, )
    bandreject_biquad = audio.BandrejectBiquad(2048, 4096, 0.808)
    bandreject_biquad(waveform)

    # test Input type is abnormal
    waveform = np.random.randint(-10, 10, (5, 5))
    bandreject_biquad = audio.BandrejectBiquad(48, 4096, 0.808)
    with pytest.raises(RuntimeError, match=".*BandrejectBiquad.*Expecting tensor in type of"
                                           " .*float, double.*But got type int*"):
        bandreject_biquad(waveform)

    # test BandrejectBiquad is normal
    waveform = np.random.randn(4, 2, 3)
    bandreject_biquad = audio.BandrejectBiquad(sample_rate=1000, central_freq=0, Q=0.5)
    bandreject_biquad(waveform)


def test_bandreject_biquad_param_check():
    """
    Feature: BandrejectBiquad
    Description: Test BandrejectBiquad with invalid input data types and parameters
    Expectation: Correct error types and messages are raised as expected
    """
    # test Input type is abnormal
    waveform = list(np.random.randn(20, ))
    with pytest.raises(TypeError, match="Input should be NumPy audio, got <class 'list'>"):
        bandreject_biquad = audio.BandrejectBiquad(2048, 4096, 0.808)
        bandreject_biquad(waveform)

    # test Input type is abnormal
    waveform = 1.5
    with pytest.raises(TypeError, match="Input should be NumPy audio, got <class 'float'>"):
        bandreject_biquad = audio.BandrejectBiquad(2048, 4096, 0.808)
        bandreject_biquad(waveform)

    # test Input type is abnormal
    waveform = np.array([["1.0", "1.1"], ["1.2", "1.3"]])
    bandreject_biquad = audio.BandrejectBiquad(2048, 40.0, 0.8)
    with pytest.raises(RuntimeError, match=".*BandrejectBiquad.*Expecting tensor in type"
                                           " of .*float, double.*But got type string.*"):
        bandreject_biquad(waveform)

    # test sample_rate is abnormal
    with pytest.raises(ValueError, match="Input sample_rate is not within the required interval"
                                         " of \\[-2147483648, 0\\) and \\(0, 2147483647\\]."):
        audio.BandrejectBiquad(sample_rate=0, central_freq=20.0, Q=1)

    # test sample_rate type is abnormal
    with pytest.raises(TypeError, match="Argument sample_rate with value 10.6 is not of "
                                        "type \\[<class 'int'>\\], but got <class 'float'>."):
        audio.BandrejectBiquad(sample_rate=10.6, central_freq=20.0, Q=1)

    # test sample_rate is abnormal
    waveform = np.random.randn(6, 8)
    with pytest.raises(ValueError, match="Input sample_rate is not within the required interval"
                                         " of \\[-2147483648, 0\\) and \\(0, 2147483647\\]."):
        bandreject_biquad = audio.BandrejectBiquad(sample_rate=2147483648, central_freq=20.0, Q=1)
        bandreject_biquad(waveform)

    # test sample_rate type is abnormal
    with pytest.raises(TypeError, match="Argument sample_rate with value \\[100\\] is not "
                                        "of type \\[<class 'int'>\\], but got <class 'list'>."):
        audio.BandrejectBiquad(sample_rate=[100], central_freq=20.0, Q=1)

    # test sample_rate type is abnormal
    with pytest.raises(TypeError, match="Argument sample_rate with value \\[100\\] is not of "
                                        "type \\[<class 'int'>\\], but got <class 'numpy.ndarray'>."):
        audio.BandrejectBiquad(sample_rate=np.array([100]), central_freq=20.0, Q=1)

    # test central_freq type is abnormal
    with pytest.raises(TypeError, match="Argument central_freq with value 20.0 is not of type"
                                        " \\[<class 'float'>, <class 'int'>\\], but got <class 'str'>."):
        audio.BandrejectBiquad(sample_rate=100, central_freq="20.0")

    # test central_freq type is abnormal
    with pytest.raises(TypeError, match="Argument central_freq with value \\[10.5\\] is not of type"
                                        " \\[<class 'float'>, <class 'int'>\\], but got <class 'list'>."):
        audio.BandrejectBiquad(sample_rate=100, central_freq=[10.5])

    # test Q type is abnormal
    with pytest.raises(TypeError, match="Argument Q with value 10 is not of type \\[<class"
                                        " 'float'>, <class 'int'>\\], but got <class 'str'>."):
        audio.BandrejectBiquad(sample_rate=100, central_freq=10.5, Q="10")

    # test Q type is abnormal
    with pytest.raises(TypeError, match="Argument Q with value \\[0.5\\] is not of type \\[<class"
                                        " 'float'>, <class 'int'>\\], but got <class 'list'>."):
        audio.BandrejectBiquad(sample_rate=100, central_freq=10.5, Q=[0.5])

    # test sample_rate  is abnormal
    with pytest.raises(ValueError, match="Input sample_rate is not within the required interval"
                                         " of \\[-2147483648, 0\\) and \\(0, 2147483647\\]"):
        audio.BandrejectBiquad(sample_rate=0, central_freq=10.5, Q=0.5)

    # test Q  is abnormal
    with pytest.raises(ValueError, match="Input Q is not within the required interval of \\(0, 1\\]."):
        audio.BandrejectBiquad(sample_rate=1000, central_freq=10.5, Q=0)

    # test Q  is abnormal
    with pytest.raises(ValueError, match="Input Q is not within the required interval of \\(0, 1\\]."):
        audio.BandrejectBiquad(sample_rate=1000, central_freq=10.5, Q=1.1)


if __name__ == "__main__":
    test_bandreject_biquad_eager()
    test_bandreject_biquad_pipeline()
    test_bandreject_biquad_invalid_input()
    test_bandreject_biquad_transform()
    test_bandreject_biquad_param_check()
