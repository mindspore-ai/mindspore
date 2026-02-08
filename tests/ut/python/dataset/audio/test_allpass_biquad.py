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
"""Test AllpassBiquad."""

import numpy as np
import pytest

import mindspore.dataset as ds
from mindspore.dataset import audio
from . import count_unequal_element


def test_allpass_biquad_eager():
    """
    Feature: AllpassBiquad
    Description: Test AllpassBiquad in eager mode with valid input
    Expectation: Output is equal to the expected output
    """

    # Original waveform
    waveform = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64)
    # Expect waveform
    expect_waveform = np.array(
        [[0.96049707, 1.0, 1.0], [1.0, 1.0, 1.0]], dtype=np.float64)
    allpass_biquad = audio.AllpassBiquad(44100, 200.0, 0.707)
    # Filtered waveform by allpassbiquad
    output = allpass_biquad(waveform)
    count_unequal_element(expect_waveform, output, 0.0001, 0.0001)


def test_allpass_biquad_pipeline():
    """
    Feature: AllpassBiquad
    Description: Test AllpassBiquad in pipeline mode with valid input
    Expectation: Output is equal to the expected output
    """

    # Original waveform
    waveform = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64)
    # Expect waveform
    expect_waveform = np.array(
        [[0.96049707, 1.0, 1.0], [1.0, 1.0, 1.0]], dtype=np.float64)
    label = np.random.random_sample((2, 1))
    data = (waveform, label)
    dataset = ds.NumpySlicesDataset(data, ["channel", "sample"], shuffle=False)
    allpass_biquad = audio.AllpassBiquad(44100, 200.0)
    # Filtered waveform by allpassbiquad
    dataset = dataset.map(
        input_columns=["channel"], operations=allpass_biquad, num_parallel_workers=8)
    i = 0
    for item in dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
        count_unequal_element(
            expect_waveform[i, :], item['channel'], 0.0001, 0.0001)
        i += 1


def test_invalid_input_all():
    """
    Feature: AllpassBiquad
    Description: Test AllpassBiquad with invalid input
    Expectation: Correct error and message are thrown as expected
    """
    waveform = np.random.rand(2, 1000)

    def test_invalid_input(sample_rate, central_freq, Q, error, error_msg):
        with pytest.raises(error) as error_info:
            audio.AllpassBiquad(sample_rate, central_freq, Q)(waveform)
        assert error_msg in str(error_info.value)

    test_invalid_input(44100.5, 200, 0.707, TypeError,
                       "Argument sample_rate with value 44100.5 is not of type [<class 'int'>],"
                       + " but got <class 'float'>.")
    test_invalid_input("44100", 200, 0.707, TypeError,
                       "Argument sample_rate with value 44100 is not of type [<class 'int'>]," +
                       " but got <class 'str'>.")
    test_invalid_input(44100, "200", 0.707, TypeError,
                       "Argument central_freq with value 200 is not of type [<class 'float'>, <class 'int'>],"
                       + " but got <class 'str'>.")
    test_invalid_input(44100, 200, "0.707", TypeError,
                       "Argument Q with value 0.707 is not of type [<class 'float'>, <class 'int'>],"
                       + " but got <class 'str'>.")
    test_invalid_input( 441324343243242342345300, 200, 0.707, ValueError,
                       "Input sample_rate is not within the required interval of [-2147483648, 0) and (0, 2147483647].")
    test_invalid_input(44100, 32434324324234321, 0.707, ValueError,
                       "Input central_freq is not within the required interval of [-16777216, 16777216].")
    test_invalid_input(None, 200, 0.707, TypeError,
                       "Argument sample_rate with value None is not of type [<class 'int'>],"
                       + " but got <class 'NoneType'>.")
    test_invalid_input(44100, None, 0.707, TypeError,
                       "Argument central_freq with value None is not of type [<class 'float'>, <class 'int'>],"
                       + " but got <class 'NoneType'>.")
    test_invalid_input(0, 200, 0.707, ValueError,
                       "Input sample_rate is not within the required interval of [-2147483648, 0) and (0, 2147483647].")
    test_invalid_input(44100, 200, 1.707, ValueError,
                       "Input Q is not within the required interval of (0, 1].")


def test_allpassbiquad_transform():
    """
    Feature: AllpassBiquad
    Description: Test AllpassBiquad with various valid input parameters and data types
    Expectation: The operation completes successfully and output values are within valid range
    """
    # test allpassbiquad is normal
    waveform = list(np.random.randn(30, 10, 20).astype(np.float32))
    dataset = ds.NumpySlicesDataset(waveform, column_names=["column1"])
    allpass_biquad = audio.AllpassBiquad(44100, 200.0, 0.707)
    dataset = dataset.map(operations=allpass_biquad)
    for data in dataset.create_dict_iterator(output_numpy=True):
        assert (data["column1"] <= 1).all()
        assert (data["column1"] >= -1).all()

    # test allpassbiquad is normal
    waveform = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64)
    dataset = ds.NumpySlicesDataset(waveform, column_names=["column1"])
    allpass_biquad = audio.AllpassBiquad(2147483647, 500.6, 0.05)
    dataset = dataset.map(operations=allpass_biquad)
    for data in dataset.create_dict_iterator(output_numpy=True):
        assert (data["column1"] <= 1).all()
        assert (data["column1"] >= -1).all()

    # test allpassbiquad is normal
    waveform = np.random.randn(5, 30, 20).astype(np.float16)
    dataset = ds.NumpySlicesDataset(waveform, column_names=["column1"])
    allpass_biquad = audio.AllpassBiquad(1024, 16777216, 1)
    dataset = dataset.map(operations=allpass_biquad)
    for data in dataset.create_dict_iterator(output_numpy=True):
        assert (data["column1"] <= 1).all()
        assert (data["column1"] >= -1).all()

    # test allpassbiquad is normal
    waveform = np.random.randn(4, 2, 3, 3, 4)
    dataset = ds.NumpySlicesDataset(waveform, column_names=["column1"])
    allpass_biquad = audio.AllpassBiquad(100, 0.4, 0.3)
    dataset = dataset.map(operations=allpass_biquad)
    for data in dataset.create_dict_iterator(output_numpy=True):
        assert (data["column1"] <= 1).all()
        assert (data["column1"] >= -1).all()

    # test allpassbiquad is normal
    waveform = np.random.randn(4, 4, 3)
    dataset = ds.NumpySlicesDataset(waveform, column_names=["column1"])
    allpass_biquad = audio.AllpassBiquad(5000, 0.6, 0.8)
    dataset = dataset.map(operations=allpass_biquad)
    for _ in dataset.create_dict_iterator(output_numpy=True):
        pass

    # test allpassbiquad is normal
    waveform = np.random.randn(4, 4)
    allpass_biquad = audio.AllpassBiquad(44100, 120)
    out = allpass_biquad(waveform)
    assert (out <= 1).all()
    assert (out >= -1).all()

    # test allpassbiquad is normal
    waveform = np.random.randn(6, 3, 8)
    allpass_biquad = audio.AllpassBiquad(500, 20.0, 1)
    out = allpass_biquad(waveform)
    assert (out <= 1).all()
    assert (out >= -1).all()

    # test allpassbiquad is normal
    waveform = np.random.randn(20, )
    allpass_biquad = audio.AllpassBiquad(2048, 4096, 0.808)
    out = allpass_biquad(waveform)
    assert (out <= 1).all()
    assert (out >= -1).all()


def test_allpassbiquad_param_check():
    """
    Feature: AllpassBiquad
    Description: Test AllpassBiquad with invalid input data types and parameters
    Expectation: Correct error types and messages are raised as expected
    """
    # test Input type is abnormal
    waveform = np.random.randint(-10, 10, (5, 5))
    allpass_biquad = audio.AllpassBiquad(48, 4096, 0.808)
    with pytest.raises(RuntimeError, match=".*AllpassBiquad: .*Expecting tensor in type "
                                           "of .*float, double.* But got.*int*"):
        allpass_biquad(waveform)

    # test Input type is abnormal
    waveform = list(np.random.randn(6, 8))
    allpass_biquad = audio.AllpassBiquad(500, 20.0, 1)
    with pytest.raises(TypeError, match="Input should be NumPy audio, got <class 'list'>."):
        allpass_biquad(waveform)

    # test Input type is abnormal
    waveform = tuple(np.random.randn(5, 5))
    allpass_biquad = audio.AllpassBiquad(2048, 4096, 0.808)
    with pytest.raises(TypeError, match="Input should be NumPy audio, got <class 'tuple'>."):
        allpass_biquad(waveform)

    # test Input type is abnormal
    waveform = 1.5
    allpass_biquad = audio.AllpassBiquad(2048, 4096, 0.808)
    with pytest.raises(TypeError, match="Input should be NumPy audio, got <class 'float'>."):
        allpass_biquad(waveform)

    # test Input type is abnormal
    waveform = np.array([["1.0", "1.1"], ["1.2", "1.3"]])
    allpass_biquad = audio.AllpassBiquad(2048, 40.0, 0.8)
    with pytest.raises(RuntimeError, match=".*AllpassBiquad:.*Expecting tensor in type "
                                           "of .*float, double.* But got.*string.*"):
        allpass_biquad(waveform)

    # test sample_rate is abnormal
    with pytest.raises(ValueError, match="Input sample_rate is not within the required interval "
                                         "of \\[-2147483648, 0\\) and \\(0, 2147483647\\]."):
        audio.AllpassBiquad(sample_rate=0, central_freq=20.0, Q=1)

    # test sample_rate type is abnormal
    with pytest.raises(TypeError, match="Argument sample_rate with value 10.6 is not "
                                        "of type \\[<class 'int'>\\], but got <class 'float'>."):
        audio.AllpassBiquad(sample_rate=10.6, central_freq=20.0, Q=1)

    # test sample_rate is abnormal
    waveform = np.random.randn(6, 8)
    with pytest.raises(ValueError, match="Input sample_rate is not within the required "
                                         "interval of \\[-2147483648, 0\\) and \\(0, 2147483647\\]."):
        allpass_biquad = audio.AllpassBiquad(sample_rate=2147483648, central_freq=20.0, Q=1)
        allpass_biquad(waveform)

    # test sample_rate type is abnormal
    with pytest.raises(TypeError, match="Argument sample_rate with value \\[100\\] is not of "
                                        "type \\[<class 'int'>\\], but got <class 'list'>."):
        audio.AllpassBiquad(sample_rate=[100], central_freq=20.0, Q=1)

    # test sample_rate type is abnormal
    with pytest.raises(TypeError, match=r"Argument sample_rate with value \[100\] is not of "
                                        r"type \[<class 'int'>\], but got <class 'numpy.ndarray'>."):
        audio.AllpassBiquad(sample_rate=np.array([100]), central_freq=20.0, Q=1)

    # test central_freq type is abnormal
    with pytest.raises(TypeError, match=r"Argument central_freq with value 20.0 is not of type "
                                        r"\[<class 'float'>, <class 'int'>\], but got <class 'str'>."):
        audio.AllpassBiquad(sample_rate=100, central_freq="20.0")

    # test central_freq type is abnormal
    with pytest.raises(TypeError, match=r"Argument central_freq with value \[10.5\] is not of type "
                                        r"\[<class 'float'>, <class 'int'>\], but got <class 'list'>."):
        audio.AllpassBiquad(sample_rate=100, central_freq=[10.5])

    # test Q type is abnormal
    with pytest.raises(TypeError, match=r"Argument Q with value 10 is not of type \[<class 'float'>, "
                                        r"<class 'int'>\], but got <class 'str'>."):
        audio.AllpassBiquad(sample_rate=100, central_freq=10.5, Q="10")

    # test Q type is abnormal
    with pytest.raises(TypeError, match=r"Argument Q with value \[0.5\] is not of type \[<class 'float'>,"
                                        r" <class 'int'>\], but got <class 'list'>."):
        audio.AllpassBiquad(sample_rate=100, central_freq=10.5, Q=[0.5])

    # test sample_rate is abnormal
    with pytest.raises(ValueError, match="Input sample_rate is not within the required interval"
                                         " of \\[-2147483648, 0\\) and \\(0, 2147483647\\]."):
        audio.AllpassBiquad(sample_rate=0, central_freq=10.5, Q=0.5)

    # test central_freq is abnormal
    with pytest.raises(RuntimeError, match=".*AllpassBiquad: central_freq can not be "
                                           "equal to zero, but got: 0.000000"):
        waveform = np.random.randn(6, 8)
        allpass_biquad = audio.AllpassBiquad(sample_rate=1000, central_freq=0, Q=0.5)
        allpass_biquad(waveform)

    # test Q is abnormal
    with pytest.raises(ValueError, match="Input Q is not within the required interval of \\(0, 1\\]."):
        audio.AllpassBiquad(sample_rate=1000, central_freq=10.5, Q=0)

    # test Q is abnormal
    with pytest.raises(ValueError, match="Input Q is not within the required interval of \\(0, 1\\]."):
        audio.AllpassBiquad(sample_rate=1000, central_freq=10.5, Q=1.01)


if __name__ == '__main__':
    test_allpass_biquad_eager()
    test_allpass_biquad_pipeline()
    test_invalid_input_all()
    test_allpassbiquad_transform()
    test_allpassbiquad_param_check()
