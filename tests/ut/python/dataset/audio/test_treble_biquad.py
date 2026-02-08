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
"""Test TrebleBiquad."""

import numpy as np
import pytest

import mindspore.dataset as ds
from mindspore.dataset import audio
from . import count_unequal_element


def test_treble_biquad_eager():
    """
    Feature: TrebleBiquad
    Description: Test TrebleBiquad in eager mode under normal test case
    Expectation: Output is equal to the expected output
    """
    # Original waveform
    waveform = np.array([[0.234, 1.873, 0.786], [-2.673, 0.886, 1.666]], dtype=np.float64)
    # Expect waveform
    expect_waveform = np.array([[1., 1., -1.], [-1., 1., -1.]], dtype=np.float64)
    treble_biquad = audio.TrebleBiquad(44100, 200.0)
    # Filtered waveform by TrebleBiquad
    output = treble_biquad(waveform)
    count_unequal_element(expect_waveform, output, 0.0001, 0.0001)


def test_treble_biquad_pipeline():
    """
    Feature: TrebleBiquad
    Description: Test TrebleBiquad in pipeline mode under normal test case
    Expectation: Output is equal to the expected output
    """
    # Original waveform
    waveform = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64)
    # Expect waveform
    expect_waveform = np.array([[1., -1., 1.], [1., -1., 1.]], dtype=np.float64)
    dataset = ds.NumpySlicesDataset(waveform, ["waveform"], shuffle=False)
    treble_biquad = audio.TrebleBiquad(44100, 200.0)
    # Filtered waveform by TrebleBiquad
    dataset = dataset.map(input_columns=["waveform"], operations=treble_biquad)
    i = 0
    for item in dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
        count_unequal_element(expect_waveform[i, :], item['waveform'], 0.0001, 0.0001)
        i += 1


def test_treble_biquad_invalid_input():
    """
    Feature: TrebleBiquad
    Description: Test TrebleBiquad with invalid input
    Expectation: Error is raised as expected
    """

    def test_invalid_input(sample_rate, gain, central_freq, Q, error, error_msg):
        with pytest.raises(error) as error_info:
            audio.TrebleBiquad(sample_rate, gain, central_freq, Q)
        assert error_msg in str(error_info.value)

    test_invalid_input(44100.5, 0.2, 3000, 0.707, TypeError,
                       "Argument sample_rate with value 44100.5 is not of type [<class 'int'>],"
                       " but got <class 'float'>.")
    test_invalid_input("44100", 0.2, 3000, 0.707, TypeError,
                       "Argument sample_rate with value 44100 is not of type [<class 'int'>], "
                       "but got <class 'str'>.")
    test_invalid_input(4410, "0", 3000, 0.707, TypeError,
                       "Argument gain with value 0 is not of type [<class 'float'>, <class 'int'>],"
                       + " but got <class 'str'>.")
    test_invalid_input(4410, 0.2, None, 0.707, TypeError,
                       "Argument central_freq with value None is not of type [<class 'float'>, <class 'int'>]," +
                       " but got <class 'NoneType'>.")
    test_invalid_input(4410, 0.2, 3000, "0", TypeError,
                       "Argument Q with value 0 is not of type [<class 'float'>, <class 'int'>]," +
                       " but got <class 'str'>.")
    test_invalid_input(0, 0.2, 3000, 0.707, ValueError,
                       "Input sample_rate is not within the required interval of [-2147483648, 0) and (0, 2147483647].")
    test_invalid_input(441324343243242342345300, 0.2, 3000, 0.707, ValueError,
                       "Input sample_rate is not within the required interval of [-2147483648, 0) and (0, 2147483647].")
    test_invalid_input(44100, 32434324324234321, 3000, 0.707, ValueError,
                       "Input gain is not within the required interval of [-16777216, 16777216].")
    test_invalid_input(44100, 0.2, 32434324324234321, 0.707, ValueError,
                       "Input central_freq is not within the required interval of [-16777216, 16777216].")
    test_invalid_input(44100, 0.2, 3000, 1.707, ValueError,
                       "Input Q is not within the required interval of (0, 1].")
    test_invalid_input(44100, 0.2, 3000, 0, ValueError,
                       "Input Q is not within the required interval of (0, 1].")


def test_treble_biquad_transform():
    """
    Feature: TrebleBiquad
    Description: Test TrebleBiquad with various valid input parameters and data types
    Expectation: The operation completes successfully
    """

    # Test with 2D input (float32)
    waveform = np.random.randn(30, 100).astype(np.float32)
    dataset = ds.NumpySlicesDataset(waveform, column_names=["column1"])
    treble_biquad = audio.TrebleBiquad(-10240, -1.868, 12.8, 0.05)
    dataset = dataset.map(operations=treble_biquad)
    for data in dataset.create_dict_iterator(output_numpy=True):
        assert (data["column1"] <= 1).all()
        assert (data["column1"] >= -1).all()

    # Test with 3D input (float16)
    waveform = np.random.randn(5, 30, 20).astype(np.float16)
    dataset = ds.NumpySlicesDataset(waveform, column_names=["column1"])
    treble_biquad = audio.TrebleBiquad(1024, 100.65, -2000, 1)
    dataset = dataset.map(operations=treble_biquad)
    for _ in dataset.create_dict_iterator(output_numpy=True):
        pass

    # Test with 4D+ input
    waveform = np.random.randn(4, 2, 3, 3, 4)
    dataset = ds.NumpySlicesDataset(waveform, column_names=["column1"])
    treble_biquad = audio.TrebleBiquad(10000, 0.4, 0.6, 0.3)
    dataset = dataset.map(operations=treble_biquad)
    for data in dataset.create_dict_iterator(output_numpy=True):
        assert (data["column1"] <= 1).all()
        assert (data["column1"] >= -1).all()

    # Test with 3D input
    waveform = np.random.randn(4, 4, 3)
    dataset = ds.NumpySlicesDataset(waveform, column_names=["column1"])
    treble_biquad = audio.TrebleBiquad(5, -0.20, 0, 0.8)
    dataset = dataset.map(operations=treble_biquad)
    for _ in dataset.create_dict_iterator(output_numpy=True):
        pass

    # Test with 2D input
    waveform = np.random.randn(4, 4)
    treble_biquad = audio.TrebleBiquad(44100, 0, 0, 0.6)
    output = treble_biquad(waveform)
    assert (output <= 1).all()
    assert (output >= -1).all()

    # Test with 1D input
    waveform = np.random.randn(128)
    treble_biquad = audio.TrebleBiquad(2147483647, 16777216, 16777216, 1)
    treble_biquad(waveform)

    # Test with 1D input
    waveform = np.random.randn(128)
    treble_biquad = audio.TrebleBiquad(-2147483647, -16777216, -16777216, 0.0006)
    treble_biquad(waveform)

    # Test with 3D input
    waveform = np.random.randn(6, 3, 8)
    treble_biquad = audio.TrebleBiquad(500, -20.0)
    output = treble_biquad(waveform)
    assert (output <= 1).all()
    assert (output >= -1).all()

    # Test with 1D input
    waveform = np.random.randn(20, )
    treble_biquad = audio.TrebleBiquad(2048, 2.5, 200, 0.808)
    output = treble_biquad(waveform)
    assert (output <= 1).all()
    assert (output >= -1).all()


def test_treble_biquad_param_check():
    """
    Feature: TrebleBiquad
    Description: Test TrebleBiquad with invalid input data types and parameters
    Expectation: Correct error types and messages are raised as expected
    """

    # Test with valid parameters
    waveform = 1.5
    treble_biquad = audio.TrebleBiquad(2048, 100, 4096, 0.808)
    with pytest.raises(TypeError, match="Input should be NumPy audio, got <class 'float'>."):
        treble_biquad(waveform)

    # Test with string input (invalid type)
    waveform = np.array([["1.0", "1.1"], ["1.2", "1.3"]])
    treble_biquad = audio.TrebleBiquad(2048, 100, 40.0, 0.8)
    with pytest.raises(RuntimeError, match="the data type of input tensor does not match the requirement of operator."
                                           " Expecting tensor in type of \\[float, double\\]. But got type string."):
        treble_biquad(waveform)

    # Test with exception
    waveform = np.random.randint(-10, 10, (5, 5))
    treble_biquad = audio.TrebleBiquad(48, 100, 4096, 0.808)
    with pytest.raises(RuntimeError, match="the data type of input tensor does not match the requirement of operator."
                                           " Expecting tensor in type of \\[float, double\\]. But got type int*."):
        treble_biquad(waveform)

    # Test with invalid value using gain=200, sample_rate=0
    with pytest.raises(ValueError, match="Input sample_rate is not within the required interval "
                                         "of \\[-2147483648, 0\\) and \\(0, 2147483647\\]."):
        audio.TrebleBiquad(sample_rate=0, gain=200)

    # Test with invalid type using gain=200, sample_rate=100.0, central_freq parameter, Q parameter
    with pytest.raises(TypeError, match="Argument sample_rate with value 100.0 is not "
                                        "of type \\[<class 'int'>\\], but got <class 'float'>."):
        audio.TrebleBiquad(sample_rate=100.0, gain=200, central_freq=20.0, Q=1)

    # Test with invalid value with 2D input using gain=200, sample_rate=2147483648, central_freq parameter, Q parameter
    waveform = np.random.randn(6, 8)
    with pytest.raises(ValueError, match="Input sample_rate is not within the required "
                                         "interval of \\[-2147483648, 0\\) and \\(0, 2147483647\\]."):
        treble_biquad = audio.TrebleBiquad(sample_rate=2147483648, gain=200, central_freq=20.0, Q=1)
        treble_biquad(waveform)

    # Test with invalid type using gain=200, sample_rate=[100], central_freq parameter, Q parameter
    with pytest.raises(TypeError, match="Argument sample_rate with value \\[100\\] is not of "
                                        "type \\[<class 'int'>\\], but got <class 'list'>."):
        audio.TrebleBiquad(sample_rate=[100], gain=200, central_freq=20.0, Q=1)

    # Test with invalid type with custom array input using gain=200, sample_rate=np.array([100], central_freq parameter, Q parameter
    with pytest.raises(TypeError, match=r"Argument sample_rate with value \[100\] is not of "
                                        r"type \[<class 'int'>\], but got <class 'numpy.ndarray'>."):
        audio.TrebleBiquad(sample_rate=np.array([100]), gain=200, central_freq=20.0, Q=1)

    # Test with invalid value using gain=16777216.1, sample_rate=100
    with pytest.raises(ValueError, match=r"Input gain is not within the required interval of \[-16777216, 16777216\]."):
        audio.TrebleBiquad(sample_rate=100, gain=16777216.1)

    # Test with invalid value using gain=-16777216.1, sample_rate=100
    with pytest.raises(ValueError, match=r"Input gain is not within the required interval of \[-16777216, 16777216\]."):
        audio.TrebleBiquad(sample_rate=100, gain=-16777216.1)

    # Test with invalid type using gain="2.6", sample_rate=100
    with pytest.raises(TypeError, match=r"Argument gain with value 2.6 is not of type \[<class"
                                        r" 'float'>, <class 'int'>\], but got <class 'str'>."):
        audio.TrebleBiquad(sample_rate=100, gain="2.6")

    # Test with invalid type using gain=True, sample_rate=100
    with pytest.raises(TypeError, match=r"Argument gain with value True is not of type \(<class"
                                        r" 'float'>, <class 'int'>\), but got <class 'bool'>."):
        audio.TrebleBiquad(sample_rate=100, gain=True)

    # Test with invalid type using gain=[2.0], sample_rate=100
    with pytest.raises(TypeError, match=r"Argument gain with value \[2.0\] is not of type \[<class"
                                        r" 'float'>, <class 'int'>\], but got <class 'list'>."):
        audio.TrebleBiquad(sample_rate=100, gain=[2.0])

    # Test with invalid type using gain=None, sample_rate=100
    with pytest.raises(TypeError, match=r"Argument gain with value None is not of type \[<class"
                                        r" 'float'>, <class 'int'>\], but got <class 'NoneType'>."):
        audio.TrebleBiquad(sample_rate=100, gain=None)

    # Test with invalid type using gain=200, sample_rate=100, central_freq parameter
    with pytest.raises(TypeError, match=r"Argument central_freq with value 20.0 is not of type "
                                        r"\[<class 'float'>, <class 'int'>\], but got <class 'str'>."):
        audio.TrebleBiquad(sample_rate=100, gain=200, central_freq="20.0")

    # Test with invalid type using gain=200, sample_rate=100, central_freq parameter
    with pytest.raises(TypeError, match=r"Argument central_freq with value \(10.5,\) is not of type"
                                        r" \[<class 'float'>, <class 'int'>\], but got <class 'tuple'>."):
        audio.TrebleBiquad(sample_rate=100, gain=200, central_freq=(10.5,))

    # Test with invalid type using gain=200, sample_rate=100, central_freq parameter
    with pytest.raises(TypeError, match=r"Argument central_freq with value False is not of type"
                                        r" \(<class 'float'>, <class 'int'>\), but got <class 'bool'>."):
        audio.TrebleBiquad(sample_rate=100, gain=200, central_freq=False)

    # Test with invalid value using gain=200, sample_rate=100, central_freq parameter
    with pytest.raises(ValueError,
                       match=r"Input central_freq is not within the required interval of \[-16777216, 16777216\]."):
        audio.TrebleBiquad(sample_rate=100, gain=200, central_freq=16777216.1)

    # Test with invalid value using gain=200, sample_rate=100, central_freq parameter
    with pytest.raises(ValueError,
                       match=r"Input central_freq is not within the required interval of \[-16777216, 16777216\]."):
        audio.TrebleBiquad(sample_rate=100, gain=200, central_freq=-16777216.1)

    # Test with invalid type using gain=200, sample_rate=100, central_freq parameter
    with pytest.raises(TypeError, match=r"Argument central_freq with value None is not of type \[<class"
                                        r" 'float'>, <class 'int'>\], but got <class 'NoneType'>."):
        audio.TrebleBiquad(sample_rate=100, gain=200, central_freq=None)

    # Test with invalid type using gain=200, sample_rate=100, central_freq parameter, Q parameter
    with pytest.raises(TypeError, match=r"Argument Q with value 10 is not of type \[<class 'float'>, "
                                        r"<class 'int'>\], but got <class 'str'>."):
        audio.TrebleBiquad(sample_rate=100, gain=200, central_freq=10.5, Q="10")

    # Test with invalid type using gain=200, sample_rate=100, central_freq parameter, Q parameter
    with pytest.raises(TypeError, match=r"Argument Q with value \[0.5\] is not of type \[<class 'float'>,"
                                        r" <class 'int'>\], but got <class 'list'>."):
        audio.TrebleBiquad(sample_rate=100, gain=200, central_freq=10.5, Q=[0.5])

    # Test with invalid value using gain=200, sample_rate=0, central_freq parameter, Q parameter
    with pytest.raises(ValueError, match="Input sample_rate is not within the required interval"
                                         " of \\[-2147483648, 0\\) and \\(0, 2147483647\\]."):
        audio.TrebleBiquad(sample_rate=0, gain=200, central_freq=10.5, Q=0.5)

    # Test with invalid value using gain=200, sample_rate=1000, central_freq parameter, Q parameter
    with pytest.raises(ValueError, match="Input Q is not within the required interval of \\(0, 1\\]."):
        audio.TrebleBiquad(sample_rate=1000, gain=200, central_freq=10.5, Q=0)

    # Test with invalid value using gain=200, sample_rate=1000, central_freq parameter, Q parameter
    with pytest.raises(ValueError, match="Input Q is not within the required interval of \\(0, 1\\]."):
        audio.TrebleBiquad(sample_rate=1000, gain=200, central_freq=10.5, Q=1.01)

    # Test with invalid type with 2D input
    waveform = list(np.random.randn(6, 8))
    treble_biquad = audio.TrebleBiquad(500, 200, 20.0, 1)
    with pytest.raises(TypeError, match="Input should be NumPy audio, got <class 'list'>."):
        treble_biquad(waveform)

    # Test with invalid type with 2D input
    waveform = tuple(np.random.randn(5, 5))
    treble_biquad = audio.TrebleBiquad(2048, 8.8, 4096, 0.808)
    with pytest.raises(TypeError, match="Input should be NumPy audio, got <class 'tuple'>."):
        treble_biquad(waveform)


if __name__ == "__main__":
    test_treble_biquad_eager()
    test_treble_biquad_pipeline()
    test_treble_biquad_invalid_input()
    test_treble_biquad_transform()
    test_treble_biquad_param_check()
