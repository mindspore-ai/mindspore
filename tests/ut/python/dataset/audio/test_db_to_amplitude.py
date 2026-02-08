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
"""Testing DBToAmplitude."""

import numpy as np
import pytest

import mindspore.dataset as ds
from mindspore.dataset import audio
from . import count_unequal_element


def test_db_to_amplitude_eager():
    """
    Feature: DBToAmplitude
    Description: Test DBToAmplitude in eager mode
    Expectation: The data is processed successfully
    """
    # Original waveform
    waveform = np.array([1, 2, 3, 4, 5, 6], dtype=np.float64)
    # Expect waveform
    expect_waveform = np.array([3.1698, 5.0238, 7.9621, 12.6191, 20.0000, 31.6979], dtype=np.float64)
    db_to_amplitude = audio.DBToAmplitude(2, 2)
    # Filtered waveform by DBToAmplitude
    output = db_to_amplitude(waveform)
    count_unequal_element(expect_waveform, output, 0.0001, 0.0001)


def test_db_to_amplitude_pipeline():
    """
    Feature: DBToAmplitude
    Description: Test DBToAmplitude in pipeline mode
    Expectation: The data is processed successfully
    """
    # Original waveform
    waveform = np.array([[2, 2, 3], [0.1, 0.2, 0.3]], dtype=np.float64)
    # Expect waveform
    expect_waveform = np.array([[2.5119, 2.5119, 3.9811],
                                [1.0471, 1.0965, 1.1482]], dtype=np.float64)

    dataset = ds.NumpySlicesDataset(waveform, ["audio"], shuffle=False)
    db_to_amplitude = audio.DBToAmplitude(1, 2)
    # Filtered waveform by DBToAmplitude
    dataset = dataset.map(input_columns=["audio"], operations=db_to_amplitude, num_parallel_workers=8)
    i = 0
    for item in dataset.create_dict_iterator(output_numpy=True):
        count_unequal_element(expect_waveform[i, :], item['audio'], 0.0001, 0.0001)
        i += 1


def test_db_to_amplitude_invalid_input():
    """
    Feature: DBToAmplitude
    Description: Test param check of DBToAmplitude
    Expectation: Throw correct error and message
    """
    def test_invalid_input(ref, power, error, error_msg):
        with pytest.raises(error) as error_info:
            audio.DBToAmplitude(ref, power)
        assert error_msg in str(error_info.value)

    test_invalid_input("1.0", 1.0, TypeError,
                       "Argument ref with value 1.0 is not of type [<class 'float'>, <class 'int'>],"
                       " but got <class 'str'>.")
    test_invalid_input(122323242445423534543, 1.0, ValueError,
                       "Input ref is not within the required interval of [-16777216, 16777216].")
    test_invalid_input(1.0, "1.0", TypeError,
                       "Argument power with value 1.0 is not of type [<class 'float'>, <class 'int'>],"
                       " but got <class 'str'>.")
    test_invalid_input(1.0, 1343454254325445, ValueError,
                       "Input power is not within the required interval of [-16777216, 16777216].")


def test_db_to_amplitude_transform():
    """
    Feature: DBToAmplitude
    Description: Test DBToAmplitude with various valid input parameters and data types
    Expectation: The operation completes successfully
    """
    # test DBToAmplitude normal
    waveform = np.array([[0.8236, 0.2049, 0.3335], [0.5933, 0.9911, 0.2482],
                         [0.3007, 0.9054, 0.7598], [0.5394, 0.2842, 0.5634], [0.6363, 0.2226, 0.2288]])
    db_to_amplitude = audio.DBToAmplitude(-0.5, 0.5)
    dataset1 = ds.NumpySlicesDataset(waveform, ["audio"], shuffle=False)
    dataset2 = ds.NumpySlicesDataset(waveform, ["audio"], shuffle=False)
    dataset2 = dataset2.map(input_columns=["audio"], operations=db_to_amplitude)
    for _, _ in zip(dataset1.create_dict_iterator(output_numpy=True),
                    dataset2.create_dict_iterator(output_numpy=True)):
        pass

    # test DBToAmplitude normal
    waveform = np.random.randn(10, 10, 32).astype(np.float64)
    db_to_amplitude = audio.DBToAmplitude(1.2, 1)
    dataset = ds.NumpySlicesDataset(waveform, ["audio"], shuffle=False)
    dataset = dataset.map(input_columns=["audio"], operations=db_to_amplitude)
    for _ in dataset.create_dict_iterator(output_numpy=True):
        pass

    # Test eager
    waveform = np.random.randint(-1000, 1000, (10, 10, 10))
    db_to_amplitude = audio.DBToAmplitude(0.086, 1)
    db_to_amplitude(waveform)

    # Test eager
    waveform = np.random.randn(128,)
    db_to_amplitude = audio.DBToAmplitude(4.568, -1)
    db_to_amplitude(waveform)

    # Test eager
    waveform = np.random.randn(10, 10, 8, 5)
    db_to_amplitude = audio.DBToAmplitude(-0.6, 1024)
    db_to_amplitude(waveform)

    # Test eager
    waveform = np.random.randn(10, 10)
    db_to_amplitude = audio.DBToAmplitude(0, 1)
    db_to_amplitude(waveform)

    # Test eager
    waveform = np.random.randn(10, 10)
    db_to_amplitude = audio.DBToAmplitude(10008, 0)
    db_to_amplitude(waveform)

    # Test eager
    with pytest.raises(RuntimeError, match="The op is OneToOne, can only accept one tensor as input"):
        waveform = np.array([0.5, 0.5])
        db_to_amplitude = audio.DBToAmplitude(0.5, 0.5)
        db_to_amplitude(waveform, waveform)


def test_db_to_amplitude_param_check():
    """
    Feature: DBToAmplitude
    Description: Test DBToAmplitude with invalid input data types and parameters
    Expectation: Correct error types and messages are raised as expected
    """
    # Test eager
    with pytest.raises(RuntimeError, match="the data type of input tensor does not match the requirement of operator. "
                                           "Expecting tensor in type of \\[int, float, double\\]. But got type string"):
        waveform = np.array(["0", "1"])
        db_to_amplitude = audio.DBToAmplitude(0.5, 0.5)
        db_to_amplitude(waveform)

    # Test eager
    with pytest.raises(RuntimeError, match="the shape of input tensor does not match the requirement of operator. Expe"
                                           "cting tensor in shape of <..., time>. But got tensor with dimension 0."):
        waveform = np.array(1)
        db_to_amplitude = audio.DBToAmplitude(0.5, 0.5)
        db_to_amplitude(waveform)

    # Test eager
    with pytest.raises(TypeError, match="Input should be NumPy audio, got <class 'int'>."):
        waveform = np.array(1).tolist()
        db_to_amplitude = audio.DBToAmplitude(0.5, 0.5)
        db_to_amplitude(waveform)

    # Test eager
    with pytest.raises(TypeError, match="Input should be NumPy audio, got <class 'int'>."):
        waveform = 10
        db_to_amplitude = audio.DBToAmplitude(0.5, 0.5)
        db_to_amplitude(waveform)

    # Test eager
    with pytest.raises(RuntimeError, match="Input Tensor is not valid."):
        db_to_amplitude = audio.DBToAmplitude(0.5, 0.5)
        db_to_amplitude()

    # Test eager
    with pytest.raises(TypeError, match="Argument ref with value True is not of type \\(<class"
                                        " 'float'>, <class 'int'>\\), but got <class 'bool'>."):
        audio.DBToAmplitude(True, power=0.5)

    # Test eager
    with pytest.raises(TypeError, match="Argument ref with value 1 is not of type \\[<class"
                                        " 'float'>, <class 'int'>\\], but got <class 'str'>."):
        audio.DBToAmplitude("1", power=0.5)

    # Test eager
    with pytest.raises(ValueError, match="Input ref is not within the required interval of \\[-16777216, 16777216\\]."):
        audio.DBToAmplitude(16777216.1, power=0.5)

    # Test eager
    with pytest.raises(ValueError, match="Input ref is not within the required interval of \\[-16777216, 16777216\\]."):
        audio.DBToAmplitude(-16777216.1, power=0.5)

    # Test eager
    with pytest.raises(TypeError, match="Argument ref with value \\[1\\] is not of type \\[<class"
                                        " 'float'>, <class 'int'>\\], but got <class 'list'>."):
        audio.DBToAmplitude([1], power=0.5)

    # Test eager
    with pytest.raises(TypeError, match="Argument power with value True is not of type \\(<class"
                                        " 'float'>, <class 'int'>\\), but got <class 'bool'>."):
        audio.DBToAmplitude(0.5, True)

    # Test eager
    with pytest.raises(TypeError, match="Argument power with value 0.5 is not of type \\[<class"
                                        " 'float'>, <class 'int'>\\], but got <class 'str'>."):
        audio.DBToAmplitude(0.5, "0.5")

    # Test eager
    with pytest.raises(TypeError, match="Argument power with value None is not of type \\[<class"
                                        " 'float'>, <class 'int'>\\], but got <class 'NoneType'>."):
        audio.DBToAmplitude(0.5, None)

    # Test eager
    with pytest.raises(TypeError, match="Argument power with value \\[0.5\\] is not of type"
                                        " \\[<class 'float'>, <class 'int'>\\], but got <class 'list'>."):
        audio.DBToAmplitude(0.5, [0.5])

    # Test eager
    with pytest.raises(ValueError,
                       match="Input power is not within the required interval of \\[-16777216, 16777216\\]."):
        audio.DBToAmplitude(0.5, 16777216.1)

    # Test eager
    with pytest.raises(ValueError,
                       match="Input power is not within the required interval of \\[-16777216, 16777216\\]."):
        audio.DBToAmplitude(0.5, -16777216.1)


if __name__ == "__main__":
    test_db_to_amplitude_eager()
    test_db_to_amplitude_pipeline()
    test_db_to_amplitude_invalid_input()
    test_db_to_amplitude_transform()
    test_db_to_amplitude_param_check()
