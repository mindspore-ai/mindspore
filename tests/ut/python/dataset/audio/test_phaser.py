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
"""Test Phaser."""

import numpy as np
import pytest

import mindspore.dataset as ds
from mindspore.dataset import audio
from . import count_unequal_element


def test_phaser_eager():
    """
    Feature: Phaser
    Description: Test Phaser in eager mode
    Expectation: The results are as expected
    """
    # Original waveform
    waveform = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    # Expect waveform
    expect_waveform = np.array([[0.296, 0.71040004, 1.],
                                [1., 1., 1.]], dtype=np.float32)
    sample_rate = 44100
    # Filtered waveform by phaser
    output = audio.Phaser(sample_rate=sample_rate)(waveform)
    count_unequal_element(expect_waveform, output, 0.0001, 0.0001)


def test_phaser_pipeline():
    """
    Feature: Phaser
    Description: Test Phaser in pipeline mode
    Expectation: The results are as expected
    """
    # Original waveform
    waveform = np.array([[0.1, 1.2, 5.3], [0.4, 5.5, 1.6]], dtype=np.float32)
    # Expect waveform
    expect_waveform = np.array([[0.0296, 0.36704, 1.],
                                [0.11840001, 1., 1.]], dtype=np.float32)
    sample_rate = 44100
    dataset = ds.NumpySlicesDataset(waveform, ["waveform"], shuffle=False)
    phaser = audio.Phaser(sample_rate)
    # Filtered waveform by phaser
    dataset = dataset.map(
        input_columns=["waveform"], operations=phaser)
    i = 0
    for item in dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
        count_unequal_element(expect_waveform[i, :],
                              item['waveform'], 0.0001, 0.0001)
        i += 1


def test_phaser_invalid_input():
    """
    Feature: Phaser
    Description: Test invalid parameter of Phaser
    Expectation: Catch exceptions correctly
    """

    def test_invalid_input(sample_rate, gain_in, gain_out, delay_ms, decay, mod_speed, sinusoidal, error,
                           error_msg):
        with pytest.raises(error) as error_info:
            audio.Phaser(sample_rate, gain_in, gain_out, delay_ms, decay, mod_speed, sinusoidal)
        assert error_msg in str(error_info.value)

    test_invalid_input(44100.5, 0.4, 0.74, 3.0, 0.4, 0.5, True,
                       TypeError, "Argument sample_rate with value 44100.5 is not of type [<class 'int'>],"
                                  " but got <class 'float'>.")
    test_invalid_input(44100, "1", 0.74, 3.0, 0.4, 0.5, True,
                       TypeError, "Argument gain_in with value 1 is not of type [<class 'float'>, <class 'int'>],"
                       + " but got <class 'str'>.")
    test_invalid_input(44100, 0.4, "10", 3.0, 0.4, 0.5, True, TypeError,
                       "Argument gain_out with value 10 is not of type [<class 'float'>, <class 'int'>],"
                       + " but got <class 'str'>.")
    test_invalid_input(44100, 0.4, 0.74, "2", 0.4, 0.5, True, TypeError,
                       "Argument delay_ms with value 2 is not of type [<class 'float'>, <class 'int'>],"
                       + " but got <class 'str'>.")
    test_invalid_input(44100, 0.4, 0.74, 3.0, "0", 0.5, True, TypeError,
                       "Argument decay with value 0 is not of type [<class 'float'>, <class 'int'>],"
                       + " but got <class 'str'>.")
    test_invalid_input(44100, 0.4, 0.74, 3.0, 0.4, "3", True, TypeError,
                       "Argument mod_speed with value 3 is not of type [<class 'float'>, <class 'int'>],"
                       + " but got <class 'str'>.")
    test_invalid_input(44100, 0.4, 0.74, 3.0, 0.4, 0.5, "True", TypeError,
                       "Argument sinusoidal with value True is not of type [<class 'bool'>],"
                       + " but got <class 'str'>.")
    test_invalid_input(441324343243242342345300, 0.5, 0.74, 3.0, 0.4, 0.5, True,
                       ValueError, "Input sample_rate is not within the required interval of "
                                   "[-2147483648, 2147483647].")
    test_invalid_input(44100, 2.0, 0.74, 3.0, 0.4, 0.5, True, ValueError,
                       "Input gain_in is not within the required interval of [0, 1].")
    test_invalid_input(44100, 0.4, -2.0, 3.0, 0.4, 0.5, True, ValueError,
                       "Input gain_out is not within the required interval of [0, 1000000000.0].")
    test_invalid_input(44100, 0.4, 0.74, 6.0, 0.4, 0.5, True, ValueError,
                       "Input delay_ms is not within the required interval of [0, 5.0].")
    test_invalid_input(44100, 0.4, 0.74, 3.0, 1.2, 0.5, True, ValueError,
                       "Input decay is not within the required interval of [0, 0.99].")
    test_invalid_input(44100, 0.4, 0.74, 3.0, 0.4, 0.003, True, ValueError,
                       "Input mod_speed is not within the required interval of [0.1, 2].")


def test_phaser_transform():
    """
    Feature: Phaser
    Description: Test Phaser with various valid input parameters and data types
    Expectation: The operation completes successfully
    """

    waveform = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float16)
    output = audio.Phaser(35000)(waveform)
    assert np.shape(output) == (2, 3)

    # mindspore eager mode acc testcase:Phaser
    waveform = np.array([[0.4, 1.3, 0.0786], [-2.63, 0.6, 1.]], dtype=np.float64)
    output = audio.Phaser(88000)(waveform)
    assert np.shape(output) == (2, 3)


def test_phaser_param_check():
    """
    Feature: Phaser
    Description: Test Phaser with invalid input data types and parameters
    Expectation: Correct error types and messages are raised as expected
    """

    error_msg = "Argument sample_rate with value 4100.5 is not of type [<class 'int'>], but got <class 'float'>"
    with pytest.raises(TypeError) as error_info:
        audio.Phaser(4100.5, 0.4, 0.74, 3.0, 0.4, 0.5, True)
    assert error_msg in str(error_info)

    # Test with invalid gain_in parameter type (TypeError expected)
    error_msg = "Argument gain_in with value 1 is not of type [<class \'float\'>, <class \'int\'>], but got <clas" \
                "s \'str\'>"
    with pytest.raises(TypeError) as error_info:
        audio.Phaser(44100, "1", 0.74, 3.0, 0.4, 0.5, True)
    assert error_msg in str(error_info)

    # Test with invalid gain_out parameter type (TypeError expected)
    error_msg = "Argument gain_out with value 10 is not of type [<class \'float\'>, <class \'int\'>], but got <class" \
                " \'str\'>."
    with pytest.raises(TypeError) as error_info:
        audio.Phaser(44100, 0.4, "10", 3.0, 0.4, 0.5, True)
    assert error_msg in str(error_info)

    # Test with invalid delay_ms parameter type (TypeError expected)
    error_msg = "Argument delay_ms with value 2 is not of type [<class \'float\'>, <class \'int\'>], but got <class" \
                " \'str\'>."
    with pytest.raises(TypeError) as error_info:
        audio.Phaser(44100, 0.4, 0.74, "2", 0.4, 0.5, True)
    assert error_msg in str(error_info)

    # Test with invalid decay parameter type (TypeError expected)
    error_msg = "Argument decay with value 0 is not of type [<class \'float\'>, <class \'int\'>], but got <class \'" \
                "str\'>."
    with pytest.raises(TypeError) as error_info:
        audio.Phaser(44100, 0.4, 0.74, 3.0, "0", 0.5, True)
    assert error_msg in str(error_info)

    # Test with invalid mod_speed parameter type (TypeError expected)
    error_msg = "Argument mod_speed with value 3 is not of type [<class \'float\'>, <class \'int\'>], but got <cl" \
                "ass \'str\'>."
    with pytest.raises(TypeError) as error_info:
        audio.Phaser(44100, 0.4, 0.74, 3.0, 0.4, "3", True)
    assert error_msg in str(error_info)

    # Test with invalid sinusoidal parameter type (TypeError expected)
    error_msg = "Argument sinusoidal with value True is not of type [<class \'bool\'>], but got <class \'str\'>."
    with pytest.raises(TypeError) as error_info:
        audio.Phaser(44100, 0.4, 0.74, 3.0, 0.4, 0.5, "True")
    assert error_msg in str(error_info)

    # Test with invalid sample_rate parameter value (ValueError expected)
    with pytest.raises(ValueError) as error_info:
        error_msg = "Input sample_rate is not within the required interval of [-2147483648, 2147483647]."
        audio.Phaser(441324343243242342345300, 0.5, 0.74, 3.0, 0.4, 0.5, True)
    assert error_msg in str(error_info)

    # Test with invalid gain_in parameter value (ValueError expected)
    with pytest.raises(ValueError) as error_info:
        error_msg = "Input gain_in is not within the required interval of [0, 1]."
        audio.Phaser(44100, 2.0, 0.74, 3.0, 0.4, 0.5, True)
    assert error_msg in str(error_info)

    # Test with invalid gain_out parameter value (ValueError expected)
    with pytest.raises(ValueError) as error_info:
        error_msg = "Input gain_out is not within the required interval of [0, 1000000000.0]."
        audio.Phaser(44100, 0.4, -2.0, 3.0, 0.4, 0.5, True)
    assert error_msg in str(error_info)

    # Test with invalid delay_ms parameter value (ValueError expected)
    with pytest.raises(ValueError) as error_info:
        error_msg = "Input delay_ms is not within the required interval of [0, 5.0]."
        audio.Phaser(44100, 0.4, 0.74, 6.0, 0.4, 0.5, True)
    assert error_msg in str(error_info)

    # Test with invalid decay parameter value (ValueError expected)
    with pytest.raises(ValueError) as error_info:
        error_msg = "Input decay is not within the required interval of [0, 0.99]."
        audio.Phaser(44100, 0.4, 0.74, 3.0, 1.2, 0.5, True)
    assert error_msg in str(error_info)

    # Test with invalid mod_speed parameter value (ValueError expected)
    with pytest.raises(ValueError) as error_info:
        error_msg = "Input mod_speed is not within the required interval of [0.1, 2]."
        audio.Phaser(44100, 0.4, 0.74, 3.0, 0.4, 0.003, True)
    assert error_msg in str(error_info)

    # Test Phaser normal
    waveform = np.array(1, dtype=np.float32)
    with pytest.raises(RuntimeError, match=".*Phaser: the shape of input tensor does not match "
                                           "the requirement of operator. Expecting tensor in "
                                           "shape of <..., time>. But got tensor with dimension 0."):
        audio.Phaser(5)(waveform)

    # Test Overdrive normal
    waveform = "1"
    with pytest.raises(TypeError, match="Input should be NumPy audio, got <class 'str'>."):
        audio.Phaser(44100)(waveform)


if __name__ == "__main__":
    test_phaser_eager()
    test_phaser_pipeline()
    test_phaser_invalid_input()
    test_phaser_transform()
    test_phaser_param_check()
