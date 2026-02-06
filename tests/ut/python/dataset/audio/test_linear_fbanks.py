# Copyright 2022-2025 Huawei Technologies Co., Ltd
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
"""Test LinearFbanks."""

import numpy as np
import pytest

from mindspore.dataset import audio
from . import count_unequal_element


def test_linear_fbanks_normal():
    """
    Feature: linear_fbanks.
    Description: Test normal operation.
    Expectation: The output data is as expected.
    """
    expect = np.array([[0.0000, 0.0000, 0.0000, 0.0000],
                       [0.5357, 0.0000, 0.0000, 0.0000],
                       [0.7202, 0.2798, 0.0000, 0.0000],
                       [0.0000, 0.9762, 0.0238, 0.0000],
                       [0.0000, 0.2321, 0.7679, 0.0000],
                       [0.0000, 0.0000, 0.4881, 0.5119],
                       [0.0000, 0.0000, 0.0000, 0.7440],
                       [0.0000, 0.0000, 0.0000, 0.0000]], dtype=np.float64)
    output = audio.linear_fbanks(8, 2, 50, 4, 100)
    count_unequal_element(expect, output, 0.0001, 0.0001)


def test_linear_fbanks_invalid_input():
    """
    Feature: linear_fbanks.
    Description: Test operation with invalid input.
    Expectation: Throw exception as expected.
    """

    def test_invalid_input(n_freqs, f_min, f_max, n_filter, sample_rate, error, error_msg):
        with pytest.raises(error) as error_info:
            audio.linear_fbanks(n_freqs, f_min, f_max, n_filter, sample_rate)
        print(error_info)
        assert error_msg in str(error_info.value)

    test_invalid_input(99999999999, 0, 50, 5, 100, ValueError, "n_freqs")
    test_invalid_input(10.5, 0, 50, 5, 100, TypeError, "n_freqs")
    test_invalid_input(10, None, 50, 5, 100, TypeError, "f_min")
    test_invalid_input(10, 0, None, 5, 100, TypeError, "f_max")
    test_invalid_input(10, 0, 50, 10.1, 100, TypeError, "n_filter")
    test_invalid_input(20, 0, 50, 999999999999, 100, ValueError, "n_filter")
    test_invalid_input(10, 0, 50, 5, 100.1, TypeError, "sample_rate")
    test_invalid_input(20, 0, 50, 5, 999999999999, ValueError, "sample_rate")


def test_linear_fbanks_transform():
    """
    Feature: LinearFbanks
    Description: Test LinearFbanks with various valid input parameters and data types
    Expectation: The operation completes successfully
    """

    output = audio.linear_fbanks(n_freqs=8, f_min=2, f_max=50, n_filter=4, sample_rate=100)
    assert output.shape == (8, 4)
    assert output.dtype == np.float32

    # linear_fbanks:test f_min is 0
    audio.linear_fbanks(n_freqs=8, f_min=0, f_max=50, n_filter=4, sample_rate=100)

    # linear_fbanks:test example
    output = audio.linear_fbanks(n_freqs=4096, f_min=0, f_max=8000, n_filter=40, sample_rate=16000)
    assert output.shape == (4096, 40)
    assert output.dtype == np.float32

    # linear_fbanks:test n_freqs is 0
    audio.linear_fbanks(n_freqs=0, f_min=2, f_max=50, n_filter=4, sample_rate=100)

    # linear_fbanks:test f_max is float
    output = audio.linear_fbanks(n_freqs=8, f_min=2, f_max=50.0, n_filter=4, sample_rate=100)
    assert output.shape == (8, 4)
    assert output.dtype == np.float32

    # linear_fbanks:test f_min is float
    output = audio.linear_fbanks(n_freqs=8, f_min=0.1, f_max=50, n_filter=4, sample_rate=100)
    assert output.shape == (8, 4)
    assert output.dtype == np.float32


def test_linear_fbanks_param_check():
    """
    Feature: LinearFbanks
    Description: Test LinearFbanks with invalid input data types and parameters
    Expectation: Correct error types and messages are raised as expected
    """

    with pytest.raises(ValueError, match="Input n_freqs is not within the required interval of \\[0, 2147483647\\]"):
        audio.linear_fbanks(n_freqs=2147483648, f_min=2, f_max=50, n_filter=4, sample_rate=100)

    # linear_fbanks:test n_freqs is float
    with pytest.raises(TypeError, match="Argument n_freqs with value 8.0 is not of type \\[<class 'int'>\\], "
                                        "but got <class 'float'>"):
        audio.linear_fbanks(n_freqs=8.0, f_min=2, f_max=50, n_filter=4, sample_rate=100)

    # linear_fbanks:test n_freqs is str
    with pytest.raises(TypeError, match="Argument n_freqs with value test is not of type \\[<class 'int'>\\], "
                                        "but got <class 'str'>."):
        audio.linear_fbanks(n_freqs="test", f_min=2, f_max=50, n_filter=4, sample_rate=100)

    # linear_fbanks:test n_freqs is None
    with pytest.raises(TypeError, match="Argument n_freqs with value None is not of type \\[<class 'int'>\\], "
                                        "but got <class 'NoneType'>"):
        audio.linear_fbanks(n_freqs=None, f_min=2, f_max=50, n_filter=4, sample_rate=100)

    # linear_fbanks:test n_freqs is bool
    with pytest.raises(TypeError, match="Argument n_freqs with value True is not of type \\(<class 'int'>,\\), "
                                        "but got <class 'bool'>."):
        audio.linear_fbanks(n_freqs=True, f_min=2, f_max=50, n_filter=4, sample_rate=100)

    # linear_fbanks:test f_min is 16777217
    with pytest.raises(ValueError, match="Input f_min is not within the required interval of \\[0, 16777216\\]"):
        audio.linear_fbanks(n_freqs=8, f_min=16777217, f_max=50, n_filter=4, sample_rate=100)

    # linear_fbanks:test f_min is negative
    with pytest.raises(ValueError, match="Input f_min is not within the required interval of \\[0, 16777216\\]"):
        audio.linear_fbanks(n_freqs=8, f_min=-1, f_max=50, n_filter=4, sample_rate=100)

    # linear_fbanks:test f_min is str
    with pytest.raises(TypeError, match="Argument f_min with value test is not of type "
                                        "\\[<class 'int'>, <class 'float'>\\], but got <class 'str'>"):
        audio.linear_fbanks(n_freqs=8, f_min="test", f_max=50, n_filter=4, sample_rate=100)

    # linear_fbanks:test f_min is None
    with pytest.raises(TypeError, match="Argument f_min with value None is not of type "
                                        "\\[<class 'int'>, <class 'float'>\\], but got <class 'NoneType'>."):
        audio.linear_fbanks(n_freqs=8, f_min=None, f_max=50, n_filter=4, sample_rate=100)

    # linear_fbanks:test f_min is bool
    with pytest.raises(TypeError, match="Argument f_min with value True is not of type "
                                        "\\(<class 'int'>, <class 'float'>\\), but got <class 'bool'>."):
        audio.linear_fbanks(n_freqs=8, f_min=True, f_max=50, n_filter=4, sample_rate=100)

    # linear_fbanks:test f_max is 0
    with pytest.raises(ValueError, match="Input f_max is not within the required interval of \\(0, 16777216\\]"):
        audio.linear_fbanks(n_freqs=8, f_min=2, f_max=0, n_filter=4, sample_rate=100)

    # linear_fbanks:test f_max is 16777217
    with pytest.raises(ValueError, match="Input f_max is not within the required interval of \\(0, 16777216\\]."):
        audio.linear_fbanks(n_freqs=8, f_min=2, f_max=16777217, n_filter=4, sample_rate=100)

    # linear_fbanks:test f_max is str
    with pytest.raises(TypeError, match="Argument f_max with value test is not of type "
                                        "\\[<class 'int'>, <class 'float'>\\], but got <class 'str'>."):
        audio.linear_fbanks(n_freqs=8, f_min=2, f_max="test", n_filter=4, sample_rate=100)

    # linear_fbanks:test f_max is None
    with pytest.raises(TypeError, match="Argument f_max with value None is not of type "
                                        "\\[<class 'int'>, <class 'float'>\\], but got <class 'NoneType'>."):
        audio.linear_fbanks(n_freqs=8, f_min=2, f_max=None, n_filter=4, sample_rate=100)

    # linear_fbanks:test f_max is bool
    with pytest.raises(TypeError, match="Argument f_max with value True is not of type "
                                        "\\(<class 'int'>, <class 'float'>\\), but got <class 'bool'>."):
        audio.linear_fbanks(n_freqs=8, f_min=2, f_max=True, n_filter=4, sample_rate=100)

    # linear_fbanks:test f_min > f_max
    with pytest.raises(ValueError, match="Input f_min should be no more than f_max, but got f_min: 50 and f_max: 2"):
        audio.linear_fbanks(n_freqs=8, f_min=50, f_max=2, n_filter=4, sample_rate=100)

    # linear_fbanks:test n_filter is 0
    with pytest.raises(ValueError, match="Input n_filter is not within the required interval of \\[1, 2147483647\\]"):
        audio.linear_fbanks(n_freqs=8, f_min=2, f_max=50, n_filter=0, sample_rate=100)

    # linear_fbanks:test n_filter is 2147483648
    with pytest.raises(ValueError, match="Input n_filter is not within the required interval of \\[1, 2147483647\\]"):
        audio.linear_fbanks(n_freqs=8, f_min=2, f_max=50, n_filter=2147483648, sample_rate=100)

    # linear_fbanks:test n_filter is float
    with pytest.raises(TypeError, match="Argument n_filter with value 4.0 is not of type \\[<class 'int'>\\], "
                                        "but got <class 'float'>."):
        audio.linear_fbanks(n_freqs=8, f_min=2, f_max=50, n_filter=4.0, sample_rate=100)

    # linear_fbanks:test n_filter is str
    with pytest.raises(TypeError, match="Argument n_filter with value test is not of type \\[<class 'int'>\\], "
                                        "but got <class 'str'>."):
        audio.linear_fbanks(n_freqs=8, f_min=2, f_max=50, n_filter="test", sample_rate=100)

    # linear_fbanks:test n_filter is None
    with pytest.raises(TypeError, match="Argument n_filter with value None is not of type \\[<class 'int'>\\], "
                                        "but got <class 'NoneType'>"):
        audio.linear_fbanks(n_freqs=8, f_min=2, f_max=50, n_filter=None, sample_rate=100)

    # linear_fbanks:test n_filter is bool
    with pytest.raises(TypeError, match="Argument n_filter with value True is not of type \\(<class 'int'>,\\), "
                                        "but got <class 'bool'>."):
        audio.linear_fbanks(n_freqs=8, f_min=2, f_max=50, n_filter=True, sample_rate=100)

    # linear_fbanks:test sample_rate is 0
    with pytest.raises(ValueError, match="Input sample_rate is not within the required interval of "
                                         "\\[1, 2147483647\\]."):
        audio.linear_fbanks(n_freqs=8, f_min=2, f_max=50, n_filter=4, sample_rate=0)

    # linear_fbanks:test sample_rate is 2147483648
    with pytest.raises(ValueError, match="Input sample_rate is not within the required interval of "
                                         "\\[1, 2147483647\\]."):
        audio.linear_fbanks(n_freqs=8, f_min=2, f_max=50, n_filter=4, sample_rate=2147483648)

    # linear_fbanks:test sample_rate is float
    with pytest.raises(TypeError, match="Argument sample_rate with value 100.0 is not of type \\[<class 'int'>\\], "
                                        "but got <class 'float'>."):
        audio.linear_fbanks(n_freqs=8, f_min=2, f_max=50, n_filter=4, sample_rate=100.0)

    # linear_fbanks:test sample_rate is str
    with pytest.raises(TypeError, match="Argument sample_rate with value test is not of type \\[<class 'int'>\\], "
                                        "but got <class 'str'>."):
        audio.linear_fbanks(n_freqs=8, f_min=2, f_max=50, n_filter=4, sample_rate="test")

    # linear_fbanks:test sample_rate is None
    with pytest.raises(TypeError, match="Argument sample_rate with value None is not of type \\[<class 'int'>\\], "
                                        "but got <class 'NoneType'>."):
        audio.linear_fbanks(n_freqs=8, f_min=2, f_max=50, n_filter=4, sample_rate=None)

    # linear_fbanks:test sample_rate is bool
    with pytest.raises(TypeError, match="Argument sample_rate with value True is not of type \\(<class 'int'>,\\), "
                                        "but got <class 'bool'>."):
        audio.linear_fbanks(n_freqs=8, f_min=2, f_max=50, n_filter=4, sample_rate=True)


if __name__ == "__main__":
    test_linear_fbanks_normal()
    test_linear_fbanks_invalid_input()
    test_linear_fbanks_transform()
    test_linear_fbanks_param_check()
