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
"""Test create_dct."""

import numpy as np
import pytest

from mindspore.dataset import audio
from . import count_unequal_element


def test_create_dct_none():
    """
    Feature: Create DCT transformation
    Description: Test create_dct in eager mode with no normalization
    Expectation: The returned result is as expected
    """
    expect = np.array([[2.00000000, 1.84775901],
                       [2.00000000, 0.76536685],
                       [2.00000000, -0.76536703],
                       [2.00000000, -1.84775925]], dtype=np.float64)
    output = audio.create_dct(2, 4, audio.NormMode.NONE)
    count_unequal_element(expect, output, 0.0001, 0.0001)


def test_create_dct_ortho():
    """
    Feature: Create DCT transformation
    Description: Test create_dct in eager mode with orthogonal normalization
    Expectation: The returned result is as expected
    """
    output = audio.create_dct(1, 3, audio.NormMode.ORTHO)
    expect = np.array([[0.57735026],
                       [0.57735026],
                       [0.57735026]], dtype=np.float64)
    count_unequal_element(expect, output, 0.0001, 0.0001)


def test_create_dct_invalid_input():
    """
    Feature: Create DCT transformation
    Description: Test create_dct with invalid inputs
    Expectation: Error is raised as expected
    """

    def test_invalid_input(n_mfcc, n_mels, norm, error, error_msg):
        with pytest.raises(error) as error_info:
            audio.create_dct(n_mfcc, n_mels, norm)
        assert error_msg in str(error_info.value)

    test_invalid_input(100.5, 200, audio.NormMode.NONE, TypeError,
                       "n_mfcc with value 100.5 is not of type <class 'int'>, but got <class 'float'>.")
    test_invalid_input("100", 200, audio.NormMode.NONE, TypeError,
                       "n_mfcc with value 100 is not of type <class 'int'>, but got <class 'str'>.")
    test_invalid_input(100, "200", audio.NormMode.NONE, TypeError,
                       "n_mels with value 200 is not of type <class 'int'>, but got <class 'str'>.")
    test_invalid_input(0, 200, audio.NormMode.NONE, ValueError,
                       "n_mfcc must be greater than 0, but got 0.")
    test_invalid_input(100, 0, audio.NormMode.NONE, ValueError,
                       "n_mels must be greater than 0, but got 0.")
    test_invalid_input(-100, 200, audio.NormMode.NONE, ValueError,
                       "n_mfcc must be greater than 0, but got -100.")
    test_invalid_input(None, 100, audio.NormMode.NONE, TypeError,
                       "n_mfcc with value None is not of type <class 'int'>, but got <class 'NoneType'>.")
    test_invalid_input(100, None, audio.NormMode.NONE, TypeError,
                       "n_mels with value None is not of type <class 'int'>, but got <class 'NoneType'>.")
    test_invalid_input(100, 200, "None", TypeError,
                       "norm with value None is not of type <enum 'NormMode'>, but got <class 'str'>.")


def test_create_dct_transform():
    """
    Feature: create_dct function
    Description: Test create_dct with various parameter validation scenarios
    Expectation: Correct error types and messages are raised for invalid parameters
    """
    # test
    with pytest.raises(TypeError) as error_info:
        audio.create_dct(100, [10])
    assert "n_mels with value [10] is not of type <class 'int'>, but got <class 'list'>." in str(error_info.value)

    # test
    with pytest.raises(ValueError, match="n_mfcc must be greater than 0, but got 0."):
        audio.create_dct(0, 100)

    # test
    with pytest.raises(ValueError) as error_info:
        audio.create_dct(-1, 10)
    assert "n_mfcc must be greater than 0, but got -1." in str(error_info.value)

    # test
    with pytest.raises(TypeError,
                       match="n_mfcc with value 100.5 is not of type <class 'int'>, but got <class 'float'>."):
        audio.create_dct(100.5, 200, audio.NormMode.NONE)

    # test
    with pytest.raises(TypeError,
                       match="n_mels with value 200.5 is not of type <class 'int'>, but got <class 'float'>."):
        audio.create_dct(100, 200.5, audio.NormMode.NONE)

    # test
    with pytest.raises(TypeError,
                       match="norm with value None is not of type <enum 'NormMode'>, but got <class 'str'>."):
        audio.create_dct(100, 200, "None")


def test_create_dct_param_check():
    """
    Feature: create_dct function
    Description: Test create_dct with additional invalid parameter combinations
    Expectation: Correct error types and messages are raised as expected
    """
    # Test that n_mfcc with value 0 raises a ValueError.
    with pytest.raises(ValueError, match="n_mfcc must be greater than 0, but got 0."):
        audio.create_dct(0, 100)

    # Test that n_mels with a negative value raises a ValueError.
    with pytest.raises(ValueError, match="n_mels must be greater than 0, but got -1."):
        audio.create_dct(10, -1)

    # Test that n_mfcc with string type raises a TypeError.
    with pytest.raises(TypeError, match="n_mfcc with value 100 is not of type <class 'int'>, but got <class 'str'>."):
        audio.create_dct("100", 200, audio.NormMode.NONE)

    # Test that n_mels with float type raises a TypeError.
    with pytest.raises(TypeError,
                       match="n_mels with value 200.5 is not of type <class 'int'>, but got <class 'float'>."):
        audio.create_dct(100, 200.5, audio.NormMode.NONE)

    # Test that n_mfcc with None type raises a TypeError.
    with pytest.raises(TypeError,
                       match="n_mfcc with value None is not of type <class 'int'>, but got <class 'NoneType'>."):
        audio.create_dct(None, 100, audio.NormMode.NONE)

    # Test that n_mels with None type raises a TypeError.
    with pytest.raises(TypeError,
                       match="n_mels with value None is not of type <class 'int'>, but got <class 'NoneType'>."):
        audio.create_dct(100, None, audio.NormMode.NONE)

    # Test that norm with boolean type raises a TypeError.
    with pytest.raises(TypeError,
                       match="norm with value True is not of type <enum 'NormMode'>, but got <class 'bool'>."):
        audio.create_dct(100, 200, True)


if __name__ == "__main__":
    test_create_dct_none()
    test_create_dct_ortho()
    test_create_dct_invalid_input()
    test_create_dct_transform()
    test_create_dct_param_check()
