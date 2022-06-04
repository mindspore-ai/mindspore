# Copyright 2021-2022 Huawei Technologies Co., Ltd
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
import numpy as np
import pytest

from mindspore.dataset.audio import create_dct, NormMode
from mindspore import log as logger


def count_unequal_element(data_expected, data_me, rtol, atol):
    assert data_expected.shape == data_me.shape
    total_count = len(data_expected.flatten())
    error = np.abs(data_expected - data_me)
    greater = np.greater(error, atol + np.abs(data_expected) * rtol)
    loss_count = np.count_nonzero(greater)
    assert (loss_count / total_count) < rtol, \
        "\ndata_expected_std:{0}\ndata_me_error:{1}\nloss:{2}". \
        format(data_expected[greater], data_me[greater], error[greater])


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
    output = create_dct(2, 4, NormMode.NONE)
    count_unequal_element(expect, output, 0.0001, 0.0001)


def test_create_dct_ortho():
    """
    Feature: Create DCT transformation
    Description: Test create_dct in eager mode with orthogonal normalization
    Expectation: The returned result is as expected
    """
    output = create_dct(1, 3, NormMode.ORTHO)
    expect = np.array([[0.57735026],
                       [0.57735026],
                       [0.57735026]], dtype=np.float64)
    count_unequal_element(expect, output, 0.0001, 0.0001)


def test_createdct_invalid_input():
    """
    Feature: Create DCT transformation
    Description: Test create_dct with invalid inputs
    Expectation: Error is raised as expected
    """
    def test_invalid_input(test_name, n_mfcc, n_mels, norm, error, error_msg):
        logger.info("Test CreateDct with bad input: {0}".format(test_name))
        with pytest.raises(error) as error_info:
            create_dct(n_mfcc, n_mels, norm)
        assert error_msg in str(error_info.value)

    test_invalid_input("invalid n_mfcc parameter type as a float", 100.5, 200, NormMode.NONE, TypeError,
                       "n_mfcc with value 100.5 is not of type <class 'int'>, but got <class 'float'>.")
    test_invalid_input("invalid n_mfcc parameter type as a String", "100", 200, NormMode.NONE, TypeError,
                       "n_mfcc with value 100 is not of type <class 'int'>, but got <class 'str'>.")
    test_invalid_input("invalid n_mels parameter type as a String", 100, "200", NormMode.NONE, TypeError,
                       "n_mels with value 200 is not of type <class 'int'>, but got <class 'str'>.")
    test_invalid_input("invalid n_mels parameter type as a String", 0, 200, NormMode.NONE, ValueError,
                       "n_mfcc must be greater than 0, but got 0.")
    test_invalid_input("invalid n_mels parameter type as a String", 100, 0, NormMode.NONE, ValueError,
                       "n_mels must be greater than 0, but got 0.")
    test_invalid_input("invalid n_mels parameter type as a String", -100, 200, NormMode.NONE, ValueError,
                       "n_mfcc must be greater than 0, but got -100.")
    test_invalid_input("invalid n_mfcc parameter value", None, 100, NormMode.NONE, TypeError,
                       "n_mfcc with value None is not of type <class 'int'>, but got <class 'NoneType'>.")
    test_invalid_input("invalid n_mels parameter value", 100, None, NormMode.NONE, TypeError,
                       "n_mels with value None is not of type <class 'int'>, but got <class 'NoneType'>.")
    test_invalid_input("invalid n_mels parameter value", 100, 200, "None", TypeError,
                       "norm with value None is not of type <enum 'NormMode'>, but got <class 'str'>.")


if __name__ == "__main__":
    test_create_dct_none()
    test_create_dct_ortho()
    test_createdct_invalid_input()
