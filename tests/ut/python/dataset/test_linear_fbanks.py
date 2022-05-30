# Copyright 2022 Huawei Technologies Co., Ltd
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

import mindspore.dataset.audio.utils as audio
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


def test_linear_fbanks_normal():
    """
    Feature: linear_fbanks.
    Description: Test normal operation.
    Expectation: The output data is the same as the result of torchaudio.functional.linear_fbanks.
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

    def test_invalid_input(test_name, n_freqs, f_min, f_max, n_filter, sample_rate, error, error_msg):
        logger.info("Test linear_fbanks with bad input: {0}".format(test_name))
        with pytest.raises(error) as error_info:
            audio.linear_fbanks(n_freqs, f_min, f_max, n_filter, sample_rate)
        print(error_info)
        assert error_msg in str(error_info.value)

    test_invalid_input("invalid n_freqs parameter Value", 99999999999, 0, 50, 5, 100, ValueError, "n_freqs")
    test_invalid_input("invalid n_freqs parameter type", 10.5, 0, 50, 5, 100, TypeError, "n_freqs")
    test_invalid_input("invalid f_min parameter type", 10, None, 50, 5, 100, TypeError, "f_min")
    test_invalid_input("invalid f_max parameter type", 10, 0, None, 5, 100, TypeError, "f_max")
    test_invalid_input("invalid n_filter parameter type", 10, 0, 50, 10.1, 100, TypeError, "n_filter")
    test_invalid_input("invalid n_filter parameter Value", 20, 0, 50, 999999999999, 100, ValueError, "n_filter")
    test_invalid_input("invalid sample_rate parameter type", 10, 0, 50, 5, 100.1, TypeError, "sample_rate")
    test_invalid_input("invalid sample_rate parameter Value", 20, 0, 50, 5, 999999999999, ValueError, "sample_rate")


if __name__ == "__main__":
    test_linear_fbanks_normal()
    test_linear_fbanks_invalid_input()
