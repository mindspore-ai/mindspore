# Copyright 2021 Huawei Technologies Co., Ltd
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

import mindspore.dataset as ds
import mindspore.dataset.audio.transforms as audio
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


def test_func_biquad_eager():
    """ mindspore eager mode normal testcase:biquad op"""
    # Original waveform
    waveform = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64)
    # Expect waveform
    expect_waveform = np.array([[0.0100, 0.0388, 0.1923],
                                [0.0400, 0.1252, 0.6530]], dtype=np.float64)
    biquad_op = audio.Biquad(0.01, 0.02, 0.13, 1, 0.12, 0.3)
    # Filtered waveform by biquad
    output = biquad_op(waveform)
    count_unequal_element(expect_waveform, output, 0.0001, 0.0001)


def test_func_biquad_pipeline():
    """ mindspore pipeline mode normal testcase:biquad op"""
    # Original waveform
    waveform = np.array([[3.2, 2.1, 1.3], [6.2, 5.3, 6]], dtype=np.float64)
    # Expect waveform
    expect_waveform = np.array([[1.0000, 1.0000, 0.5844],
                                [1.0000, 1.0000, 1.0000]], dtype=np.float64)
    dataset = ds.NumpySlicesDataset(waveform, ["audio"], shuffle=False)
    biquad_op = audio.Biquad(1, 0.02, 0.13, 1, 0.12, 0.3)
    # Filtered waveform by biquad
    dataset = dataset.map(input_columns=["audio"], operations=biquad_op, num_parallel_workers=8)
    i = 0
    for item in dataset.create_dict_iterator(output_numpy=True):
        count_unequal_element(expect_waveform[i, :],
                              item['audio'], 0.0001, 0.0001)
        i += 1


def test_biquad_invalid_input():
    def test_invalid_input(test_name, b0, b1, b2, a0, a1, a2, error, error_msg):
        logger.info("Test Biquad with bad input: {0}".format(test_name))
        with pytest.raises(error) as error_info:
            audio.Biquad(b0, b1, b2, a0, a1, a2)
        assert error_msg in str(error_info.value)

    test_invalid_input("invalid b0 parameter type as a String", "0.01", 0.02, 0.13, 1, 0.12, 0.3, TypeError,
                       "Argument b0 with value 0.01 is not of type [<class 'float'>, <class 'int'>],"
                       " but got <class 'str'>.")
    test_invalid_input("invalid b0 parameter value", 441324343243242342345300, 0.02, 0.13, 1, 0.12, 0.3, ValueError,
                       "Input b0 is not within the required interval of [-16777216, 16777216].")
    test_invalid_input("invalid b1 parameter type as a String", 0.01, "0.02", 0.13, 0, 0.12, 0.3, TypeError,
                       "Argument b1 with value 0.02 is not of type [<class 'float'>, <class 'int'>],"
                       " but got <class 'str'>.")
    test_invalid_input("invalid b1 parameter value", 0.01, 441324343243242342345300, 0.13, 1, 0.12, 0.3, ValueError,
                       "Input b1 is not within the required interval of [-16777216, 16777216].")
    test_invalid_input("invalid b2 parameter type as a String", 0.01, 0.02, "0.13", 0, 0.12, 0.3, TypeError,
                       "Argument b2 with value 0.13 is not of type [<class 'float'>, <class 'int'>],"
                       " but got <class 'str'>.")
    test_invalid_input("invalid b2 parameter value", 0.01, 0.02, 441324343243242342345300, 1, 0.12, 0.3, ValueError,
                       "Input b2 is not within the required interval of [-16777216, 16777216].")
    test_invalid_input("invalid a0 parameter type as a String", 0.01, 0.02, 0.13, '1', 0.12, 0.3, TypeError,
                       "Argument a0 with value 1 is not of type [<class 'float'>, <class 'int'>],"
                       " but got <class 'str'>.")
    test_invalid_input("invalid a0 parameter value", 0.01, 0.02, 0.13, 0, 0.12, 0.3, ValueError,
                       "Input a0 is not within the required interval of [-16777216, 0) and (0, 16777216].")
    test_invalid_input("invalid a0 parameter value", 0.01, 0.02, 0.13, 441324343243242342345300, 0.12, 0.3, ValueError,
                       "Input a0 is not within the required interval of [-16777216, 0) and (0, 16777216].")
    test_invalid_input("invalid a1 parameter type as a String", 0.01, 0.02, 0.13, 1, '0.12', 0.3, TypeError,
                       "Argument a1 with value 0.12 is not of type [<class 'float'>, <class 'int'>],"
                       " but got <class 'str'>.")
    test_invalid_input("invalid a1 parameter value", 0.01, 0.02, 0.13, 1, 441324343243242342345300, 0.3, ValueError,
                       "Input a1 is not within the required interval of [-16777216, 16777216].")
    test_invalid_input("invalid a2 parameter type as a String", 0.01, 0.02, 0.13, 1, 0.12, '0.3', TypeError,
                       "Argument a2 with value 0.3 is not of type [<class 'float'>, <class 'int'>],"
                       " but got <class 'str'>.")
    test_invalid_input("invalid a1 parameter value", 0.01, 0.02, 0.13, 1, 0.12, 441324343243242342345300, ValueError,
                       "Input a2 is not within the required interval of [-16777216, 16777216].")


if __name__ == '__main__':
    test_func_biquad_eager()
    test_func_biquad_pipeline()
    test_biquad_invalid_input()
