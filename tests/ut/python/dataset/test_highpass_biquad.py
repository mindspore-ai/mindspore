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
"""
Testing HighpassBiquad op in DE
"""

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


def test_highpass_biquad_eager():
    """ mindspore eager mode normal testcase:highpass_biquad op"""
    # Original waveform
    waveform = np.array([[0.8236, 0.2049, 0.3335], [0.5933, 0.9911, 0.2482],
                         [0.3007, 0.9054, 0.7598], [0.5394, 0.2842, 0.5634], [0.6363, 0.2226, 0.2288]])
    # Expect waveform
    expect_waveform = np.array([[0.2745, -0.4808, 0.1576], [0.1978, -0.0652, -0.4462],
                                [0.1002, 0.1013, -0.2835], [0.1798, -0.2649, 0.1182], [0.2121, -0.3500, 0.0693]])
    highpass_biquad_op = audio.HighpassBiquad(4000, 1000.0, 1)
    # Filtered waveform by highpass_biquad
    output = highpass_biquad_op(waveform)
    count_unequal_element(expect_waveform, output, 0.0001, 0.0001)


def test_highpass_biquad_pipeline():
    """ mindspore pipeline mode normal testcase:highpass_biquad op"""
    # Original waveform
    waveform = np.array([[0.4063, 0.7729, 0.2325], [0.2687, 0.1426, 0.8987],
                         [0.6914, 0.6681, 0.1783], [0.2704, 0.2680, 0.7975], [0.5880, 0.1776, 0.6323]])
    # Expect waveform
    expect_waveform = np.array([[0.1354, -0.0133, -0.3474], [0.0896, -0.1316, 0.2642],
                                [0.2305, -0.2382, -0.2323], [0.0901, -0.0909, 0.1473], [0.1960, -0.3328, 0.2230]])
    dataset = ds.NumpySlicesDataset(waveform, ["col1"], shuffle=False)
    highpass_biquad_op = audio.HighpassBiquad(4000, 1000.0, 1)
    # Filtered waveform by highpass_biquad
    dataset = dataset.map(
        input_columns=["col1"], operations=highpass_biquad_op, num_parallel_workers=4)
    i = 0
    for item in dataset.create_dict_iterator(output_numpy=True):
        count_unequal_element(expect_waveform[i, :],
                              item["col1"], 0.0001, 0.0001)
        i += 1


def test_highpass_biquad_invalid_input():
    """
    Test invalid input of HighpassBiquad
    """
    def test_invalid_input(test_name, sample_rate, cutoff_freq, Q, error, error_msg):
        logger.info("Test HighpassBiquad with bad input: {0}".format(test_name))
        with pytest.raises(error) as error_info:
            audio.HighpassBiquad(sample_rate, cutoff_freq, Q)
        assert error_msg in str(error_info.value)

    test_invalid_input("invalid sample_rate parameter type as a float", 44100.5, 1000, 0.707, TypeError,
                       "Argument sample_rate with value 44100.5 is not of type [<class 'int'>],"
                       " but got <class 'float'>.")
    test_invalid_input("invalid sample_rate parameter type as a String", "44100", 1000, 0.707, TypeError,
                       "Argument sample_rate with value 44100 is not of type [<class 'int'>],"
                       " but got <class 'str'>.")
    test_invalid_input("invalid cutoff_freq parameter type as a String", 44100, "1000", 0.707, TypeError,
                       "Argument cutoff_freq with value 1000 is not of type [<class 'float'>, <class 'int'>],"
                       " but got <class 'str'>.")
    test_invalid_input("invalid Q parameter type as a String", 44100, 1000, "0.707", TypeError,
                       "Argument Q with value 0.707 is not of type [<class 'float'>, <class 'int'>],"
                       " but got <class 'str'>.")

    test_invalid_input("invalid sample_rate parameter value", 441324343243242342345300, 1000, 0.707, ValueError,
                       "Input sample_rate is not within the required interval of [-2147483648, 0) and (0, 2147483647].")
    test_invalid_input("invalid cutoff_freq parameter value", 44100, 32434324324234321, 0.707, ValueError,
                       "Input cutoff_freq is not within the required interval of [-16777216, 16777216].")

    test_invalid_input("invalid sample_rate parameter value", None, 1000, 0.707, TypeError,
                       "Argument sample_rate with value None is not of type [<class 'int'>], "
                       "but got <class 'NoneType'>.")
    test_invalid_input("invalid cutoff_rate parameter value", 44100, None, 0.707, TypeError,
                       "Argument cutoff_freq with value None is not of type [<class 'float'>, <class 'int'>],"
                       " but got <class 'NoneType'>.")

    test_invalid_input("invalid sample_rate parameter value", 0, 1000, 0.707, ValueError,
                       "Input sample_rate is not within the required interval of [-2147483648, 0) and (0, 2147483647].")
    test_invalid_input("invalid Q parameter value", 44100, 1000, 0, ValueError,
                       "Input Q is not within the required interval of (0, 1].")


if __name__ == "__main__":
    test_highpass_biquad_eager()
    test_highpass_biquad_pipeline()
    test_highpass_biquad_invalid_input()
