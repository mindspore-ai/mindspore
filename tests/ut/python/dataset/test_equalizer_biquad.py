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
Testing EqualizerBiquad op in DE
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
    assert (loss_count / total_count) < rtol, "\ndata_expected_std:{0}\ndata_me_error:{1}\nloss:{2}".format(
        data_expected[greater], data_me[greater], error[greater])


def test_equalizer_biquad_eager():
    """ mindspore eager mode normal testcase:highpass_biquad op"""
    # Original waveform
    waveform = np.array([[0.8236, 0.2049, 0.3335], [0.5933, 0.9911, 0.2482],
                         [0.3007, 0.9054, 0.7598], [0.5394, 0.2842, 0.5634], [0.6363, 0.2226, 0.2288]])
    # Expect waveform
    expect_waveform = np.array([[1.0000, 0.2532, 0.1273], [0.7333, 1.0000, 0.1015],
                                [0.3717, 1.0000, 0.8351], [0.6667, 0.3513, 0.5098], [0.7864, 0.2751, 0.0627]])
    equalizer_biquad_op = audio.EqualizerBiquad(4000, 1000.0, 5.5, 1)
    # Filtered waveform by highpass_biquad
    output = equalizer_biquad_op(waveform)
    count_unequal_element(expect_waveform, output, 0.0001, 0.0001)


def test_equalizer_biquad_pipeline():
    """ mindspore pipeline mode normal testcase:highpass_biquad op"""
    # Original waveform
    waveform = np.array([[0.4063, 0.7729, 0.2325], [0.2687, 0.1426, 0.8987],
                         [0.6914, 0.6681, 0.1783], [0.2704, 0.2680, 0.7975], [0.5880, 0.1776, 0.6323]])
    # Expect waveform
    expect_waveform = np.array([[0.5022, 0.9553, 0.1468], [0.3321, 0.1762, 1.0000],
                                [0.8545, 0.8257, -0.0188], [0.3342, 0.3312, 0.8921], [0.7267, 0.2195, 0.5781]])
    dataset = ds.NumpySlicesDataset(waveform, ["col1"], shuffle=False)
    equalizer_biquad_op = audio.EqualizerBiquad(4000, 1000.0, 5.5, 1)
    # Filtered waveform by equalizer_biquad
    dataset = dataset.map(input_columns=["col1"], operations=equalizer_biquad_op, num_parallel_workers=4)
    i = 0
    for item in dataset.create_dict_iterator(output_numpy=True):
        count_unequal_element(expect_waveform[i, :],
                              item["col1"], 0.0001, 0.0001)
        i += 1


def test_equalizer_biquad_invalid_input():
    """
    Test invalid input of HighpassBiquad
    """
    def test_invalid_input(test_name, sample_rate, center_freq, gain, Q, error, error_msg):
        logger.info("Test EqualizerBiquad with bad input: {0}".format(test_name))
        with pytest.raises(error) as error_info:
            audio.EqualizerBiquad(sample_rate, center_freq, gain, Q)
        assert error_msg in str(error_info.value)

    test_invalid_input("invalid sample_rate parameter type as a float", 44100.5, 1000, 5.5, 0.707, TypeError,
                       "Argument sample_rate with value 44100.5 is not of type [<class 'int'>],"
                       " but got <class 'float'>.")
    test_invalid_input("invalid sample_rate parameter type as a String", "44100", 1000, 5.5, 0.707, TypeError,
                       "Argument sample_rate with value 44100 is not of type [<class 'int'>],"
                       " but got <class 'str'>.")
    test_invalid_input("invalid central_freq parameter type as a String", 44100, "1000", 5.5, 0.707, TypeError,
                       "Argument central_freq with value 1000 is not of type [<class 'float'>, <class 'int'>],"
                       " but got <class 'str'>.")
    test_invalid_input("invalid gain parameter type as a String", 44100, 1000, "5.5", 0.707, TypeError,
                       "Argument gain with value 5.5 is not of type [<class 'float'>, <class 'int'>],"
                       " but got <class 'str'>.")
    test_invalid_input("invalid Q parameter type as a String", 44100, 1000, 5.5, "0.707", TypeError,
                       "Argument Q with value 0.707 is not of type [<class 'float'>, <class 'int'>],"
                       " but got <class 'str'>.")

    test_invalid_input("invalid sample_rate parameter value", 441324343243242342345300, 1000, 5.5, 0.707, ValueError,
                       "Input sample_rate is not within the required interval of [-2147483648, 0) and (0, 2147483647].")
    test_invalid_input("invalid central_freq parameter value", 44100, 3243432434, 5.5, 0.707, ValueError,
                       "Input central_freq is not within the required interval of [-16777216, 16777216].")
    test_invalid_input("invalid sample_rate parameter value", 0, 1000, 5.5, 0.707, ValueError,
                       "Input sample_rate is not within the required interval of [-2147483648, 0) and (0, 2147483647].")
    test_invalid_input("invalid Q parameter value", 44100, 1000, 5.5, 0, ValueError,
                       "Input Q is not within the required interval of (0, 1].")

    test_invalid_input("invalid sample_rate parameter value", None, 1000, 5.5, 0.707, TypeError,
                       "Argument sample_rate with value None is not of type [<class 'int'>], "
                       "but got <class 'NoneType'>.")
    test_invalid_input("invalid central_freq parameter value", 44100, None, 5.5, 0.707, TypeError,
                       "Argument central_freq with value None is not of type [<class 'float'>, <class 'int'>],"
                       " but got <class 'NoneType'>.")
    test_invalid_input("invalid gain parameter value", 44100, 200, None, 0.707, TypeError,
                       "Argument gain with value None is not of type [<class 'float'>, <class 'int'>], "
                       "but got <class 'NoneType'>.")


if __name__ == "__main__":
    test_equalizer_biquad_eager()
    test_equalizer_biquad_pipeline()
    test_equalizer_biquad_invalid_input()
