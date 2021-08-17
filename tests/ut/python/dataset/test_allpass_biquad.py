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
    assert (loss_count / total_count) < rtol, "\ndata_expected_std:{0}\ndata_me_error:{1}\nloss:{2}".format(
        data_expected[greater], data_me[greater], error[greater])


def test_func_allpass_biquad_eager():
    """ mindspore eager mode normal testcase:allpass_biquad op"""

    # Original waveform
    waveform = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64)
    # Expect waveform
    expect_waveform = np.array([[0.96049707, 1.0, 1.0], [1.0, 1.0, 1.0]], dtype=np.float64)
    allpass_biquad_op = audio.AllpassBiquad(44100, 200.0, 0.707)
    # Filtered waveform by allpassbiquad
    output = allpass_biquad_op(waveform)
    count_unequal_element(expect_waveform, output, 0.0001, 0.0001)


def test_func_allpass_biquad_pipeline():
    """ mindspore pipeline mode normal testcase:allpass_biquad op"""

    # Original waveform
    waveform = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64)
    # Expect waveform
    expect_waveform = np.array([[0.96049707, 1.0, 1.0], [1.0, 1.0, 1.0]], dtype=np.float64)
    label = np.random.sample((2, 1))
    data = (waveform, label)
    dataset = ds.NumpySlicesDataset(data, ["channel", "sample"], shuffle=False)
    allpass_biquad_op = audio.AllpassBiquad(44100, 200.0)
    # Filtered waveform by allpassbiquad
    dataset = dataset.map(input_columns=["channel"], operations=allpass_biquad_op, num_parallel_workers=8)
    i = 0
    for item in dataset.create_dict_iterator(output_numpy=True):
        count_unequal_element(expect_waveform[i, :], item['channel'], 0.0001, 0.0001)
        i += 1


def test_invalid_input_all():
    waveform = np.random.rand(2, 1000)

    def test_invalid_input(test_name, sample_rate, central_freq, Q, error, error_msg):
        logger.info("Test Allpassallpassiquad with bad input: {0}".format(test_name))
        with pytest.raises(error) as error_info:
            audio.AllpassBiquad(sample_rate, central_freq, Q)(waveform)
        assert error_msg in str(error_info.value)

    test_invalid_input("invalid sample_rate parameter type as a float", 44100.5, 200, 0.707, TypeError,
                       "Argument sample_rate with value 44100.5 is not of type [<class 'int'>],"
                       + " but got <class 'float'>.")
    test_invalid_input("invalid sample_rate parameter type as a String", "44100", 200, 0.707, TypeError,
                       "Argument sample_rate with value 44100 is not of type [<class 'int'>]," +
                       " but got <class 'str'>.")
    test_invalid_input("invalid contral_freq parameter type as a String", 44100, "200", 0.707, TypeError,
                       "Argument central_freq with value 200 is not of type [<class 'float'>, <class 'int'>],"
                       + " but got <class 'str'>.")
    test_invalid_input("invalid Q parameter type as a String", 44100, 200, "0.707", TypeError,
                       "Argument Q with value 0.707 is not of type [<class 'float'>, <class 'int'>],"
                       + " but got <class 'str'>.")
    test_invalid_input("invalid sample_rate parameter value", 441324343243242342345300, 200, 0.707, ValueError,
                       "Input sample_rate is not within the required interval of [-2147483648, 0) and (0, 2147483647].")
    test_invalid_input("invalid contral_freq parameter value", 44100, 32434324324234321, 0.707, ValueError,
                       "Input central_freq is not within the required interval of [-16777216, 16777216].")
    test_invalid_input("invalid sample_rate parameter value", None, 200, 0.707, TypeError,
                       "Argument sample_rate with value None is not of type [<class 'int'>],"
                       + " but got <class 'NoneType'>.")
    test_invalid_input("invalid central_rate parameter value", 44100, None, 0.707, TypeError,
                       "Argument central_freq with value None is not of type [<class 'float'>, <class 'int'>],"
                       + " but got <class 'NoneType'>.")
    test_invalid_input("invalid sample_rate parameter value", 0, 200, 0.707, ValueError,
                       "Input sample_rate is not within the required interval of [-2147483648, 0) and (0, 2147483647].")
    test_invalid_input("invalid Q parameter value", 44100, 200, 1.707, ValueError,
                       "Input Q is not within the required interval of (0, 1].")


if __name__ == '__main__':
    test_func_allpass_biquad_eager()
    test_func_allpass_biquad_pipeline()
    test_invalid_input_all()
