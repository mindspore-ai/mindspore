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
"""
Testing RiaaBiquad op in DE.
"""

import numpy as np
import pytest

import mindspore.dataset as ds
import mindspore.dataset.audio as audio
from mindspore import log as logger


def count_unequal_element(data_expected, data_me, rtol, atol):
    assert data_expected.shape == data_me.shape
    total_count = len(data_expected.flatten())
    error = np.abs(data_expected - data_me)
    greater = np.greater(error, atol + np.abs(data_expected) * rtol)
    loss_count = np.count_nonzero(greater)
    assert (loss_count / total_count) < rtol, "\ndata_expected_std:{0}\ndata_me_error:{1}\nloss:{2}".format(
        data_expected[greater], data_me[greater], error[greater])


def test_riaa_biquad_eager():
    """
    Feature: RiaaBiquad op
    Description: Test RiaaBiquad op in eager mode under normal test case
    Expectation: Output is equal to the expected output
    """
    # Original waveform
    waveform = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64)
    # Expect waveform
    expect_waveform = np.array([[0.23806122, 0.70914434, 1.],
                                [0.95224489, 1., 1.]], dtype=np.float64)
    riaa_biquad_op = audio.RiaaBiquad(44100)
    # Filtered waveform by riaabiquad
    output = riaa_biquad_op(waveform)
    count_unequal_element(expect_waveform, output, 0.0001, 0.0001)


def test_riaa_biquad_pipeline():
    """
    Feature: RiaaBiquad op
    Description: Test RiaaBiquad op in pipeline mode under normal test case
    Expectation: Output is equal to the expected output
    """
    # Original waveform
    waveform = np.array([[1.47, 4.722, 5.863], [0.492, 0.235, 0.56]], dtype=np.float32)
    # Expect waveform
    expect_waveform = np.array([[0.18626465, 0.7859906, 1.],
                                [0.06234163, 0.09258664, 0.15710703]], dtype=np.float64)
    dataset = ds.NumpySlicesDataset(waveform, ["waveform"], shuffle=False)
    riaa_biquad_op = audio.RiaaBiquad(88200)
    # Filtered waveform by riaabiquad
    dataset = dataset.map(input_columns=["waveform"], operations=riaa_biquad_op)
    i = 0
    for item in dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
        count_unequal_element(expect_waveform[i, :], item['waveform'], 0.0001, 0.0001)
        i += 1


def test_riaa_biquad_invalid_parameter():
    """
    Feature: RiaaBiquad op
    Description: Test RiaaBiquad op with invalid parameter
    Expectation: Error is raised as expected
    """
    def test_invalid_input(test_name, sample_rate, error, error_msg):
        logger.info("Test RiaaBiquad with bad input: {0}".format(test_name))
        with pytest.raises(error) as error_info:
            audio.RiaaBiquad(sample_rate)
        assert error_msg in str(error_info.value)

    test_invalid_input("invalid sample_rate parameter type as a float", 44100.5, TypeError,
                       "Argument sample_rate with value 44100.5 is not of type [<class 'int'>],"
                       " but got <class 'float'>.")
    test_invalid_input("invalid sample_rate parameter type as a String", "44100", TypeError,
                       "Argument sample_rate with value 44100 is not of type [<class 'int'>],"
                       + " but got <class 'str'>.")
    test_invalid_input("invalid sample_rate parameter value", 45670, ValueError,
                       "sample_rate should be one of [44100, 48000, 88200, 96000], but got 45670.")


if __name__ == "__main__":
    test_riaa_biquad_eager()
    test_riaa_biquad_pipeline()
    test_riaa_biquad_invalid_parameter()
