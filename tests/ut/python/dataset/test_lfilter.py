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


def test_func_lfilter_eager():
    """ mindspore eager mode normal testcase:deemph_biquad op"""
    # Original waveform
    waveform = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=np.float64)
    # Expect waveform
    expect_waveform = np.array([[0.25, 0.45, 0.425],
                                [1., 1., 0.35]], dtype=np.float64)
    a_coeffs = [0.2, 0.2, 0.3]
    b_coeffs = [0.5, 0.4, 0.2]
    lfilter_op = audio.LFilter(a_coeffs, b_coeffs, True)
    output = lfilter_op(waveform)
    count_unequal_element(expect_waveform, output, 0.0001, 0.0001)


def test_func_lfilter_pipeline():
    """ mindspore pipeline mode normal testcase:lfilter op"""

    # Original waveform
    waveform = np.array([[0.1, 0.2, 0.3, 0.4], [0.4, 0.5, 0.6, 0.7]], dtype=np.float64)
    # Expect waveform
    expect_waveform = np.array([[0.4, 0.5, 0.6, 1.],
                                [1., 0.8, 0.9, 1.]], dtype=np.float64)
    data = (waveform, waveform.shape)
    a_coeffs = [0.1, 0.2, 0.3]
    b_coeffs = [0.4, 0.5, 0.6]
    dataset = ds.NumpySlicesDataset(data, ["channel", "sample"], shuffle=False)
    lfilter_op = audio.LFilter(a_coeffs, b_coeffs)
    # Filtered waveform by lfilter
    dataset = dataset.map(input_columns=["channel"], operations=lfilter_op, num_parallel_workers=8)
    i = 0
    for data in dataset.create_dict_iterator(output_numpy=True):
        count_unequal_element(expect_waveform[i, :], data['channel'], 0.0001, 0.0001)
        i += 1


def test_invalid_input_all():
    waveform = np.random.rand(2, 1000)

    def test_invalid_input(test_name, a_coeffs, b_coeffs, clamp, error, error_msg):
        logger.info("Test LFilter with bad input: {0}".format(test_name))
        with pytest.raises(error) as error_info:
            audio.LFilter(a_coeffs, b_coeffs, clamp)(waveform)
        assert error_msg in str(error_info.value)

    a_coeffs = ['0.1', '0.2', '0.3']
    b_coeffs = [0.1, 0.2, 0.3]
    test_invalid_input("invalid a_coeffs parameter type as a string", a_coeffs, b_coeffs, True, TypeError,
                       "Argument a_coeffs[0] with value 0.1 is not of type [<class 'float'>, <class 'int'>], "
                       + "but got <class 'str'>.")
    a_coeffs = [234322354352353453651, 0.2, 0.3]
    b_coeffs = [0.1, 0.2, 0.3]
    test_invalid_input("invalid a_coeffs parameter value", a_coeffs, b_coeffs, True, ValueError,
                       "Input a_coeffs[0] is not within the required interval of [-16777216, 16777216].")
    a_coeffs = [0.1, 0.2, 0.3]
    b_coeffs = [0.1, 0.2, 0.3]
    test_invalid_input("invalid clamp parameter type as a String", a_coeffs, b_coeffs, "True", TypeError,
                       "Argument clamp with value True is not of type [<class 'bool'>],"
                       + " but got <class 'str'>.")


if __name__ == '__main__':
    test_func_lfilter_eager()
    test_func_lfilter_pipeline()
    test_invalid_input_all()
    