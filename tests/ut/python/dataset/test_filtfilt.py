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
import pytest
import numpy as np

import mindspore.dataset as ds
import mindspore.dataset.audio.transforms as audio
from mindspore import log as logger
from util import diff_mse


def count_unequal_element(data_expected, data_me, rtol, atol):
    assert data_expected.shape == data_me.shape
    total_count = len(data_expected.flatten())
    error = np.abs(data_expected - data_me)
    greater = np.greater(error, atol + np.abs(data_expected) * rtol)
    loss_count = np.count_nonzero(greater)
    assert (loss_count / total_count) < rtol, \
        "\ndata_expected_std:{0}\ndata_me_error:{1}\nloss:{2}". \
        format(data_expected[greater], data_me[greater], error[greater])


def test_filtfilt_eager():
    """
    Feature: Filtfilt
    Description: Test Filtfilt op in eager mode under normal test case
    Expectation: Output is equal to the expected output
    """
    logger.info("test_filtfilt_eager")
    # construct input
    waveform = np.array([[0.1, 0.2, 0.3], [0.3, 0.4, 0.5]], dtype=np.float64)
    a_coeffs = [0.1, 0.2, 0.3]
    b_coeffs = [0.4, 0.5, 0.6]
    clanmp_input = True

    filtfilt_op = audio.Filtfilt(a_coeffs, b_coeffs, clamp=clanmp_input)
    our_waveform = filtfilt_op(waveform)

    # use np flip
    forward_filtered_op = audio.LFilter(a_coeffs, b_coeffs, clamp=False)
    backward_filtered_op = audio.LFilter(a_coeffs, b_coeffs, clamp=clanmp_input)

    # use np flip
    forward_filtered_waveform = forward_filtered_op(waveform)
    backward_filtered_waveform = backward_filtered_op(np.flip(forward_filtered_waveform, -1))
    expect_waveform = np.flip(backward_filtered_waveform, -1)

    mse = diff_mse(our_waveform, expect_waveform)
    assert mse == 0
    logger.info("test_filtfilt_eager Success")


def test_filtfilt_pipeline():
    """
    Feature: Filtfilt
    Description: Test Filtfilt op in pipeline mode under normal test case
    Expectation: Output is equal to the expected output
    """
    logger.info("test_filtfilt_pipeline")
    # construct input
    waveform = np.array([[0.1, 0.2, 0.3, 0.4], [0.4, 0.5, 0.6, 0.7]], dtype=np.float64)
    a_coeffs = [0.1, 0.2, 0.3]
    b_coeffs = [0.4, 0.5, 0.6]

    expect_waveform = np.array([[1, 0.2, -1, 1], [1, 0.5, -1, 1]], dtype=np.float64)

    data = (waveform, waveform.shape)
    dataset = ds.NumpySlicesDataset(data, ["channel", "sample"], shuffle=False)
    filtfilt_op = audio.Filtfilt(a_coeffs, b_coeffs, clamp=True)
    # Filtered waveform by lfilter
    dataset = dataset.map(input_columns=["channel"], operations=filtfilt_op, num_parallel_workers=8)
    i = 0
    for data in dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
        count_unequal_element(expect_waveform[i, :], data['channel'], 0.0001, 0.0001)
        i += 1


def test_filtfilt_invalid_input_all():
    """
    Feature: Filtfilt
    Description: Test Filtfilt op with invalid input
    Expectation: Correct error is raised as expected
    """
    waveform = np.random.rand(2, 1000)

    def test_invalid_input(test_name, a_coeffs, b_coeffs, clamp, error, error_msg):
        logger.info("Test Filtfilt with bad input: {0}".format(test_name))
        with pytest.raises(error) as error_info:
            audio.Filtfilt(a_coeffs, b_coeffs, clamp)(waveform)
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
    test_filtfilt_eager()
    test_filtfilt_pipeline()
    test_filtfilt_invalid_input_all()
