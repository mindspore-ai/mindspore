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

import mindspore.dataset as ds
import mindspore.dataset.audio as audio
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


def test_sliding_window_cmn_eager():
    """
    Feature: Test the basic function in eager mode.
    Description: Mindspore eager mode normal testcase:sliding_window_cmn op.
    Expectation: Compile done without error.
    """
    # Original waveform
    waveform_1 = np.array([[[0.0000, 0.1000, 0.2000],
                            [0.3000, 0.4000, 0.5000]],
                           [[0.6000, 0.7000, 0.8000],
                            [0.9000, 1.0000, 1.1000]]], dtype=np.float64)
    # Expect waveform
    expect_waveform_1 = np.array([[[-0.1500, -0.1500, -0.1500],
                                   [0.1500, 0.1500, 0.1500]],
                                  [[-0.1500, -0.1500, -0.1500],
                                   [0.1500, 0.1500, 0.1500]]], dtype=np.float64)
    sliding_window_cmn_op_1 = audio.SlidingWindowCmn(500, 200, False, False)
    # Filtered waveform by sliding_window_cmn
    output_1 = sliding_window_cmn_op_1(waveform_1)
    count_unequal_element(expect_waveform_1, output_1, 0.0001, 0.0001)

    # Original waveform
    waveform_2 = np.array([[0.0050, 0.0306, 0.6146, 0.7620, 0.6369],
                           [0.9525, 0.0362, 0.6721, 0.6867, 0.8466]], dtype=np.float32)
    # Expect waveform
    expect_waveform_2 = np.array([[-1.0000, -1.0000, -1.0000, 1.0000, -1.0000],
                                  [1.0000, 1.0000, 1.0000, -1.0000, 1.0000]], dtype=np.float32)
    sliding_window_cmn_op_2 = audio.SlidingWindowCmn(600, 100, False, True)
    # Filtered waveform by sliding_window_cmn
    output_2 = sliding_window_cmn_op_2(waveform_2)
    count_unequal_element(expect_waveform_2, output_2, 0.0001, 0.0001)

    # Original waveform
    waveform_3 = np.array([[[0.3764, 0.4168, 0.0635, 0.7082, 0.4596, 0.3457, 0.8438, 0.8860, 0.9151, 0.5746,
                             0.6630, 0.0260, 0.2631, 0.7410, 0.5627, 0.6749, 0.7099, 0.1120, 0.4794, 0.2778],
                            [0.4157, 0.2246, 0.2488, 0.2686, 0.0562, 0.4422, 0.9407, 0.0756, 0.5737, 0.7501,
                             0.3122, 0.7982, 0.3034, 0.1880, 0.2298, 0.0961, 0.7439, 0.9947, 0.8156, 0.2907]]],
                          dtype=np.float64)
    # Expect waveform
    expect_waveform_3 = np.array([[[-1.0000, 1.0000, -1.0000, 1.0000, 1.0000, -1.0000, -1.0000, 1.0000,
                                    1.0000, -1.0000, 1.0000, -1.0000, -1.0000, 1.0000, 1.0000, 1.0000,
                                    -1.0000, -1.0000, -1.0000, -1.0000],
                                   [1.0000, -1.0000, 1.0000, -1.0000, -1.0000, 1.0000, 1.0000, -1.0000,
                                    -1.0000, 1.0000, -1.0000, 1.0000, 1.0000, -1.0000, -1.0000, -1.0000,
                                    1.0000, 1.0000, 1.0000, 1.0000]]], dtype=np.float64)
    sliding_window_cmn_op_3 = audio.SlidingWindowCmn(3, 0, True, True)
    # Filtered waveform by sliding_window_cmn
    output_3 = sliding_window_cmn_op_3(waveform_3)
    count_unequal_element(expect_waveform_3, output_3, 0.0001, 0.0001)


def test_sliding_window_cmn_pipeline():
    """
    Feature: Test the basic function in pipeline mode.
    Description: Mindspore pipeline mode normal testcase:sliding_window_cmn op.
    Expectation: Compile done without error.
    """
    # Original waveform
    waveform = np.array([[[3.2, 2.1, 1.3], [6.2, 5.3, 6]]], dtype=np.float64)
    # Expect waveform
    expect_waveform = np.array([[[-1.0000, -1.0000, -1.0000],
                                 [1.0000, 1.0000, 1.0000]]], dtype=np.float64)
    dataset = ds.NumpySlicesDataset(waveform, ["audio"], shuffle=False)
    sliding_window_cmn_op = audio.SlidingWindowCmn(600, 100, False, True)
    # Filtered waveform by sliding_window_cmn
    dataset = dataset.map(input_columns=["audio"], operations=sliding_window_cmn_op, num_parallel_workers=8)
    i = 0
    for item in dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
        count_unequal_element(expect_waveform[i, :],
                              item['audio'], 0.0001, 0.0001)
        i += 1


def test_sliding_window_cmn_invalid_input():
    """
    Feature: Test the validate function with invalid parameters.
    Description: Mindspore invalid parameters testcase:sliding_window_cmn op.
    Expectation: Compile done without error.
    """
    def test_invalid_input(test_name, cmn_window, min_cmn_window, center, norm_vars, error, error_msg):
        logger.info("Test SlidingWindowCmn with bad input: {0}".format(test_name))
        with pytest.raises(error) as error_info:
            audio.SlidingWindowCmn(cmn_window, min_cmn_window, center, norm_vars)
        assert error_msg in str(error_info.value)

    test_invalid_input("invalid cmn_window parameter type as a String", "600", 100, False, False, TypeError,
                       "Argument cmn_window with value 600 is not of type [<class 'int'>],"
                       " but got <class 'str'>.")
    test_invalid_input("invalid cmn_window parameter value", 441324343243242342345300, 100, False, False, ValueError,
                       "Input cmn_window is not within the required interval of [0, 2147483647].")
    test_invalid_input("invalid min_cmn_window parameter type as a String", 600, "100", False, False, TypeError,
                       "Argument min_cmn_window with value 100 is not of type [<class 'int'>],"
                       " but got <class 'str'>.")
    test_invalid_input("invalid min_cmn_window parameter value", 600, 441324343243242342345300, False, False,
                       ValueError, "Input min_cmn_window is not within the required interval of [0, 2147483647].")
    test_invalid_input("invalid center parameter type as a String", 600, 100, "False", False, TypeError,
                       "Argument center with value False is not of type [<class 'bool'>],"
                       " but got <class 'str'>.")
    test_invalid_input("invalid norm_vars parameter type as a String", 600, 100, False, "False", TypeError,
                       "Argument norm_vars with value False is not of type [<class 'bool'>],"
                       " but got <class 'str'>.")


if __name__ == '__main__':
    test_sliding_window_cmn_eager()
    test_sliding_window_cmn_pipeline()
    test_sliding_window_cmn_invalid_input()
