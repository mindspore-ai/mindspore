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


def test_gain_eager():
    """
    Feature: Gain
    Description: Test Gain in eager mode
    Expectation: The data is processed successfully
    """
    logger.info("test Gain in eager mode")

    # Original waveform
    waveform = np.array([1, 2, 3, 4, 5, 6], dtype=np.float64)
    # Expect waveform
    expect_waveform = np.array([1.1220184, 2.2440369, 3.3660554, 4.4880738, 5.6100923, 6.7321107], dtype=np.float64)
    gain_op = audio.Gain()
    # Filtered waveform by Gain
    output = gain_op(waveform)
    count_unequal_element(expect_waveform, output, 0.00001, 0.00001)


def test_gain_pipeline():
    """
    Feature: Gain
    Description: Test Gain in pipeline mode
    Expectation: The data is processed successfully
    """
    logger.info("test Gain in pipeline mode")

    # Original waveform
    waveform = np.array([[1, 2, 3], [0.1, 0.2, 0.3]], dtype=np.float64)
    # Expect waveform
    expect_waveform = np.array([[1.05925, 2.1185, 3.1778],
                                [0.10592537, 0.21185075, 0.31777612]], dtype=np.float64)
    dataset = ds.NumpySlicesDataset(waveform, ["audio"], shuffle=False)
    gain_op = audio.Gain(0.5)
    # Filtered waveform by Gain
    dataset = dataset.map(input_columns=["audio"], operations=gain_op, num_parallel_workers=8)
    i = 0
    for item in dataset.create_dict_iterator(output_numpy=True):
        count_unequal_element(expect_waveform[i, :], item['audio'], 0.00001, 0.00001)
        i += 1


def test_gain_invalid_input():
    """
    Feature: Gain
    Description: Test param check of Gain
    Expectation: Throw correct error and message
    """
    logger.info("test param check of Gain")

    def test_invalid_input(test_name, gain_db, error, error_msg):
        logger.info("Test Gain with bad input: {0}".format(test_name))
        with pytest.raises(error) as error_info:
            audio.Gain(gain_db)
        assert error_msg in str(error_info.value)

    test_invalid_input("invalid gain_db parameter type as a String", "1.0", TypeError,
                       "Argument gain_db with value 1.0 is not of type [<class 'float'>, <class 'int'>],"
                       " but got <class 'str'>.")
    test_invalid_input("invalid gain_db parameter value", 122323242445423534543, ValueError,
                       "Input gain_db is not within the required interval of [-16777216, 16777216].")


if __name__ == "__main__":
    test_gain_eager()
    test_gain_pipeline()
    test_gain_invalid_input()
