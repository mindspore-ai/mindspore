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


def test_db_to_amplitude_eager():
    """
    Feature: DBToAmplitude
    Description: Test DBToAmplitude in eager mode
    Expectation: The data is processed successfully
    """
    logger.info("mindspore eager mode normal testcase:DBToAmplitude op")

    # Original waveform
    waveform = np.array([1, 2, 3, 4, 5, 6], dtype=np.float64)
    # Expect waveform
    expect_waveform = np.array([3.1698, 5.0238, 7.9621, 12.6191, 20.0000, 31.6979], dtype=np.float64)
    DBToAmplitude_op = audio.DBToAmplitude(2, 2)
    # Filtered waveform by DBToAmplitude
    output = DBToAmplitude_op(waveform)
    count_unequal_element(expect_waveform, output, 0.0001, 0.0001)


def test_db_to_amplitude_pipeline():
    """
    Feature: DBToAmplitude
    Description: Test DBToAmplitude in pipeline mode
    Expectation: The data is processed successfully
    """
    logger.info("mindspore pipeline mode normal testcase:DBToAmplitude op")

    # Original waveform
    waveform = np.array([[2, 2, 3], [0.1, 0.2, 0.3]], dtype=np.float64)
    # Expect waveform
    expect_waveform = np.array([[2.5119, 2.5119, 3.9811],
                                [1.0471, 1.0965, 1.1482]], dtype=np.float64)

    dataset = ds.NumpySlicesDataset(waveform, ["audio"], shuffle=False)
    DBToAmplitude_op = audio.DBToAmplitude(1, 2)
    # Filtered waveform by DBToAmplitude
    dataset = dataset.map(input_columns=["audio"], operations=DBToAmplitude_op, num_parallel_workers=8)
    i = 0
    for item in dataset.create_dict_iterator(output_numpy=True):
        count_unequal_element(expect_waveform[i, :], item['audio'], 0.0001, 0.0001)
        i += 1


def test_db_to_amplitude_invalid_input():
    """
    Feature: DBToAmplitude
    Description: Test param check of DBToAmplitude
    Expectation: Throw correct error and message
    """
    logger.info("mindspore eager mode invalid input testcase:filter_wikipedia_xml op")

    def test_invalid_input(test_name, ref, power, error, error_msg):
        logger.info("Test DBToAmplitude with bad input: {0}".format(test_name))
        with pytest.raises(error) as error_info:
            audio.DBToAmplitude(ref, power)
        assert error_msg in str(error_info.value)

    test_invalid_input("invalid ref parameter type as a String", "1.0", 1.0, TypeError,
                       "Argument ref with value 1.0 is not of type [<class 'float'>, <class 'int'>],"
                       " but got <class 'str'>.")
    test_invalid_input("invalid ref parameter value", 122323242445423534543, 1.0, ValueError,
                       "Input ref is not within the required interval of [-16777216, 16777216].")
    test_invalid_input("invalid power parameter type as a String", 1.0, "1.0", TypeError,
                       "Argument power with value 1.0 is not of type [<class 'float'>, <class 'int'>],"
                       " but got <class 'str'>.")
    test_invalid_input("invalid power parameter value", 1.0, 1343454254325445, ValueError,
                       "Input power is not within the required interval of [-16777216, 16777216].")


if __name__ == "__main__":
    test_db_to_amplitude_eager()
    test_db_to_amplitude_pipeline()
    test_db_to_amplitude_invalid_input()
