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
    assert (loss_count / total_count) < rtol, "\ndata_expected_std:{0}\ndata_me_error:{1}\nloss:{2}".format(
        data_expected[greater], data_me[greater], error[greater])


def test_phaser_eager():
    """
    Feature: Phaser
    Description: Test Phaser in eager mode
    Expectation: The results are as expected
    """
    # Original waveform
    waveform = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    # Expect waveform
    expect_waveform = np.array([[0.296, 0.71040004, 1.],
                                [1., 1., 1.]], dtype=np.float32)
    sample_rate = 44100
    # Filtered waveform by phaser
    output = audio.Phaser(sample_rate=sample_rate)(waveform)
    count_unequal_element(expect_waveform, output, 0.0001, 0.0001)


def test_phaser_pipeline():
    """
    Feature: Phaser
    Description: Test Phaser in pipline mode
    Expectation: The results are as expected
    """
    # Original waveform
    waveform = np.array([[0.1, 1.2, 5.3], [0.4, 5.5, 1.6]], dtype=np.float32)
    # Expect waveform
    expect_waveform = np.array([[0.0296, 0.36704, 1.],
                                [0.11840001, 1., 1.]], dtype=np.float32)
    sample_rate = 44100
    dataset = ds.NumpySlicesDataset(waveform, ["waveform"], shuffle=False)
    phaser_op = audio.Phaser(sample_rate)
    # Filtered waveform by phaser
    dataset = dataset.map(
        input_columns=["waveform"], operations=phaser_op)
    i = 0
    for item in dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
        count_unequal_element(expect_waveform[i, :],
                              item['waveform'], 0.0001, 0.0001)
        i += 1


def test_phaser_invalid_input():
    """
    Feature: Phaser
    Description: Test invalid parameter of Phaser
    Expectation: Catch exceptions correctly
    """
    def test_invalid_input(test_name, sample_rate, gain_in, gain_out, delay_ms, decay, mod_speed, sinusoidal, error,
                           error_msg):
        logger.info("Test Phaser with bad input: {0}".format(test_name))
        with pytest.raises(error) as error_info:
            audio.Phaser(sample_rate, gain_in, gain_out, delay_ms, decay, mod_speed, sinusoidal)
        assert error_msg in str(error_info.value)

    test_invalid_input("invalid sample_rate parameter type as a float", 44100.5, 0.4, 0.74, 3.0, 0.4, 0.5, True,
                       TypeError, "Argument sample_rate with value 44100.5 is not of type [<class 'int'>],"
                       " but got <class 'float'>.")
    test_invalid_input("invalid gain_in parameter type as a str", 44100, "1", 0.74, 3.0, 0.4, 0.5, True,
                       TypeError, "Argument gain_in with value 1 is not of type [<class 'float'>, <class 'int'>],"
                       + " but got <class 'str'>.")
    test_invalid_input("invalid gain_out parameter type as a str", 44100, 0.4, "10", 3.0, 0.4, 0.5, True, TypeError,
                       "Argument gain_out with value 10 is not of type [<class 'float'>, <class 'int'>],"
                       + " but got <class 'str'>.")
    test_invalid_input("invalid delay_ms parameter type as a str", 44100, 0.4, 0.74, "2", 0.4, 0.5, True, TypeError,
                       "Argument delay_ms with value 2 is not of type [<class 'float'>, <class 'int'>],"
                       + " but got <class 'str'>.")
    test_invalid_input("invalid decay parameter type as a str", 44100, 0.4, 0.74, 3.0, "0", 0.5, True, TypeError,
                       "Argument decay with value 0 is not of type [<class 'float'>, <class 'int'>],"
                       + " but got <class 'str'>.")
    test_invalid_input("invalid mod_speed parameter type as a str", 44100, 0.4, 0.74, 3.0, 0.4, "3", True, TypeError,
                       "Argument mod_speed with value 3 is not of type [<class 'float'>, <class 'int'>],"
                       + " but got <class 'str'>.")
    test_invalid_input("invalid sinusoidal parameter type as a str", 44100, 0.4, 0.74, 3.0, 0.4, 0.5, "True", TypeError,
                       "Argument sinusoidal with value True is not of type [<class 'bool'>],"
                       + " but got <class 'str'>.")
    test_invalid_input("invalid sample_rate parameter value", 441324343243242342345300, 0.5, 0.74, 3.0, 0.4, 0.5, True,
                       ValueError, "Input sample_rate is not within the required interval of "
                                   "[-2147483648, 2147483647].")
    test_invalid_input("invalid gain_in out of range [0, 1]", 44100, 2.0, 0.74, 3.0, 0.4, 0.5, True, ValueError,
                       "Input gain_in is not within the required interval of [0, 1].")
    test_invalid_input("invalid gain_out out of range [0, 1e9]", 44100, 0.4, -2.0, 3.0, 0.4, 0.5, True, ValueError,
                       "Input gain_out is not within the required interval of [0, 1000000000.0].")
    test_invalid_input("invalid delay_ms out of range [0, 5.0]", 44100, 0.4, 0.74, 6.0, 0.4, 0.5, True, ValueError,
                       "Input delay_ms is not within the required interval of [0, 5.0].")
    test_invalid_input("invalid decay out of range [0, 0.99]", 44100, 0.4, 0.74, 3.0, 1.2, 0.5, True, ValueError,
                       "Input decay is not within the required interval of [0, 0.99].")
    test_invalid_input("invalid mod_speed out of range [0.1, 2]", 44100, 0.4, 0.74, 3.0, 0.4, 0.003, True, ValueError,
                       "Input mod_speed is not within the required interval of [0.1, 2].")


if __name__ == "__main__":
    test_phaser_eager()
    test_phaser_pipeline()
    test_phaser_invalid_input()
