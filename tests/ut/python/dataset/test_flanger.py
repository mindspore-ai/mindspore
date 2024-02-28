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
from mindspore.dataset.audio.utils import Modulation, Interpolation


def count_unequal_element(data_expected, data_me, rtol, atol):
    assert data_expected.shape == data_me.shape
    total_count = len(data_expected.flatten())
    error = np.abs(data_expected - data_me)
    greater = np.greater(error, atol + np.abs(data_expected) * rtol)
    loss_count = np.count_nonzero(greater)
    assert (loss_count / total_count) < rtol, "\ndata_expected_std:{0}\ndata_me_error:{1}\nloss:{2}".format(
        data_expected[greater], data_me[greater], error[greater])


def test_flanger_eager_sinusoidal_linear_float64():
    """
    Feature: Flanger op
    Description: Test Flanger op in eager mode under normal test case with float64
    Expectation: Output is equal to the expected output
    """
    # Original waveform
    waveform = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=np.float64)
    # Expect waveform
    expect_waveform = np.array([[0.10000000000, 0.19999999536, 0.29999998145],
                                [0.23391812865, 0.29239766081, 0.35087719298]], dtype=np.float64)
    flanger_op = audio.Flanger(44100, 0.0, 2.0, 0.0, 71.0, 0.5, 25.0, Modulation.SINUSOIDAL, Interpolation.LINEAR)
    # Filtered waveform by flanger
    output = flanger_op(waveform)
    count_unequal_element(expect_waveform, output, 0.0001, 0.0001)


def test_flanger_eager_triangular_linear_float32():
    """
    Feature: Flanger op
    Description: Test Flanger op in eager mode under normal test case with float32
    Expectation: Output is equal to the expected output
    """
    # Original waveform
    waveform = np.array([[-1.2, 2, -3.6], [1, 2.4, 3.7]], dtype=np.float32)
    # Expect waveform
    expect_waveform = np.array([[-1.0000000000, 1.0000000000, -1.0000000000],
                                [0.58479529619, 1.0000000000, 1.0000000000]], dtype=np.float32)
    flanger_op = audio.Flanger(44100, 0.0, 2.0, 0.0, 71.0, 0.5, 25.0, Modulation.TRIANGULAR, Interpolation.LINEAR)
    # Filtered waveform by flanger
    output = flanger_op(waveform)
    count_unequal_element(expect_waveform, output, 0.0001, 0.0001)


def test_flanger_eager_triangular_linear_int():
    """
    Feature: Flanger op
    Description: Test Flanger op in eager mode under normal test case with int
    Expectation: Output is equal to the expected output
    """
    # Original waveform
    waveform = np.array([[-2, -3, 0], [2, 2, 3]], dtype=np.int)
    # Expect waveform
    expect_waveform = np.array([[-1, -1, 0],
                                [1, 1, 1]], dtype=np.int)
    flanger_op = audio.Flanger(44100, 0.0, 2.0, 0.0, 71.0, 0.5, 25.0, Modulation.TRIANGULAR, Interpolation.LINEAR)
    # Filtered waveform by flanger
    output = flanger_op(waveform)
    count_unequal_element(expect_waveform, output, 0.0001, 0.0001)


def test_flanger_shape_221():
    """
    Feature: Flanger op
    Description: Test Flanger op in eager mode under normal test case with shape of 2 * 2 * 1
    Expectation: Output is equal to the expected output
    """
    # Original waveform
    waveform = np.array([[[1], [1.1]], [[0.9], [0.6]]], dtype=np.float64)
    # Expect waveform
    expect_waveform = np.array([[[1.00000000],
                                 [0.64327485]],

                                [[0.90000000],
                                 [0.35087719]]], dtype=np.float64)

    flanger_op = audio.Flanger(44100)
    # Filtered waveform by flanger
    output = flanger_op(waveform)
    count_unequal_element(expect_waveform, output, 0.0001, 0.0001)


def test_flanger_shape_11211():
    """
    Feature: Flanger op
    Description: Test Flanger op in eager mode under normal test case with shape of 1 * 1 * 2 * 1 * 1
    Expectation: Output is equal to the expected output
    """
    # Original waveform
    waveform = np.array([[[[[0.44]], [[0.55]]]]], dtype=np.float64)
    # Expect waveform
    expect_waveform = np.array([[[[[0.44000000]], [[0.55000000]]]]], dtype=np.float64)

    flanger_op = audio.Flanger(44100)
    # Filtered waveform by flanger
    output = flanger_op(waveform)
    count_unequal_element(expect_waveform, output, 0.0001, 0.0001)


def test_flanger_pipeline():
    """
    Feature: Flanger op
    Description: Test Flanger op in pipeline mode under normal test case
    Expectation: Output is equal to the expected output
    """
    # Original waveform
    waveform = np.array([[[1.1, 1.2, 1.3], [1.4, 1.5, 1.6]]], dtype=np.float64)
    # Expect waveform
    expect_waveform = np.array([[[1.00000000000, 1.00000000000, 1.00000000000],
                                 [0.81871345029, 0.87719298245, 0.93567251461]]], dtype=np.float64)
    data = (waveform, np.random.sample((1, 2, 1)))
    dataset = ds.NumpySlicesDataset(data, ["channel", "sample"], shuffle=False)
    flanger_op = audio.Flanger(44100)
    # Filtered waveform by flanger
    dataset = dataset.map(
        input_columns=["channel"], operations=flanger_op, num_parallel_workers=1)
    i = 0
    for item in dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
        count_unequal_element(expect_waveform[i, :],
                              item['channel'], 0.0001, 0.0001)
        i += 1


def test_invalid_flanger_input():
    """
    Feature: Flanger op
    Description: Test Flanger op with invalid input
    Expectation: Error is raised as expected
    """
    def test_invalid_input(test_name, sample_rate, delay, depth, regen, width, speed, phase, modulation, interpolation,
                           error, error_msg):
        logger.info("Test Flanger with bad input: {0}".format(test_name))
        with pytest.raises(error) as error_info:
            audio.Flanger(sample_rate, delay, depth, regen, width, speed, phase, modulation, interpolation)
        assert error_msg in str(error_info.value)

    test_invalid_input("invalid sample_rate parameter value", 0, 0.0, 2.0, 0.0, 71.0, 0.5, 25.0,
                       Modulation.SINUSOIDAL, Interpolation.LINEAR, ValueError,
                       "Input sample_rate is not within the required interval of [-2147483648, 0) and (0, 2147483647].")
    test_invalid_input("invalid sample_rate parameter type as a float", 44100.5, 0.0, 2.0, 0.0, 71.0, 0.5, 25.0,
                       Modulation.SINUSOIDAL, Interpolation.LINEAR, TypeError,
                       "Argument sample_rate with value 44100.5 is not of "
                       "type [<class 'int'>], but got <class 'float'>.")
    test_invalid_input("invalid sample_rate parameter type as a String", "44100", 0.0, 2.0, 0.0, 71.0, 0.5, 25.0,
                       Modulation.SINUSOIDAL, Interpolation.LINEAR, TypeError,
                       "Argument sample_rate with value 44100 is not of "
                       "type [<class 'int'>], but got <class 'str'>.")

    test_invalid_input("invalid delay parameter type as a String", 44100, "0.0", 2.0, 0.0, 71.0, 0.5, 25.0,
                       Modulation.SINUSOIDAL, Interpolation.LINEAR, TypeError,
                       "Argument delay with value 0.0 is not of type [<class 'float'>, <class 'int'>],"
                       " but got <class 'str'>.")
    test_invalid_input("invalid delay parameter value", 44100, 50, 2.0, 0.0, 71.0, 0.5, 25.0,
                       Modulation.SINUSOIDAL, Interpolation.LINEAR, ValueError,
                       "Input delay is not within the required interval of [0, 30].")

    test_invalid_input("invalid depth parameter type as a String", 44100, 0.0, "2.0", 0.0, 71.0, 0.5, 25.0,
                       Modulation.SINUSOIDAL, Interpolation.LINEAR, TypeError,
                       "Argument depth with value 2.0 is not of type [<class 'float'>, <class 'int'>],"
                       " but got <class 'str'>.")
    test_invalid_input("invalid depth parameter value", 44100, 0.0, 50.0, 0.0, 71.0, 0.5, 25.0,
                       Modulation.SINUSOIDAL, Interpolation.LINEAR, ValueError,
                       "Input depth is not within the required interval of [0, 10].")

    test_invalid_input("invalid regen parameter type as a String", 44100, 0.0, 2.0, "0.0", 71.0, 0.5, 25.0,
                       Modulation.SINUSOIDAL, Interpolation.LINEAR, TypeError,
                       "Argument regen with value 0.0 is not of type [<class 'float'>, <class 'int'>],"
                       " but got <class 'str'>.")
    test_invalid_input("invalid regen parameter value", 44100, 0.0, 2.0, 100.0, 71.0, 0.5, 25.0,
                       Modulation.SINUSOIDAL, Interpolation.LINEAR, ValueError,
                       "Input regen is not within the required interval of [-95, 95].")

    test_invalid_input("invalid width parameter type as a String", 44100, 0.0, 2.0, 0.0, "71.0", 0.5, 25.0,
                       Modulation.SINUSOIDAL, Interpolation.LINEAR, TypeError,
                       "Argument width with value 71.0 is not of type [<class 'float'>, <class 'int'>],"
                       " but got <class 'str'>.")
    test_invalid_input("invalid width parameter value", 44100, 0.0, 2.0, 0.0, 150.0, 0.5, 25.0,
                       Modulation.SINUSOIDAL, Interpolation.LINEAR, ValueError,
                       "Input width is not within the required interval of [0, 100].")

    test_invalid_input("invalid speed parameter type as a String", 44100, 0.0, 2.0, 0.0, 71.0, "0.5", 25.0,
                       Modulation.SINUSOIDAL, Interpolation.LINEAR, TypeError,
                       "Argument speed with value 0.5 is not of type [<class 'float'>, <class 'int'>],"
                       " but got <class 'str'>.")
    test_invalid_input("invalid speed parameter value", 44100, 0.0, 2.0, 0.0, 71.0, 50, 25.0,
                       Modulation.SINUSOIDAL, Interpolation.LINEAR, ValueError,
                       "Input speed is not within the required interval of [0.1, 10].")

    test_invalid_input("invalid phase parameter type as a String", 44100, 0.0, 2.0, 0.0, 71.0, 0.5, "25.0",
                       Modulation.SINUSOIDAL, Interpolation.LINEAR, TypeError,
                       "Argument phase with value 25.0 is not of type [<class 'float'>, <class 'int'>],"
                       " but got <class 'str'>.")
    test_invalid_input("invalid phase parameter value", 44100, 0.0, 2.0, 0.0, 71.0, 0.5, 150.0,
                       Modulation.SINUSOIDAL, Interpolation.LINEAR, ValueError,
                       "Input phase is not within the required interval of [0, 100].")

    test_invalid_input("invalid modulation parameter value", 44100, 0.0, 2.0, 0.0, 71.0, 0.5, 25.0, "test",
                       Interpolation.LINEAR, TypeError,
                       "Argument modulation with value test is not of type [<enum 'Modulation'>], "
                       "but got <class 'str'>.")

    test_invalid_input("invalid modulation parameter value", 44100, 0.0, 2.0, 0.0, 71.0, 0.5, 25.0,
                       Modulation.SINUSOIDAL, "test", TypeError,
                       "Argument interpolation with value test is not of type [<enum 'Interpolation'>], "
                       "but got <class 'str'>.")


if __name__ == '__main__':
    test_flanger_eager_sinusoidal_linear_float64()
    test_flanger_eager_triangular_linear_float32()
    test_flanger_eager_triangular_linear_int()
    test_flanger_shape_221()
    test_flanger_shape_11211()
    test_flanger_pipeline()
    test_invalid_flanger_input()
