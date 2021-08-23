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
Testing AmplitudeToDB op in DE
"""
import numpy as np
import pytest

import mindspore.dataset as ds
import mindspore.dataset.audio.transforms as c_audio
from mindspore import log as logger
from mindspore.dataset.audio.utils import ScaleType

CHANNEL = 1
FREQ = 20
TIME = 15


def gen(shape):
    np.random.seed(0)
    data = np.random.random(shape)
    yield (np.array(data, dtype=np.float32),)


def count_unequal_element(data_expected, data_me, rtol, atol):
    """ Precision calculation func """
    assert data_expected.shape == data_me.shape
    total_count = len(data_expected.flatten())
    error = np.abs(data_expected - data_me)
    greater = np.greater(error, atol + np.abs(data_expected) * rtol)
    loss_count = np.count_nonzero(greater)
    assert (loss_count / total_count) < rtol, "\ndata_expected_std:{0}\ndata_me_error:{1}\nloss:{2}".format(
        data_expected[greater], data_me[greater], error[greater])


def allclose_nparray(data_expected, data_me, rtol, atol, equal_nan=True):
    """ Precision calculation formula  """
    if np.any(np.isnan(data_expected)):
        assert np.allclose(data_me, data_expected, rtol, atol, equal_nan=equal_nan)
    elif not np.allclose(data_me, data_expected, rtol, atol, equal_nan=equal_nan):
        count_unequal_element(data_expected, data_me, rtol, atol)


def test_func_amplitude_to_db_eager():
    """ mindspore eager mode normal testcase:amplitude_to_db op"""

    logger.info("check amplitude_to_db op output")
    ndarr_in = np.array([[[[-0.2197528, 0.3821656]]],
                         [[[0.57418776, 0.46741104]]],
                         [[[-0.20381108, -0.9303914]]],
                         [[[0.3693608, -0.2017813]]],
                         [[[-1.727381, -1.3708513]]],
                         [[[1.259975, 0.4981323]]],
                         [[[0.76986176, -0.5793846]]]]).astype(np.float32)
    # cal from benchmark
    out_expect = np.array([[[[-84.17748, -4.177484]]],
                           [[[-2.4094608, -3.3030105]]],
                           [[[-100., -100.]]],
                           [[[-4.325492, -84.32549]]],
                           [[[-100., -100.]]],
                           [[[1.0036192, -3.0265532]]],
                           [[[-1.1358725, -81.13587]]]]).astype(np.float32)

    amplitude_to_db_op = c_audio.AmplitudeToDB()
    out_mindspore = amplitude_to_db_op(ndarr_in)

    allclose_nparray(out_mindspore, out_expect, 0.0001, 0.0001)


def test_func_amplitude_to_db_pipeline():
    """ mindspore pipeline mode normal testcase:amplitude_to_db op"""

    logger.info("test AmplitudeToDB op with default value")
    generator = gen([CHANNEL, FREQ, TIME])

    data1 = ds.GeneratorDataset(source=generator, column_names=["multi_dimensional_data"])

    transforms = [c_audio.AmplitudeToDB()]
    data1 = data1.map(operations=transforms, input_columns=["multi_dimensional_data"])

    for item in data1.create_dict_iterator(num_epochs=1, output_numpy=True):
        out_put = item["multi_dimensional_data"]
    assert out_put.shape == (CHANNEL, FREQ, TIME)


def test_amplitude_to_db_invalid_input():
    def test_invalid_input(test_name, stype, ref_value, amin, top_db, error, error_msg):
        logger.info("Test AmplitudeToDB with bad input: {0}".format(test_name))
        with pytest.raises(error) as error_info:
            c_audio.AmplitudeToDB(stype=stype, ref_value=ref_value, amin=amin, top_db=top_db)
        assert error_msg in str(error_info.value)

    test_invalid_input("invalid stype parameter value", "test", 1.0, 1e-10, 80.0, TypeError,
                       "Argument stype with value test is not of type [<enum 'ScaleType'>], but got <class 'str'>.")
    test_invalid_input("invalid ref_value parameter value", ScaleType.POWER, -1.0, 1e-10, 80.0, ValueError,
                       "Input ref_value is not within the required interval of (0, 16777216]")
    test_invalid_input("invalid amin parameter value", ScaleType.POWER, 1.0, -1e-10, 80.0, ValueError,
                       "Input amin is not within the required interval of (0, 16777216]")
    test_invalid_input("invalid top_db parameter value", ScaleType.POWER, 1.0, 1e-10, -80.0, ValueError,
                       "Input top_db is not within the required interval of (0, 16777216]")

    test_invalid_input("invalid stype parameter value", True, 1.0, 1e-10, 80.0, TypeError,
                       "Argument stype with value True is not of type [<enum 'ScaleType'>], but got <class 'bool'>.")
    test_invalid_input("invalid ref_value parameter value", ScaleType.POWER, "value", 1e-10, 80.0, TypeError,
                       "Argument ref_value with value value is not of type [<class 'int'>, <class 'float'>], " +
                       "but got <class 'str'>")
    test_invalid_input("invalid amin parameter value", ScaleType.POWER, 1.0, "value", -80.0, TypeError,
                       "Argument amin with value value is not of type [<class 'int'>, <class 'float'>], " +
                       "but got <class 'str'>")
    test_invalid_input("invalid top_db parameter value", ScaleType.POWER, 1.0, 1e-10, "value", TypeError,
                       "Argument top_db with value value is not of type [<class 'int'>, <class 'float'>], " +
                       "but got <class 'str'>")


if __name__ == "__main__":
    test_func_amplitude_to_db_eager()
    test_func_amplitude_to_db_pipeline()
    test_amplitude_to_db_invalid_input()
