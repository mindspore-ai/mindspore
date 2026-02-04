# Copyright 2021-2025 Huawei Technologies Co., Ltd
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
"""Test AmplitudeToDB."""

import numpy as np
import pytest

import mindspore.dataset as ds
from mindspore.dataset import audio
from mindspore.dataset.audio import ScaleType
from . import count_unequal_element

CHANNEL = 1
FREQ = 20
TIME = 15


def gen(shape):
    np.random.seed(0)
    data = np.random.random(shape)
    yield (np.array(data, dtype=np.float32),)


def allclose_nparray(data_expected, data_me, rtol, atol, equal_nan=True):
    """ Precision calculation formula  """
    if np.any(np.isnan(data_expected)):
        assert np.allclose(data_me, data_expected, rtol,
                           atol, equal_nan=equal_nan)
    elif not np.allclose(data_me, data_expected, rtol, atol, equal_nan=equal_nan):
        count_unequal_element(data_expected, data_me, rtol, atol)


def test_amplitude_to_db_eager():
    """
    Feature: AmplitudeToDB
    Description: Test AmplitudeToDB in eager mode with valid input
    Expectation: Output is equal to the expected output
    """
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

    amplitude_to_db = audio.AmplitudeToDB()
    out_mindspore = amplitude_to_db(ndarr_in)

    allclose_nparray(out_mindspore, out_expect, 0.0001, 0.0001)


def test_amplitude_to_db_pipeline():
    """
    Feature: AmplitudeToDB
    Description: Test AmplitudeToDB in pipeline mode with valid input
    Expectation: Output is equal to the expected output
    """
    generator = gen([CHANNEL, FREQ, TIME])

    data1 = ds.GeneratorDataset(source=generator, column_names=[
        "multi_dimensional_data"])

    transforms = [audio.AmplitudeToDB()]
    data1 = data1.map(operations=transforms, input_columns=[
        "multi_dimensional_data"])

    for item in data1.create_dict_iterator(num_epochs=1, output_numpy=True):
        out_put = item["multi_dimensional_data"]
    assert out_put.shape == (CHANNEL, FREQ, TIME)


def test_amplitude_to_db_invalid_input():
    """
    Feature: AmplitudeToDB
    Description: Test AmplitudeToDB with invalid input
    Expectation: Correct error and message are thrown as expected
    """
    def test_invalid_input(stype, ref_value, amin, top_db, error, error_msg):
        with pytest.raises(error) as error_info:
            audio.AmplitudeToDB(
                stype=stype, ref_value=ref_value, amin=amin, top_db=top_db)
        assert error_msg in str(error_info.value)

    test_invalid_input("test", 1.0, 1e-10, 80.0, TypeError,
                       "Argument stype with value test is not of type [<enum 'ScaleType'>], but got <class 'str'>.")
    test_invalid_input(ScaleType.POWER, -1.0, 1e-10, 80.0, ValueError,
                       "Input ref_value is not within the required interval of (0, 16777216]")
    test_invalid_input(ScaleType.POWER, 1.0, -1e-10, 80.0, ValueError,
                       "Input amin is not within the required interval of (0, 16777216]")
    test_invalid_input(ScaleType.POWER, 1.0, 1e-10, -80.0, ValueError,
                       "Input top_db is not within the required interval of (0, 16777216]")
    test_invalid_input(True, 1.0, 1e-10, 80.0, TypeError,
                       "Argument stype with value True is not of type [<enum 'ScaleType'>], but got <class 'bool'>.")
    test_invalid_input(ScaleType.POWER, "value", 1e-10, 80.0, TypeError,
                       "Argument ref_value with value value is not of type [<class 'int'>, <class 'float'>], " +
                       "but got <class 'str'>")
    test_invalid_input(ScaleType.POWER, 1.0, "value", -80.0, TypeError,
                       "Argument amin with value value is not of type [<class 'int'>, <class 'float'>], " +
                       "but got <class 'str'>")
    test_invalid_input(ScaleType.POWER, 1.0, 1e-10, "value", TypeError,
                       "Argument top_db with value value is not of type [<class 'int'>, <class 'float'>], " +
                       "but got <class 'str'>")


def test_amplitude_to_db_transform():
    """
    Feature: AmplitudeToDB
    Description: Test AmplitudeToDB with various valid input parameters and data types
    Expectation: The operation completes successfully
    """
    # test amplitude_to_db is normal
    waveform = np.random.randn(100, 32, 25, 13)
    dataset = ds.NumpySlicesDataset(waveform, column_names=["column1"])
    allpass_biquad = audio.AmplitudeToDB(stype=ScaleType.MAGNITUDE, ref_value=10.356, amin=116, top_db=23.001)
    dataset = dataset.map(operations=allpass_biquad)
    for _ in dataset.create_dict_iterator(output_numpy=True):
        pass

    # test amplitude_to_db is normal
    waveform = np.random.randn(64, 40).astype(np.float32)
    amplitude_to_db = audio.AmplitudeToDB(stype=ScaleType.POWER, ref_value=1.0, amin=1e-10, top_db=80.0)
    amplitude_to_db(waveform)

    # test amplitude_to_db is normal
    waveform = np.random.randint(-200, 200, (4, 40, 10, 8)).astype(np.int64)
    amplitude_to_db = audio.AmplitudeToDB(stype=ScaleType.POWER, ref_value=16777216, amin=0.0001, top_db=0.02)
    amplitude_to_db(waveform)

    # test amplitude_to_db is normal
    waveform = np.random.randn(64, 40, 5).astype(np.float64)
    amplitude_to_db = audio.AmplitudeToDB(stype=ScaleType.MAGNITUDE, ref_value=0.02, amin=16777216, top_db=203.65)
    amplitude_to_db(waveform)

    # test amplitude_to_db is normal
    waveform = np.random.randn(1, 1, 1, 1)
    amplitude_to_db = audio.AmplitudeToDB(stype=ScaleType.POWER, ref_value=1.0, amin=1e-10, top_db=16777216)
    amplitude_to_db(waveform)

    # test amplitude_to_db is normal
    waveform = np.random.randn(10, 1, 8, 6, 12)
    amplitude_to_db = audio.AmplitudeToDB(stype=ScaleType.MAGNITUDE, ref_value=16777216, amin=167, top_db=0.01)
    amplitude_to_db(waveform)

    # test amplitude_to_db is normal
    waveform = np.random.randn(1024, 1580).astype(np.float32)
    amplitude_to_db = audio.AmplitudeToDB(stype=ScaleType.POWER, ref_value=1211.0, amin=16777216, top_db=80.0)
    amplitude_to_db(waveform)


def test_amplitude_to_db_param_check():
    """
    Feature: AmplitudeToDB
    Description: Test AmplitudeToDB with invalid input data types and parameters
    Expectation: Correct error types and messages are raised as expected
    """
    # test amplitude_to_db is normal
    waveform = 10
    amplitude_to_db = audio.AmplitudeToDB()
    with pytest.raises(TypeError, match="Input should be NumPy audio, got <class 'int'>."):
        amplitude_to_db(waveform)

    # test tensor type is abnormal
    waveform = np.array([["1", "2"], ["3", "4"]])
    amplitude_to_db = audio.AmplitudeToDB()
    with pytest.raises(RuntimeError, match=".*AmplitudeToDB:.*Expecting tensor in type"
                                           " of .*int, float, double.*But got.*string.*"):
        amplitude_to_db(waveform)

    # test tensor is abnormal
    waveform = np.array(10)
    amplitude_to_db = audio.AmplitudeToDB()
    with pytest.raises(RuntimeError, match=".*AmplitudeToDB:.*Expecting tensor in shape of"
                                           " <..., freq, time>. But got tensor with dimension 0."):
        amplitude_to_db(waveform)

    # test Input type is abnormal
    waveform = [[1, 2, 3], [4, 5, 6]]
    amplitude_to_db = audio.AmplitudeToDB()
    with pytest.raises(TypeError, match="Input should be NumPy audio, got <class 'list'>."):
        amplitude_to_db(waveform)

    # test Input Tensor is not valid.
    amplitude_to_db = audio.AmplitudeToDB()
    with pytest.raises(RuntimeError, match="Input Tensor is not valid."):
        amplitude_to_db()

    # test Too many Input Tensor .
    waveform = np.array([1, 2, 3])
    amplitude_to_db = audio.AmplitudeToDB()
    with pytest.raises(RuntimeError, match="The op is OneToOne, can only accept one tensor as input."):
        amplitude_to_db(waveform, waveform)

    # test stype type is abnormal.
    with pytest.raises(TypeError, match="Argument stype with value ScaleType.POWER is not of "
                                        "type \\[<enum 'ScaleType'>\\], but got <class 'str'>."):
        audio.AmplitudeToDB("ScaleType.POWER")

    # test stype type is abnormal.
    with pytest.raises(TypeError, match="Argument stype with value \\[<ScaleType.POWER: 'power'>\\]"
                                        " is not of type \\[<enum 'ScaleType'>\\], but got <class 'list'>."):
        audio.AmplitudeToDB([ScaleType.POWER])

    # test stype type is abnormal.
    with pytest.raises(TypeError, match="Argument stype with value 1 is not of type \\["
                                        "<enum 'ScaleType'>\\], but got <class 'int'>."):
        audio.AmplitudeToDB(1)

    # test stype type is abnormal.
    with pytest.raises(TypeError, match="Argument stype with value None is not of type \\[<"
                                        "enum 'ScaleType'>\\], but got <class 'NoneType'>."):
        audio.AmplitudeToDB(None)

    # test ref_value is abnormal.
    with pytest.raises(ValueError, match="Input ref_value is not within the required interval of \\(0, 16777216\\]."):
        audio.AmplitudeToDB(ref_value=0)

    # test ref_value is abnormal.
    with pytest.raises(ValueError, match="Input ref_value is not within the required interval of \\(0, 16777216\\]."):
        audio.AmplitudeToDB(ref_value=16777216.1)

    # test ref_value type is abnormal.
    with pytest.raises(TypeError, match="Argument ref_value with value 1 is not of type \\["
                                        "<class 'int'>, <class 'float'>\\], but got <class 'str'>."):
        audio.AmplitudeToDB(ref_value="1")

    # test ref_value type is abnormal.
    with pytest.raises(TypeError, match="Argument ref_value with value \\[1.0\\] is not of type"
                                        " \\[<class 'int'>, <class 'float'>\\], but got <class 'list'>."):
        audio.AmplitudeToDB(ref_value=[1.0])

    # test ref_value type is abnormal.
    with pytest.raises(TypeError, match="Argument ref_value with value None is not of type \\[<class"
                                        " 'int'>, <class 'float'>\\], but got <class 'NoneType'>."):
        audio.AmplitudeToDB(ref_value=None)

    # test amin is abnormal.
    with pytest.raises(ValueError, match="Input amin is not within the required interval of \\(0, 16777216\\]."):
        audio.AmplitudeToDB(amin=0)

    # test amin is abnormal.
    with pytest.raises(ValueError, match="Input amin is not within the required interval of \\(0, 16777216\\]."):
        audio.AmplitudeToDB(amin=16777216.1)

    # test amin type is abnormal.
    with pytest.raises(TypeError, match="Argument amin with value 1 is not of type \\["
                                        "<class 'int'>, <class 'float'>\\], but got <class 'str'>."):
        audio.AmplitudeToDB(amin="1")

    # test amin type is abnormal.
    with pytest.raises(TypeError, match="Argument amin with value \\[1.0\\] is not of type"
                                        " \\[<class 'int'>, <class 'float'>\\], but got <class 'list'>."):
        audio.AmplitudeToDB(amin=[1.0])

    # test amin type is abnormal.
    with pytest.raises(TypeError, match="Argument amin with value None is not of type \\[<class"
                                        " 'int'>, <class 'float'>\\], but got <class 'NoneType'>."):
        audio.AmplitudeToDB(amin=None)


if __name__ == "__main__":
    test_amplitude_to_db_eager()
    test_amplitude_to_db_pipeline()
    test_amplitude_to_db_invalid_input()
    test_amplitude_to_db_transform()
    test_amplitude_to_db_param_check()
