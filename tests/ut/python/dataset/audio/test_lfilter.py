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
"""Test LFilter."""

import numpy as np
import pytest

import mindspore
import mindspore.dataset as ds
from mindspore.dataset import audio
from . import count_unequal_element


def test_lfilter_eager():
    """
    Feature: LFilter
    Description: Test LFilter in eager mode under normal test case
    Expectation: Output is equal to the expected output
    """
    # Original waveform
    waveform = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=np.float64)
    # Expect waveform
    expect_waveform = np.array([[0.25, 0.45, 0.425],
                                [1., 1., 0.35]], dtype=np.float64)
    a_coeffs = [0.2, 0.2, 0.3]
    b_coeffs = [0.5, 0.4, 0.2]
    lfilter = audio.LFilter(a_coeffs, b_coeffs, True)
    output = lfilter(waveform)
    count_unequal_element(expect_waveform, output, 0.0001, 0.0001)


def test_lfilter_pipeline():
    """
    Feature: LFilter
    Description: Test LFilter in pipeline mode under normal test case
    Expectation: Output is equal to the expected output
    """

    # Original waveform
    waveform = np.array([[0.1, 0.2, 0.3, 0.4], [0.4, 0.5, 0.6, 0.7]], dtype=np.float64)
    # Expect waveform
    expect_waveform = np.array([[0.4, 0.5, 0.6, 1.],
                                [1., 0.8, 0.9, 1.]], dtype=np.float64)
    data = (waveform, waveform.shape)
    a_coeffs = [0.1, 0.2, 0.3]
    b_coeffs = [0.4, 0.5, 0.6]
    dataset = ds.NumpySlicesDataset(data, ["channel", "sample"], shuffle=False)
    lfilter = audio.LFilter(a_coeffs, b_coeffs)
    # Filtered waveform by lfilter
    dataset = dataset.map(input_columns=["channel"], operations=lfilter, num_parallel_workers=8)
    i = 0
    for data in dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
        count_unequal_element(expect_waveform[i, :], data['channel'], 0.0001, 0.0001)
        i += 1


def test_invalid_input_all():
    """
    Feature: LFilter
    Description: Test LFilter with invalid input
    Expectation: Correct error is raised as expected
    """
    waveform = np.random.rand(2, 1000)

    def test_invalid_input(a_coeffs, b_coeffs, clamp, error, error_msg):
        with pytest.raises(error) as error_info:
            audio.LFilter(a_coeffs, b_coeffs, clamp)(waveform)
        assert error_msg in str(error_info.value)

    test_invalid_input(['0.1', '0.2', '0.3'],[0.1, 0.2, 0.3], True, TypeError,
                       "Argument a_coeffs[0] with value 0.1 is not of type [<class 'float'>, <class 'int'>], "
                       "but got <class 'str'>.")
    test_invalid_input([234322354352353453651, 0.2, 0.3], [0.1, 0.2, 0.3], True, ValueError,
                       "Input a_coeffs[0] is not within the required interval of [-16777216, 16777216].")
    test_invalid_input([0.1, 0.2, 0.3], [0.1, 0.2, 0.3], "True", TypeError,
                       "Argument clamp with value True is not of type [<class 'bool'>],"
                       " but got <class 'str'>.")


def test_lfilter_transform():
    """
    Feature: LFilter
    Description: Test LFilter with various valid input parameters and data types
    Expectation: The operation completes successfully and output values are within valid range
    """
    # test lfilter is normal
    waveform = np.random.randn(30, 10, 20).astype(np.float32)
    dataset = ds.NumpySlicesDataset(waveform, column_names=["column1"])
    lfilter = audio.LFilter(a_coeffs=[1.02, -1.9], b_coeffs=[0.99, 1.9])
    dataset = dataset.map(operations=lfilter)
    for data in dataset.create_dict_iterator(output_numpy=True):
        assert (data["column1"] <= 1).all()
        assert (data["column1"] >= -1).all()

    # mindspore eager mode acc testcase:lfilter
    waveform = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float16)
    audio.LFilter(a_coeffs=(1.0201494451137518, -1.9991880801438362, 10.6538523654),
                  b_coeffs=[0.999797020035959, -1.999594040071918, 12])(waveform)

    # mindspore eager mode acc testcase:lfilter
    waveform = np.random.randn(10, 10)
    output = audio.LFilter(a_coeffs=[2, 5], b_coeffs=(10, 20))(waveform)
    assert (output <= 1).all()
    assert (output >= -1).all()

    # mindspore eager mode acc testcase:lfilter
    waveform = np.random.randint(-10000, 10000, (10, 10, 3)).astype(np.double)
    audio.LFilter(a_coeffs=[6.5], b_coeffs=[10.3], clamp=False)(waveform)

    # mindspore eager mode acc testcase:lfilter
    waveform = np.random.randn(10, 10, 10, 10, 10, 10)
    output = audio.LFilter(a_coeffs=[6.5, 1.0, 2.8, -0.8, -2.6, 4], b_coeffs=[10.3, 3.5, 4.6, 0.5, -1.2, 0.1])(waveform)
    assert (output <= 1).all()
    assert (output >= -1).all()

    # mindspore eager mode acc testcase:lfilter
    waveform = np.random.randn(4, 3, 2, 1)
    output = audio.LFilter(a_coeffs=[16777216, -16777216], b_coeffs=[16777216, -16777216])(waveform)
    assert (output <= 1).all()
    assert (output >= -1).all()

    # mindspore eager mode acc testcase:lfilter
    waveform = np.random.randn(4, 3)
    audio.LFilter(a_coeffs=[0.02, -0.05], b_coeffs=[0, 0], clamp=False)(waveform)


def test_lfilter_param_check():
    """
    Feature: LFilter
    Description: Test LFilter with invalid input data types and parameters
    Expectation: Correct error types and messages are raised as expected
    """
    # mindspore eager mode acc testcase:lfilter
    waveform = list(np.random.randn(4, 3))
    with pytest.raises(TypeError, match="Input should be NumPy audio, got <class 'list'>."):
        audio.LFilter(a_coeffs=[0.2, 0.8], b_coeffs=[1.0, 1.6], clamp=False)(waveform)

    # mindspore eager mode acc testcase:lfilter
    waveform = np.array(10.0)
    with pytest.raises(RuntimeError, match=".*LFilter: the shape of input tensor does not match the "
                                           "requirement of operator. Expecting tensor in shape of "
                                           "<..., time>. But got tensor with dimension 0."):
        audio.LFilter(a_coeffs=[0.2, 0.8], b_coeffs=[1.0, 1.6], clamp=False)(waveform)

    # mindspore eager mode acc testcase:lfilter
    waveform = mindspore.Tensor(np.random.randn(4, 3))
    with pytest.raises(TypeError, match="Input should be NumPy audio, got <class 'mindspore.common.tensor.Tensor'>."):
        audio.LFilter(a_coeffs=[0.2, 0.8], b_coeffs=[1.0, 1.6], clamp=False)(waveform)

    # mindspore eager mode acc testcase:lfilter
    waveform = 1.0
    with pytest.raises(TypeError, match="Input should be NumPy audio, got <class 'float'>."):
        audio.LFilter(a_coeffs=[0.2, 0.8], b_coeffs=[1.0, 1.6], clamp=False)(waveform)

    # mindspore eager mode acc testcase:lfilter
    waveform = np.array([10, 11, 12])
    with pytest.raises(RuntimeError, match=".*LFilter: the data type of input tensor does not match"
                                           " the requirement of operator. Expecting tensor in type"
                                           " of.*float, double.*But got type int*."):
        audio.LFilter(a_coeffs=[0.2, 0.8], b_coeffs=[1.0, 1.6], clamp=False)(waveform)

    # mindspore eager mode acc testcase:lfilter
    with pytest.raises(ValueError, match="The size of a_coeffs should be the same as that of b_coeffs."):
        audio.LFilter(a_coeffs=[1.5], b_coeffs=[1.0, 1.6])

    # mindspore eager mode acc testcase:lfilter
    waveform = np.array([1.0, 1.1, 1.2])
    with pytest.raises(RuntimeError, match=".*lfilter: a_coeffs.*0.* can not be equal to zero, but got: 0.000000"):
        audio.LFilter(a_coeffs=[0, 0], b_coeffs=[1.0, 1.6])(waveform)

    # mindspore eager mode acc testcase:lfilter
    with pytest.raises(ValueError,
                       match="Input a_coeffs\\[1\\] is not within the required interval of \\[-16777216, 16777216\\]."):
        audio.LFilter(a_coeffs=[1.5, 16777216.1], b_coeffs=[1.0, 1.6])

    # mindspore eager mode acc testcase:lfilter
    with pytest.raises(ValueError,
                       match="Input a_coeffs\\[1\\] is not within the required interval of \\[-16777216, 16777216\\]."):
        audio.LFilter(a_coeffs=[1.5, -16777216.1], b_coeffs=[1.0, 1.6])

    # mindspore eager mode acc testcase:lfilter
    with pytest.raises(TypeError, match="Argument a_coeffs with value \\[1.5 1.1\\] is not of type \\[<class"
                                        " 'list'>, <class 'tuple'>\\], but got <class 'numpy.ndarray'>."):
        audio.LFilter(a_coeffs=np.array([1.5, 1.1]), b_coeffs=[1.0, 1.6])

    # mindspore eager mode acc testcase:lfilter
    with pytest.raises(TypeError, match="Argument a_coeffs\\[0\\] with value 1.5 is not of type"
                                        " \\[<class 'float'>, <class 'int'>\\], but got <class 'str'>."):
        audio.LFilter(a_coeffs=["1.5", "1.1"], b_coeffs=[1.0, 1.6])

    # mindspore eager mode acc testcase:lfilter
    with pytest.raises(TypeError, match="Argument a_coeffs with value 1.0 is not of type \\[<class"
                                        " 'list'>, <class 'tuple'>\\], but got <class 'float'>."):
        audio.LFilter(a_coeffs=1.0, b_coeffs=[1.0, 1.6])

    # mindspore eager mode acc testcase:lfilter
    with pytest.raises(TypeError, match="Argument a_coeffs with value None is not of type \\[<class"
                                        " 'list'>, <class 'tuple'>\\], but got <class 'NoneType'>."):
        audio.LFilter(a_coeffs=None, b_coeffs=[1.0, 1.6])

    # mindspore eager mode acc testcase:lfilter
    with pytest.raises(ValueError, match="Input b_coeffs\\[0\\] is not within the required"
                                         " interval of \\[-16777216, 16777216\\]."):
        audio.LFilter(a_coeffs=[6.1, 0.5], b_coeffs=[16777216.01, 1.6])

    # mindspore eager mode acc testcase:lfilter
    with pytest.raises(ValueError, match="Input b_coeffs\\[1\\] is not within the required"
                                         " interval of \\[-16777216, 16777216\\]."):
        audio.LFilter(a_coeffs=[-1.1, 0.5], b_coeffs=[1.0, -16777216.01])

    # mindspore eager mode acc testcase:lfilter
    with pytest.raises(TypeError, match="Argument b_coeffs with value \\[1.  1.6\\] is not of type \\[<class"
                                        " 'list'>, <class 'tuple'>\\], but got <class 'numpy.ndarray'>."):
        audio.LFilter(a_coeffs=[1.5, 1.1], b_coeffs=np.array([1.0, 1.6]))

    # mindspore eager mode acc testcase:lfilter
    with pytest.raises(TypeError, match="Argument b_coeffs\\[0\\] with value True is not of type"
                                        " \\(<class 'float'>, <class 'int'>\\), but got <class 'bool'>."):
        audio.LFilter(a_coeffs=[1.5, 1.1], b_coeffs=[True, False])

    # mindspore eager mode acc testcase:lfilter
    with pytest.raises(TypeError, match="Argument b_coeffs with value 1.0 is not of type \\[<class "
                                        "'list'>, <class 'tuple'>\\], but got <class 'float'>."):
        audio.LFilter(a_coeffs=[1.5, 1.1], b_coeffs=1.0)

    # mindspore eager mode acc testcase:lfilter
    with pytest.raises(TypeError, match="Argument b_coeffs with value None is not of type \\[<class"
                                        " 'list'>, <class 'tuple'>\\], but got <class 'NoneType'>."):
        audio.LFilter(a_coeffs=[1.5, 1.1], b_coeffs=None)

    # mindspore eager mode acc testcase:lfilter
    with pytest.raises(TypeError, match="Argument clamp with value 1.0 is not of type"
                                        " \\[<class 'bool'>\\], but got <class 'float'>."):
        audio.LFilter(a_coeffs=[1.5, 1.1], b_coeffs=[1.5, 1.1], clamp=1.0)

    # mindspore eager mode acc testcase:lfilter
    with pytest.raises(TypeError, match="Argument clamp with value True is not of "
                                        "type \\[<class 'bool'>\\], but got <class 'str'>."):
        audio.LFilter(a_coeffs=[1.5, 1.1], b_coeffs=[1.5, 1.1], clamp="True")

    # mindspore eager mode acc testcase:lfilter
    with pytest.raises(TypeError, match="Argument clamp with value \\[True\\] is not of"
                                        " type \\[<class 'bool'>\\], but got <class 'list'>."):
        audio.LFilter(a_coeffs=[1.5, 1.1], b_coeffs=[1.5, 1.1], clamp=[True])


if __name__ == '__main__':
    test_lfilter_eager()
    test_lfilter_pipeline()
    test_invalid_input_all()
    test_lfilter_transform()
    test_lfilter_param_check()
