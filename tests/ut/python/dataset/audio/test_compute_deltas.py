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
"""Test ComputeDeltas."""

import numpy as np
import pytest

import mindspore.dataset as ds
from mindspore.dataset import audio
from mindspore.dataset.audio import BorderType
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
        assert np.allclose(data_me, data_expected, rtol, atol, equal_nan=equal_nan)
    elif not np.allclose(data_me, data_expected, rtol, atol, equal_nan=equal_nan):
        count_unequal_element(data_expected, data_me, rtol, atol)


def test_compute_deltas_eager():
    """
    Feature: Test the basic function in eager mode.
    Description: Mindspore eager mode normal testcase:compute_deltas.
    Expectation: Compile done without error.
    """
    ndarr_in = np.array([[[0.08746047, -0.33246294, 0.5240939, 0.6064913, -0.70366],
                          [1.1420338, 0.50532603, 0.73435473, -0.83435977, -1.0607501],
                          [-1.4298731, -0.86117035, -0.7773941, -0.60023546, 1.1807907],
                          [0.4973711, 0.5299286, 0.818514, 0.7559297, -0.3418539],
                          [-0.2824797, 0.30402678, 0.7848569, -0.4135576, 0.19522846],
                          [-0.11636204, -0.4780833, 1.2691815, 0.9824286, 0.029275],
                          [-1.2611166, -1.1957082, 0.26212585, 0.35354254, 0.3609486]]]).astype(np.float32)

    out_expect = np.array([[[0.0453, 0.1475, -0.0643, -0.1970, -0.3766],
                            [-0.1452, -0.4360, -0.5745, -0.4927, -0.3817],
                            [0.1874, 0.2312, 0.5482, 0.6042, 0.5697],
                            [0.0675, 0.0838, -0.1452, -0.2904, -0.3419],
                            [0.2721, 0.0805, 0.0238, -0.0807, -0.0570],
                            [0.2409, 0.3583, 0.1752, -0.0225, -0.3433],
                            [0.3112, 0.4753, 0.4793, 0.3212, 0.0205]]]).astype(np.float32)

    compute_deltas = audio.ComputeDeltas()
    out_mindspore = compute_deltas(ndarr_in)

    allclose_nparray(out_mindspore, out_expect, 0.0001, 0.0001)


def test_compute_deltas_pipeline():
    """
    Feature: Test the basic function in pipeline mode.
    Description: Mindspore pipeline mode normal testcase:compute_deltas.
    Expectation: Compile done without error.
    """
    generator = gen([CHANNEL, FREQ, TIME])

    data1 = ds.GeneratorDataset(
        source=generator, column_names=["multi_dimensional_data"]
    )

    transforms = [audio.ComputeDeltas()]
    data1 = data1.map(operations=transforms, input_columns=["multi_dimensional_data"])

    for item in data1.create_dict_iterator(num_epochs=1, output_numpy=True):
        out_put = item["multi_dimensional_data"]
    assert out_put.shape == (CHANNEL, FREQ, TIME)


def test_compute_deltas_invalid_input():
    """
    Feature: Test the validate function with invalid parameters.
    Description: Mindspore invalid parameters testcase:compute_deltas.
    Expectation: Compile done without error.
    """
    def test_invalid_input(win_length, pad_mode, error, error_msg):
        with pytest.raises(error) as error_info:
            audio.ComputeDeltas(win_length=win_length, pad_mode=pad_mode)
        assert error_msg in str(error_info.value)

    test_invalid_input("test", BorderType.EDGE, TypeError,
        "Argument win_length with value test is not of type [<class 'int'>], but got <class 'str'>.",
    )
    test_invalid_input(2, BorderType.EDGE, ValueError,
        "Input win_length is not within the required interval of [3, 2147483647]",
    )
    test_invalid_input(5, 2, TypeError,
        "Argument pad_mode with value 2 is not of type [<enum 'BorderType'>], but got <class 'int'>.",
    )


def test_compute_deltas_transform():
    """
    Feature: ComputeDeltas
    Description: Test ComputeDeltas with various valid input parameters and data types
    Expectation: The operation completes successfully
    """
    # test ComputeDeltas  is normal
    waveform = np.random.randn(20, 20, 10, 10).astype(np.float64)
    dataset = ds.NumpySlicesDataset(waveform, column_names=["column1"])
    allpass_biquad = audio.ComputeDeltas(3, BorderType.CONSTANT)
    dataset = dataset.map(operations=allpass_biquad)
    for _ in dataset.create_dict_iterator(output_numpy=True):
        pass

    # test ComputeDeltas  is normal
    waveform = np.random.randn(20, 20, 3).astype(np.float64)
    dataset = ds.NumpySlicesDataset(waveform, column_names=["column1"])
    allpass_biquad = audio.ComputeDeltas(20, BorderType.EDGE)
    dataset = dataset.map(operations=allpass_biquad)
    for data in dataset.create_dict_iterator(output_numpy=True):
        assert (np.abs(data["column1"]) <= 1).all()

    # mindspore eager mode acc testcase:deemph_biquad
    waveform = np.random.randn(100, 20).astype(np.float32)
    deemph_biquad = audio.ComputeDeltas(8, BorderType.REFLECT)
    deemph_biquad(waveform)

    # mindspore eager mode acc testcase:deemph_biquad
    waveform = np.random.randn(100, 100).astype(np.float64)
    deemph_biquad = audio.ComputeDeltas(50)
    deemph_biquad(waveform)

    # mindspore eager mode acc testcase:deemph_biquad
    waveform = np.random.randint(-10000, 10000, (10, 10, 10, 10))
    deemph_biquad = audio.ComputeDeltas(6, BorderType.SYMMETRIC)
    deemph_biquad(waveform)


def test_compute_deltas_param_check():
    """
    Feature: ComputeDeltas
    Description: Test ComputeDeltas with invalid input data types and parameters
    Expectation: Correct error types and messages are raised as expected
    """
    # mindspore eager mode acc testcase:deemph_biquad
    waveform = np.random.randn(100, )
    deemph_biquad = audio.ComputeDeltas()
    with pytest.raises(RuntimeError, match="the shape of input tensor does not match the requirement"
                                           " of operator. Expecting tensor in shape of <..., freq, "
                                           "time>. But got tensor with dimension 1."):
        deemph_biquad(waveform)

    # mindspore eager mode acc testcase:deemph_biquad
    waveform = list(np.random.randn(10, 10))
    deemph_biquad = audio.ComputeDeltas()
    with pytest.raises(TypeError, match="Input should be NumPy audio, got <class 'list'>."):
        deemph_biquad(waveform)

    # mindspore eager mode acc testcase:deemph_biquad
    waveform = 10
    deemph_biquad = audio.ComputeDeltas()
    with pytest.raises(TypeError, match="Input should be NumPy audio, got <class 'int'>."):
        deemph_biquad(waveform)

    # mindspore eager mode acc testcase:deemph_biquad
    waveform = np.array([["1", "2", "3"]])
    deemph_biquad = audio.ComputeDeltas()
    with pytest.raises(RuntimeError, match="the data type of input tensor does not match the requirement of operator. "
                                           "Expecting tensor in type of \\[int, float, double\\]. But got type string"):
        deemph_biquad(waveform)

    # mindspore eager mode acc testcase:deemph_biquad
    waveform = None
    deemph_biquad = audio.ComputeDeltas()
    with pytest.raises(TypeError, match="Input should be NumPy audio, got <class 'NoneType'>."):
        deemph_biquad(waveform)

    # mindspore eager mode acc testcase:deemph_biquad
    with pytest.raises(ValueError, match="Input win_length is not within the required interval of \\[3, 2147483647\\]"):
        audio.ComputeDeltas(2)

    # mindspore eager mode acc testcase:deemph_biquad
    with pytest.raises(TypeError, match="Argument win_length with value 3.0 is not of "
                                        "type \\[<class 'int'>\\], but got <class 'float'>."):
        audio.ComputeDeltas(3.0)

    # mindspore eager mode acc testcase:deemph_biquad
    with pytest.raises(TypeError, match="Argument win_length with value 5 is not of "
                                        "type \\[<class 'int'>\\], but got <class 'str'>."):
        audio.ComputeDeltas("5")

    # mindspore eager mode acc testcase:deemph_biquad
    with pytest.raises(TypeError, match="Argument win_length with value \\[6\\] is not of"
                                        " type \\[<class 'int'>\\], but got <class 'list'>."):
        audio.ComputeDeltas([6])

    # mindspore eager mode acc testcase:deemph_biquad
    with pytest.raises(ValueError, match="Input win_length is not within the required interval of \\[3, 2147483647\\]"):
        audio.ComputeDeltas(2147483648)

    # mindspore eager mode acc testcase:deemph_biquad
    with pytest.raises(TypeError, match="Argument pad_mode with value 10 is not of "
                                        "type \\[<enum 'BorderType'>\\], but got <class 'int'>."):
        audio.ComputeDeltas(5, 10)

    # mindspore eager mode acc testcase:deemph_biquad
    with pytest.raises(TypeError, match="Argument pad_mode with value BorderType.REFLECT is not"
                                        " of type \\[<enum 'BorderType'>\\], but got <class 'str'>."):
        audio.ComputeDeltas(5, "BorderType.REFLECT")

    # mindspore eager mode acc testcase:deemph_biquad
    with pytest.raises(TypeError, match="Argument pad_mode with value \\[<BorderType.REFLECT: 'reflect'>\\] is not"
                                        " of type \\[<enum 'BorderType'>\\], but got <class 'list'>."):
        audio.ComputeDeltas(5, [BorderType.REFLECT])


if __name__ == "__main__":
    test_compute_deltas_eager()
    test_compute_deltas_pipeline()
    test_compute_deltas_invalid_input()
    test_compute_deltas_transform()
    test_compute_deltas_param_check()
