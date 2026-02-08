# Copyright 2022-2025 Huawei Technologies Co., Ltd
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
"""Test PhaseVocoder."""

import numpy as np
import pytest

import mindspore.dataset as ds
from mindspore.dataset import audio
from . import count_unequal_element


def gen(shape):
    np.random.seed(0)
    data = np.random.random(shape)
    yield (np.array(data, dtype=np.float32),)


def allclose_nparray(data_expected, data_me, rtol, atol, equal_nan=True):
    if np.any(np.isnan(data_expected)):
        assert np.allclose(data_me, data_expected, rtol, atol, equal_nan=equal_nan)
    elif not np.allclose(data_me, data_expected, rtol, atol, equal_nan=equal_nan):
        count_unequal_element(data_expected, data_me, rtol, atol)


def test_phase_vocoder_compare():
    """
    Feature: PhaseVocoder
    Description: Mindspore eager mode checking precision
    Expectation: The returned result is as expected
    """
    data0 = np.array(
        [
            [
                [
                    [0.43189, 2.3049924],
                    [-0.01202229, 0.9176453],
                    [-0.6258611, 0.66475236],
                    [0.13541847, 1.2829605],
                    [0.9725325, 1.1669061],
                ],
                [
                    [-0.35001752, -1.0989336],
                    [-1.4930767, 0.86829656],
                    [0.3355314, -0.41216415],
                    [-1.1828239, 1.0075365],
                    [-0.19343425, 0.38364533],
                ],
            ]
        ]
    ).astype("float32")
    data1 = np.array(
        [
            [
                [
                    [0.43189, 2.3049924],
                    [-0.01202229, 0.9176453],
                    [-0.6258611, 0.66475236],
                    [0.13541847, 1.2829605],
                    [0.9725325, 1.1669061],
                ],
                [
                    [-0.35001752, -1.0989336],
                    [-1.4930767, 0.86829656],
                    [0.3355314, -0.41216415],
                    [-1.1828239, 1.0075365],
                    [-0.19343425, 0.38364533],
                ],
            ]
        ]
    ).astype("float64")
    rate = 2.0
    phase_advance0 = np.array([[0.0000], [3.9270]]).astype("float32")
    phase_vocoder0 = audio.PhaseVocoder(rate, phase_advance0)
    phase_advance1 = np.array([[0.0000], [3.9270]]).astype("float64")
    phase_vocoder1 = audio.PhaseVocoder(rate, phase_advance1)
    output0 = phase_vocoder0(data0)
    output1 = phase_vocoder1(data1)
    stand_outdata = np.array(
        [
            [
                [
                    [0.43189007, 2.3049924],
                    [-0.01196056, 0.9129374],
                    [1.1385509, 1.00558],
                ],
                [
                    [-0.35001755, -1.0989336],
                    [-0.4594292, 0.26718047],
                    [0.404371, -0.14520557],
                ],
            ]
        ]
    ).astype("float32")
    allclose_nparray(output0, stand_outdata, 0.0001, 0.0001)
    allclose_nparray(output1, stand_outdata, 0.0001, 0.0001)


def test_phase_vocoder_eager():
    """
    Feature: PhaseVocoder
    Description: Mindspore eager mode with normal testcase
    Expectation: The returned result is as expected
    """
    stft = next(gen([10, 10, 10, 2]))[0]
    output = audio.PhaseVocoder(1.3, np.random.randn(10, 1).astype("float32"))(stft)
    assert output.shape == (10, 10, 8, 2)


def test_phase_vocoder_pipeline():
    """
    Feature: PhaseVocoder
    Description: Mindspore pipeline mode with normal testcase
    Expectation: The returned result is as expected
    """
    generator = gen([32, 33, 333, 2])
    dataset = ds.GeneratorDataset(source=generator, column_names=["input"])

    transforms = [audio.PhaseVocoder(0.8, np.random.randn(33, 1).astype("float32"))]
    dataset = dataset.map(operations=transforms, input_columns=["input"])

    for item in dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
        output = item["input"]
    assert output.shape == (32, 33, 417, 2)


def test_phase_vocoder_invalid_input():
    """
    Feature: PhaseVocoder
    Description: Mindspore eager mode with invalid input
    Expectation: The returned result is as expected
    """

    def test_invalid_param(rate, phase_advance, error, error_msg):
        with pytest.raises(error) as error_info:
            audio.PhaseVocoder(rate, phase_advance)
        assert error_msg in str(error_info.value)

    def test_invalid_input(spec, rate, phase_advance, error, error_msg):
        with pytest.raises(error) as error_info:
            audio.PhaseVocoder(rate, phase_advance)(spec)
        assert error_msg in str(error_info.value)

    test_invalid_param(
        2, None, TypeError, "Argument phase_advance with value None is not of type"
    )
    test_invalid_param(
        0,
        np.random.randn(4, 1),
        ValueError,
        "Input rate is not within the required interval of (0, 16777216].",
    )
    spec = next(gen([1, 2, 2]))[0]
    test_invalid_input(
        spec,
        1.23,
        np.random.randn(4),
        RuntimeError,
        "PhaseVocoder: invalid parameter, 'phase_advance' should be in shape of <freq, 1>.",
    )
    test_invalid_input(
        spec,
        1.1,
        np.random.randn(4, 4, 1),
        RuntimeError,
        "PhaseVocoder: invalid parameter, 'phase_advance' should be in shape of <freq, 1>.",
    )
    test_invalid_input(
        spec,
        2,
        np.random.randn(3, 1),
        RuntimeError,
        "PhaseVocoder: invalid parameter, 'first dimension of 'phase_advance'' should be equal",
    )
    data = np.random.randn(4, 4, 2).astype("float32")
    input_phase_advance = np.random.randn(4, 1).astype("float64")
    test_invalid_input(
        data,
        2,
        input_phase_advance,
        RuntimeError,
        "PhaseVocoder: invalid parameter, data type of phase_advance should be equal to data",
    )


def test_phase_vocoder_transform():
    """
    Feature: PhaseVocoder
    Description: Test PhaseVocoder with various valid input parameters and data types
    Expectation: The operation completes successfully
    """

    complex_specgrams = np.random.randn(2, 200, 300, 2)
    phase_advance = np.random.randn(200, 1)
    phase_vocoder = audio.PhaseVocoder(2, phase_advance)
    output = phase_vocoder(complex_specgrams)
    assert output.shape == (2, 200, 150, 2)

    # eager mode
    stft = np.random.random([10, 10, 10, 2]).astype(np.float32)
    output = audio.PhaseVocoder(1.3, np.random.randn(10, 1).astype("float32"))(stft)
    assert output.shape == (10, 10, 8, 2)

    # pipeline
    dataset = ds.NumpySlicesDataset(
        np.random.random([10, 32, 33, 333, 2]).astype(np.float32),
        column_names=["input"],
    )

    transforms = [audio.PhaseVocoder(0.8, np.random.randn(33, 1).astype("float32"))]
    dataset = dataset.map(operations=transforms, input_columns=["input"])

    for item in dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
        output = item["input"]
        assert output.shape == (32, 33, 417, 2)

    # phase_advance
    complex_specgrams = np.random.randn(2, 2, 200, 200, 2)
    phase_advance = np.random.randn(200, 1)
    phase_vocoder = audio.PhaseVocoder(1.3, phase_advance)
    output = phase_vocoder(complex_specgrams)
    assert output.shape == (2, 2, 200, 154, 2)


def test_phase_vocoder_param_check():
    """
    Feature: PhaseVocoder
    Description: Test PhaseVocoder with invalid input data types and parameters
    Expectation: Correct error types and messages are raised as expected
    """

    data = np.random.randn(2, 200, 300, 2).astype("float64")
    phase_advance = np.random.randn(200, 1).astype("float32")
    phase_vocoder = audio.PhaseVocoder(2, phase_advance)
    with pytest.raises(RuntimeError, match=r"invalid parameter"):
        phase_vocoder(data)

    # phase advance (freq) != input tensor (freq)
    data = np.random.randn(2, 1025, 300, 2)
    phase_advance = np.random.randn(1024, 1)
    phase_vocoder = audio.PhaseVocoder(2, phase_advance)
    with pytest.raises(RuntimeError, match=r"invalid parameter"):
        phase_vocoder(data)

    # input tensor rank is less than 3
    data = np.random.randn(20, 30)
    phase_advance = np.random.randn(20, 1)
    phase_vocoder = audio.PhaseVocoder(2, phase_advance)
    with pytest.raises(
        RuntimeError,
        match=r"the shape of input tensor does not match the requirement of operator",
    ):
        phase_vocoder(data)

    # the shape of input tensor is <20, 30 ,1>
    data = np.random.randn(20, 30, 1)
    phase_advance = np.random.randn(20, 1)
    phase_vocoder = audio.PhaseVocoder(2, phase_advance)
    with pytest.raises(
        RuntimeError,
        match=r"the shape of input tensor does not match the requirement of operator",
    ):
        phase_vocoder(data)

    # the type of input tensor is str
    data = np.array(
        [
            [
                ["aaa", "b"],
                ["a", "a"],
            ],
            [
                ["a", "a"],
                ["a", "a"],
            ],
        ]
    )
    phase_advance = np.random.randn(2, 1)
    phase_vocoder = audio.PhaseVocoder(2, phase_advance)
    with pytest.raises(
        RuntimeError,
        match=r"the data type of input tensor does not match the requirement of operator",
    ):
        phase_vocoder(data)

    # Parameter exception
    with pytest.raises(TypeError, match="is not of type"):
        audio.PhaseVocoder(1.1, 32)

    # Parameter exception
    with pytest.raises(TypeError, match="is not of type"):
        audio.PhaseVocoder(1.1, [[1], [1]])

    # Parameter exception
    data = np.random.randn(20, 30, 2)
    phase_advance = np.random.randn(5, 2)
    with pytest.raises(RuntimeError, match=r"invalid parameter"):
        phase_vocoder = audio.PhaseVocoder(1.1, phase_advance)
        phase_vocoder(data)

    # Parameter exception
    data = np.random.randn(20, 30, 2)
    phase_advance = np.random.randn(5)
    with pytest.raises(RuntimeError, match=r"invalid parameter"):
        phase_vocoder = audio.PhaseVocoder(1.1, phase_advance)
        phase_vocoder(data)

    # Input phase_vocoder parameter exception
    data = np.random.randn(20, 30, 2)
    phase_advance = np.random.randn(1, 4, 2)
    with pytest.raises(RuntimeError, match=r"invalid parameter"):
        phase_vocoder = audio.PhaseVocoder(1.1, phase_advance)
        phase_vocoder(data)

    # rate
    phase_advance = np.random.randn(33, 1).astype("float32")
    with pytest.raises(TypeError, match=r"is not of type"):
        audio.PhaseVocoder(rate=[1], phase_advance=phase_advance)

    # rate
    phase_advance = np.random.randn(33, 1).astype("float32")
    with pytest.raises(TypeError, match=r"is not of type"):
        audio.PhaseVocoder(rate="1", phase_advance=phase_advance)

    # phase_advance
    with pytest.raises(TypeError, match=r"is not of type"):
        audio.PhaseVocoder(rate=1.0, phase_advance=1)

    # phase_advance
    with pytest.raises(TypeError, match=r"is not of type"):
        audio.PhaseVocoder(rate=1.0, phase_advance={})

    # phase_advance
    with pytest.raises(TypeError, match=r"is not of type"):
        audio.PhaseVocoder(rate=1.0, phase_advance=[1, 2, 3])

    # phase_advance
    with pytest.raises(TypeError, match=r"is not of type"):
        audio.PhaseVocoder(rate=1.0, phase_advance="1")


if __name__ == "__main__":
    test_phase_vocoder_compare()
    test_phase_vocoder_eager()
    test_phase_vocoder_pipeline()
    test_phase_vocoder_invalid_input()
    test_phase_vocoder_transform()
    test_phase_vocoder_param_check()
