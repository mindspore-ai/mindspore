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
"""Test Fade."""

import numpy as np
import pytest

import mindspore.dataset as ds
from mindspore.dataset import audio
from mindspore.dataset.audio import FadeShape


def test_fade_linear():
    """
    Feature: Fade
    Description: Test Fade when fade shape is linear
    Expectation: The output and the expected output is equal
    """
    waveform = [[[9.1553e-05, 6.1035e-05, 6.1035e-05, 6.1035e-05, 1.2207e-04, 1.2207e-04,
                  9.1553e-05, 9.1553e-05, 9.1553e-05, 9.1553e-05, 9.1553e-05, 6.1035e-05,
                  1.2207e-04, 1.2207e-04, 1.2207e-04, 9.1553e-05, 9.1553e-05, 9.1553e-05,
                  6.1035e-05, 9.1553e-05]]]
    dataset = ds.NumpySlicesDataset(
        data=waveform, column_names='audio', shuffle=False)
    transforms = [audio.Fade(
        fade_in_len=10, fade_out_len=5, fade_shape=FadeShape.LINEAR)]
    dataset = dataset.map(operations=transforms, input_columns=["audio"])

    for item in dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
        out_put = item["audio"]
    # The result of the reference operation
    expected_output = np.array([[0.0000000000000000000, 6.781666797905927e-06, 1.356333359581185e-05,
                                 2.034499993897043e-05, 5.425333438324742e-05, 6.781666888855398e-05,
                                 6.103533087298274e-05, 7.120789086911827e-05, 8.138045086525380e-05,
                                 9.155300358543172e-05, 9.155300358543172e-05, 6.103499981691129e-05,
                                 0.0001220699996338225, 0.0001220699996338225, 0.0001220699996338225,
                                 9.155300358543172e-05, 6.866475450806320e-05, 4.577650179271586e-05,
                                 1.525874995422782e-05, 0.0000000000000000000]], dtype=np.float32)
    assert np.mean(out_put - expected_output) < 0.0001


def test_fade_exponential():
    """
    Feature: Fade
    Description: Test Fade when fade shape is exponential
    Expectation: The output and the expected output is equal
    """
    waveform = [[[1, 2, 3, 4, 5, 6],
                 [5, 7, 3, 78, 8, 4]]]
    dataset = ds.NumpySlicesDataset(
        data=waveform, column_names='audio', shuffle=False)
    transforms = [audio.Fade(
        fade_in_len=5, fade_out_len=6, fade_shape=FadeShape.EXPONENTIAL)]
    dataset = dataset.map(operations=transforms, input_columns=["audio"])

    for item in dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
        out_put = item["audio"]
    # The result of the reference operation
    expected_output = np.array([[0.0000, 0.2071, 0.4823, 0.6657, 0.5743, 0.0000],
                                [0.0000, 0.7247, 0.4823, 12.9820, 0.9190, 0.0000]], dtype=np.float32)
    assert np.mean(out_put - expected_output) < 0.0001


def test_fade_logarithmic():
    """
    Feature: Fade
    Description: Test Fade when fade shape is logarithmic
    Expectation: The output and the expected output is equal
    """
    waveform = [[[0.03424072265625, 0.01476832226565, 0.04995727590625,
                  -0.0205993652375, -0.0356467868775, 0.01290893546875]]]
    dataset = ds.NumpySlicesDataset(
        data=waveform, column_names='audio', shuffle=False)
    transforms = [audio.Fade(
        fade_in_len=4, fade_out_len=2, fade_shape=FadeShape.LOGARITHMIC)]
    dataset = dataset.map(operations=transforms, input_columns=["audio"])

    for item in dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
        out_put = item["audio"]
    # The result of the reference operation
    expected_output = np.array([[0.0000e+00, 9.4048e-03, 4.4193e-02,
                                 -2.0599e-02, -3.5647e-02, 1.5389e-09]],
                               dtype=np.float32)
    assert np.mean(out_put - expected_output) < 0.0001


def test_fade_quarter_sine():
    """
    Feature: Fade
    Description: Test Fade when fade shape is quarter_sine
    Expectation: The output and the expected output is equal
    """
    waveform = np.array([[[1, 2, 3, 4, 5, 6],
                          [5, 7, 3, 78, 8, 4],
                          [1, 2, 3, 4, 5, 6]]], dtype=np.float64)
    dataset = ds.NumpySlicesDataset(
        data=waveform, column_names='audio', shuffle=False)
    transforms = [audio.Fade(
        fade_in_len=6, fade_out_len=6, fade_shape=FadeShape.QUARTER_SINE)]
    dataset = dataset.map(operations=transforms, input_columns=["audio"])

    for item in dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
        out_put = item["audio"]
    # The result of the reference operation
    expected_output = np.array([[0.0000, 0.5878, 1.4266, 1.9021, 1.4695, 0.0000],
                                [0.0000, 2.0572, 1.4266, 37.091, 2.3511, 0.0000],
                                [0.0000, 0.5878, 1.4266, 1.9021, 1.4695, 0.0000]], dtype=np.float64)
    assert np.mean(out_put - expected_output) < 0.0001


def test_fade_half_sine():
    """
    Feature: Fade
    Description: Test Fade when fade shape is half_sine
    Expectation: The output and the expected output is equal
    """
    waveform = [[[0.03424072265625, 0.013580322265625, -0.011871337890625,
                  -0.0205993652343, -0.01049804687500, 0.0129089355468750],
                 [0.04125976562500, 0.060577392578125, 0.0499572753906250,
                  0.01306152343750, -0.019683837890625, -0.018829345703125]]]
    dataset = ds.NumpySlicesDataset(
        data=waveform, column_names='audio', shuffle=False)
    transforms = [audio.Fade(
        fade_in_len=3, fade_out_len=3, fade_shape=FadeShape.HALF_SINE)]
    dataset = dataset.map(operations=transforms, input_columns=["audio"])

    for item in dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
        out_put = item["audio"]
    # The result of the reference operation
    expected_output = np.array([[0.0000, 0.0068, -0.0119, -0.0206, -0.0052, 0.0000],
                                [0.0000, 0.0303, 0.0500, 0.0131, -0.0098, -0.0000]], dtype=np.float32)
    assert np.mean(out_put - expected_output) < 0.0001


def test_fade_wrong_arguments():
    """
    Feature: Fade
    Description: Test Fade with invalid arguments
    Expectation: Correct error is thrown as expected
    """
    try:
        audio.Fade(-1, 0)
    except ValueError as e:
        assert "fade_in_len is not within the required interval of [0, 2147483647]" in str(e)
    try:
        audio.Fade(0, -1)
    except ValueError as e:
        assert "fade_out_len is not within the required interval of [0, 2147483647]" in str(e)
    try:
        audio.Fade(fade_shape='123')
    except TypeError as e:
        assert "is not of type [<enum 'FadeShape'>]" in str(e)


def test_fade_eager():
    """
    Feature: Fade
    Description: Test Fade in eager mode
    Expectation: The output and the expected output is equal
    """
    data = np.array([[9.1553e-05, 6.1035e-05, 6.1035e-05, 6.1035e-05, 1.2207e-04, 1.2207e-04,
                      9.1553e-05, 9.1553e-05, 9.1553e-05, 9.1553e-05, 9.1553e-05, 6.1035e-05,
                      1.2207e-04, 1.2207e-04, 1.2207e-04, 9.1553e-05, 9.1553e-05, 9.1553e-05,
                      6.1035e-05, 9.1553e-05]]).astype(np.float32)
    expected_output = np.array([0.0000000000000000000, 6.781666797905927e-06, 1.356333359581185e-05,
                                2.034499993897043e-05, 5.425333438324742e-05, 6.781666888855398e-05,
                                6.103533087298274e-05, 7.120789086911827e-05, 8.138045086525380e-05,
                                9.155300358543172e-05, 9.155300358543172e-05, 6.103499981691129e-05,
                                0.0001220699996338225, 0.0001220699996338225, 0.0001220699996338225,
                                9.155300358543172e-05, 6.866475450806320e-05, 4.577650179271586e-05,
                                1.525874995422782e-05, 0.0000000000000000000], dtype=np.float32)
    fade = audio.Fade(10, 5, fade_shape=FadeShape.LINEAR)
    out_put = fade(data)
    assert np.mean(out_put - expected_output) < 0.0001


def test_fade_transform():
    """
    Feature: Fade
    Description: Test Fade with various valid input parameters and data types
    Expectation: The operation completes successfully and output shapes are correct
    """
    # Test with float64 2D array, fade in length 10 and fade out length 33, LINEAR shape
    waveform = np.random.randn(4, 200)
    fade = audio.Fade(10, 33, FadeShape.LINEAR)
    out = fade(waveform)
    assert np.shape(out) == (4, 200)

    # Test with float64 3D array, fade in length 500, fade out length 0, HALF_SINE shape
    waveform = np.random.randn(4, 4, 1000)
    fade = audio.Fade(500, 0, FadeShape.HALF_SINE)
    out = fade(waveform)
    assert np.shape(out) == (4, 4, 1000)

    # Test with float64 4D array, fade in length 0, fade out length 10, LOGARITHMIC shape
    waveform = np.random.randn(4, 5, 4, 200)
    fade = audio.Fade(0, 10, FadeShape.LOGARITHMIC)
    out = fade(waveform)
    assert np.shape(out) == (4, 5, 4, 200)

    # Test with float32 2D array, fade in length 0, fade out length 200, LOGARITHMIC shape
    waveform = np.random.randn(4, 200).astype(np.float32)
    fade = audio.Fade(0, 200, FadeShape.LOGARITHMIC)
    out = fade(waveform)
    assert np.shape(out) == (4, 200)

    # Test with float64 2D array, fade in length 200, fade out length 0, QUARTER_SINE shape
    waveform = np.random.randn(4, 200).astype(np.float64)
    fade = audio.Fade(200, 0, FadeShape.QUARTER_SINE)
    out = fade(waveform)
    assert np.shape(out) == (4, 200)

    # Test with int16 2D array, fade in length 100, fade out length 200, HALF_SINE shape
    waveform = np.random.randn(4, 200).astype(np.int16)
    fade = audio.Fade(100, 200, FadeShape.HALF_SINE)
    out = fade(waveform)
    assert np.shape(out) == (4, 200)

    # Test with int32 2D array, fade in/out length both 100, EXPONENTIAL shape
    waveform = np.random.randn(4, 200).astype(np.int32)
    fade = audio.Fade(100, 100, FadeShape.EXPONENTIAL)
    out = fade(waveform)
    assert np.shape(out) == (4, 200)

    # Test with int64 2D array, fade in/out length both 200, LINEAR shape
    waveform = np.random.randn(4, 200).astype(np.int64)
    fade = audio.Fade(200, 200, FadeShape.LINEAR)
    out = fade(waveform)
    assert np.shape(out) == (4, 200)

    # Test with 1D input, fade in length 1, fade out length 1, LINEAR shape
    waveform = np.random.randn(2)
    fade = audio.Fade(1, 1, FadeShape.LINEAR)
    out = fade(waveform)
    assert np.shape(out) == (2,)


def test_fade_param_check():
    """
    Feature: Fade
    Description: Test Fade with invalid input data types and parameters
    Expectation: Correct error types and messages are raised as expected
    """
    # Test with empty input array, expect RuntimeError about no data
    waveform = np.random.randn(0)
    fade = audio.Fade(1, 1, FadeShape.QUARTER_SINE)
    with pytest.raises(RuntimeError, match="Input Tensor has no data."):
        fade(waveform)

    # Test when fade out length is greater than waveform length, expect RuntimeError
    waveform = np.array([[220, 200], [240, 200]])
    fade = audio.Fade(2, 3, FadeShape.LINEAR)
    with pytest.raises(RuntimeError, match=".*Fade:.*'fade_out_len' should be no "
                                           "greater than 'length of waveform'.*"):
        fade(waveform)

    # Test with list type as input instead of numpy array, expect TypeError
    waveform = np.random.randn(4, 200).tolist()
    fade = audio.Fade(200, 200, FadeShape.LINEAR)
    with pytest.raises(TypeError, match="Input should be NumPy audio, got <class 'list'>."):
        fade(waveform)

    # Test fade_in_len negative, expect ValueError
    with pytest.raises(ValueError) as error_info:
        audio.Fade(-3, 2, FadeShape.LINEAR)
    assert "Input fade_in_len is not within the required interval of [0, 2147483647]." in str(error_info.value)

    # Test fade_out_len negative, expect ValueError
    with pytest.raises(ValueError) as error_info:
        audio.Fade(2, -3, FadeShape.LINEAR)
    assert "Input fade_out_len is not within the required interval of [0, 2147483647]." in str(error_info.value)

    # Test fade_shape is a string, expect TypeError
    with pytest.raises(TypeError) as error_info:
        audio.Fade(2, 3, 'aaa')
    assert "Argument fade_shape with value aaa is not of type [<enum 'FadeShape'>], but got <class 'str'>." in str(
        error_info.value)

    # Test fade_in_len is float, expect TypeError
    with pytest.raises(TypeError) as error_info:
        audio.Fade(2.1, 3, FadeShape.LINEAR)
    assert "Argument fade_in_len with value 2.1 is not of type [<class 'int'>], but got <class 'float'>" in str(
        error_info.value)

    # Test fade_in_len is list, expect TypeError
    with pytest.raises(TypeError) as error_info:
        audio.Fade([2], 3, FadeShape.LINEAR)
    assert "Argument fade_in_len with value [2] is not of type [<class 'int'>], but got <class 'list'>" in str(
        error_info.value)

    # Test fade_in_len is str, expect TypeError
    with pytest.raises(TypeError) as error_info:
        audio.Fade("3", 3, FadeShape.LINEAR)
    assert "Argument fade_in_len with value 3 is not of type [<class 'int'>], but got <class 'str'>." in str(
        error_info.value)

    # Test fade_in_len is None, expect TypeError
    with pytest.raises(TypeError) as error_info:
        audio.Fade(None, 3, FadeShape.LINEAR)
    assert "Argument fade_in_len with value None is not of type [<class 'int'>], but got <class 'NoneType'>." in str(
        error_info.value)

    # Test fade_out_len is float, expect TypeError
    with pytest.raises(TypeError) as error_info:
        audio.Fade(2, 3.02, FadeShape.LINEAR)
    assert "Argument fade_out_len with value 3.02 is not of type [<class 'int'>], but got <class 'float'>" in str(
        error_info.value)

    # Test fade_out_len is list, expect TypeError
    with pytest.raises(TypeError) as error_info:
        audio.Fade(2, [3], FadeShape.LINEAR)
    assert "Argument fade_out_len with value [3] is not of type [<class 'int'>], but got <class 'list'>" in str(
        error_info.value)

    # Test fade_out_len is str, expect TypeError
    with pytest.raises(TypeError) as error_info:
        audio.Fade(2, "3", FadeShape.LINEAR)
    assert "Argument fade_out_len with value 3 is not of type [<class 'int'>], but got <class 'str'>." in str(
        error_info.value)

    # Test fade_out_len is None, expect TypeError
    with pytest.raises(TypeError) as error_info:
        audio.Fade(2, None, FadeShape.LINEAR)
    assert "Argument fade_out_len with value None is not of type [<class 'int'>], but got <class 'NoneType'>." in str(
        error_info.value)

    # Test fade_shape is int, expect TypeError
    with pytest.raises(TypeError) as error_info:
        audio.Fade(2, 3, 2)
    assert "Argument fade_shape with value 2 is not of type [<enum 'FadeShape'>], but got <class 'int'>." in str(
        error_info.value)

    # Test fade_shape is list, expect TypeError
    with pytest.raises(TypeError) as error_info:
        audio.Fade(2, 3, [FadeShape.LINEAR])
    assert "Argument fade_shape with value [<FadeShape.LINEAR: 'linear'>] is not of type [<enum 'FadeShape'>], but g" \
           "ot <class 'list'>." in str(error_info.value)

    # Test fade_shape is None, expect TypeError
    with pytest.raises(TypeError) as error_info:
        audio.Fade(2, 3, None)
    assert "Argument fade_shape with value None is not of type [<enum 'FadeShape'>], but got <class 'NoneType'>." in \
           str(error_info.value)


if __name__ == '__main__':
    test_fade_linear()
    test_fade_exponential()
    test_fade_logarithmic()
    test_fade_quarter_sine()
    test_fade_half_sine()
    test_fade_wrong_arguments()
    test_fade_eager()
    test_fade_transform()
    test_fade_param_check()
