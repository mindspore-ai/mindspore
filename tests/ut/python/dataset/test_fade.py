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
"""
Testing fade op in DE
"""
import numpy as np

import mindspore.dataset as ds
import mindspore.dataset.audio as audio
from mindspore.dataset.audio.utils import FadeShape
from mindspore import log as logger


def test_fade_linear():
    """
    Feature: Fade
    Description: Test Fade when fade shape is linear
    Expectation: The output and the expected output is equal
    """
    logger.info("test fade, fade shape is 'linear'")

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
    logger.info("test fade, fade shape is 'exponential'")

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
    logger.info("test fade, fade shape is 'logarithmic'")

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
    logger.info("test fade, fade shape is 'quarter sine'")

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
    logger.info("test fade, fade shape is 'half sine'")

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
    logger.info("test fade with invalid arguments")
    try:
        _ = audio.Fade(-1, 0)
    except ValueError as e:
        logger.info("Got an exception in Fade: {}".format(str(e)))
        assert "fade_in_len is not within the required interval of [0, 2147483647]" in str(
            e)
    try:
        _ = audio.Fade(0, -1)
    except ValueError as e:
        logger.info("Got an exception in Fade: {}".format(str(e)))
        assert "fade_out_len is not within the required interval of [0, 2147483647]" in str(
            e)
    try:
        _ = audio.Fade(fade_shape='123')
    except TypeError as e:
        logger.info("Got an exception in Fade: {}".format(str(e)))
        assert "is not of type [<enum 'FadeShape'>]" in str(e)


def test_fade_eager():
    """
    Feature: Fade
    Description: Test Fade in eager mode
    Expectation: The output and the expected output is equal
    """
    logger.info("test fade eager")

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


if __name__ == '__main__':
    test_fade_linear()
    test_fade_exponential()
    test_fade_logarithmic()
    test_fade_quarter_sine()
    test_fade_half_sine()
    test_fade_wrong_arguments()
    test_fade_eager()
