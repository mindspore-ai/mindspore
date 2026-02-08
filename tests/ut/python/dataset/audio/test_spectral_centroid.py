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
"""Test SpectralCentroid."""

import numpy as np
import pytest

import mindspore.dataset as ds
from mindspore.dataset import audio
from . import count_unequal_element


def test_spectral_centroid_pipeline():
    """
    Feature: Mindspore pipeline mode normal testcase: spectral_centroid.
    Description: Input audio signal to test pipeline.
    Expectation: Success.
    """
    waveform = [[[1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 4, 4, 3, 3, 2, 2, 1, 1, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5]]]
    dataset = ds.NumpySlicesDataset(waveform, column_names=["audio"], shuffle=False)
    output = audio.SpectralCentroid(sample_rate=44100, n_fft=8)
    dataset = dataset.map(operations=output, input_columns=["audio"], output_columns=["SpectralCentroid"])
    result = np.array([[[4436.1182, 3580.0718, 2902.4917, 3334.8962, 5199.8350, 6284.4814,
                         3580.0718, 2895.5659]]])
    for data1 in dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
        count_unequal_element(data1["SpectralCentroid"], result, 0.0001, 0.0001)


def test_spectral_centroid_eager():
    """
    Feature: Mindspore eager mode normal testcase: spectral_centroid.
    Description: Input audio signal to test eager.
    Expectation: Success.
    """
    waveform = np.array([[1.2, 1, 2, 2, 3, 3, 4, 4, 5, 5, 4, 4, 3, 3, 2, 2, 1, 1, 0, 0,
                          1, 1, 2, 2, 3, 3, 4, 4, 5.5, 6.5]])
    spectral_centroid = audio.SpectralCentroid(sample_rate=48000, n_fft=8)
    output = spectral_centroid(waveform)
    result = np.array([[[5276.65022959, 3896.67543098, 3159.17400004, 3629.81957922,
                         5659.68456649, 6840.25126846, 3896.67543098, 3316.97434286]]])
    count_unequal_element(output, result, 0.0001, 0.0001)


def test_spectral_centroid_param():
    """
    Feature: Test spectral_centroid invalid parameter.
    Description: Test some invalid parameters.
    Expectation: Success.
    """
    try:
        audio.SpectralCentroid(sample_rate=-1)
    except ValueError as error:
        assert "Input sample_rate is not within the required interval of [0, 2147483647]." in str(error)
    try:
        audio.SpectralCentroid(sample_rate=48000, n_fft=-1)
    except ValueError as error:
        assert "Input n_fft is not within the required interval of [1, 2147483647]." in str(error)
    try:
        audio.SpectralCentroid(sample_rate=48000, n_fft=0)
    except ValueError as error:
        assert "Input n_fft is not within the required interval of [1, 2147483647]." in str(error)
    try:
        audio.SpectralCentroid(sample_rate=48000, win_length=-1)
    except ValueError as error:
        assert "Input win_length is not within the required interval of [1, 2147483647]." in str(error)
    try:
        audio.SpectralCentroid(sample_rate=48000, win_length="s")
    except TypeError as error:
        assert "Argument win_length with value s is not of type [<class 'int'>], but got <class 'str'>." in str(error)
    try:
        audio.SpectralCentroid(sample_rate=48000, hop_length=-1)
    except ValueError as error:
        assert "Input hop_length is not within the required interval of [1, 2147483647]." in str(error)
    try:
        audio.SpectralCentroid(sample_rate=48000, hop_length=-100)
    except ValueError as error:
        assert "Input hop_length is not within the required interval of [1, 2147483647]." in str(error)
    try:
        audio.SpectralCentroid(sample_rate=48000, win_length=300, n_fft=200)
    except ValueError as error:
        assert "Input win_length should be no more than n_fft, but got win_length: 300 and n_fft: 200." \
               in str(error)


def test_spectral_centroid_transform():
    """
    Feature: SpectralCentroid
    Description: Test SpectralCentroid with various valid input parameters and data types
    Expectation: The operation completes successfully
    """

    waveform = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    spectral_centroid = audio.SpectralCentroid(44100)
    spectral_centroid(waveform)

    # Test SpectralCentroid parameter completeness validation, WindowType
    waveform = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    spectral_centroid = audio.SpectralCentroid(sample_rate=44100, n_fft=100, pad=1, window=audio.WindowType.HAMMING)
    spectral_centroid(waveform)

    # Test SpectralCentroid parameter sample_rate maximum value validation
    waveform = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    spectral_centroid = audio.SpectralCentroid(sample_rate=2147483647, n_fft=200, pad=0,
                                                 window=audio.WindowType.KAISER)
    spectral_centroid(waveform)


def test_spectral_centroid_param_check():
    """
    Feature: SpectralCentroid
    Description: Test SpectralCentroid with invalid input data types and parameters
    Expectation: Correct error types and messages are raised as expected
    """

    waveform = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    with pytest.raises(ValueError, match="Input win_length should be no more than n_fft, "
                                         "but got win_length: 500 and n_fft: 400."):
        spectral_centroid = audio.SpectralCentroid(sample_rate=44100, n_fft=400, win_length=500)
        spectral_centroid(waveform)

    # Test error SpectralCentroid pipeline exception functionality validation
    waveform = np.array([[1., 1., 2., 2., 3., 3., 4., 4., 5., 5., 4., 4., 3., 3., 2., 2., 1., 1., 0., 0., 1., 1., 2.,
                          2., 3., 3., 4., 4., 5., 5.]])
    dataset1 = ds.NumpySlicesDataset(waveform, ["audio"], shuffle=False)
    dataset2 = ds.NumpySlicesDataset(waveform, ["audio"], shuffle=False)
    error_msg = "Argument n_fft with value 4.5 is not of type \\[<class 'int'>\\], but got <class 'float'>."
    with pytest.raises(TypeError) as error_info:
        spectral_centroid = audio.SpectralCentroid(sample_rate=44100, n_fft=4.5)
        assert error_msg in str(error_info.value)
        dataset2 = dataset2.map(input_columns=["audio"], operations=spectral_centroid)
        for _, _ in zip(dataset1.create_dict_iterator(output_numpy=True),
                        dataset2.create_dict_iterator(output_numpy=True)):
            pass

    # Test SpectralCentroid 06 eager
    spectral_centroid = audio.SpectralCentroid(44100)
    with pytest.raises(RuntimeError, match=".*SpectralCentroid: the shape of input tensor does not "
                                           "match the requirement of operator. Expecting tensor in "
                                           "shape of <..., time>. But got tensor with dimension 0.*"):
        spectral_centroid(False)

    # Test SpectralCentroid 07 eager
    waveform = np.array(["a", "b", "c"])
    with pytest.raises(RuntimeError, match="the data type of input tensor does not match the requirement of operator"):
        spectral_centroid = audio.SpectralCentroid(44100)
        spectral_centroid(waveform)

    # Test SpectralCentroid 08 eager
    waveform = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    with pytest.raises(ValueError, match="Input sample_rate is not within the required interval of \\[0, 2147483647\\]"
                                         "."):
        spectral_centroid = audio.SpectralCentroid(sample_rate=-1)
        spectral_centroid(waveform)

    # Test SpectralCentroid 010 eager
    waveform = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    with pytest.raises(TypeError, match="Argument sample_rate with value s is not of type \\[<class 'int'>\\], "
                                        "but got <class 'str'>."):
        spectral_centroid = audio.SpectralCentroid(sample_rate="s")
        spectral_centroid(waveform)

    # Test SpectralCentroid 011 eager
    waveform = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    error_msg = "Argument n_fft with value s is not of type \\[<class 'int'>\\], but got <class 'str'>."
    with pytest.raises(TypeError, match=error_msg):
        spectral_centroid = audio.SpectralCentroid(sample_rate=44100, n_fft="s")
        spectral_centroid(waveform)

    # Test SpectralCentroid 012 eager
    waveform = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    error_msg = "Input n_fft is not within the required interval of \\[1, 2147483647\\]."
    with pytest.raises(ValueError, match=error_msg):
        spectral_centroid = audio.SpectralCentroid(sample_rate=44100, n_fft=0)
        spectral_centroid(waveform)

    # Test eager SpectralCentroid win_length=-1=
    waveform = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    error_msg = "Input win_length is not within the required interval of \\[1, 2147483647]\\."
    with pytest.raises(ValueError, match=error_msg):
        spectral_centroid = audio.SpectralCentroid(sample_rate=44100, win_length=-1)
        spectral_centroid(waveform)

    # Test SpectralCentroid 014 eager
    waveform = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    error_msg = "Argument win_length with value s is not of type \\[<class 'int'>\\], but got <class 'str'>."
    with pytest.raises(TypeError, match=error_msg):
        spectral_centroid = audio.SpectralCentroid(sample_rate=44100, win_length="s")
        spectral_centroid(waveform)

    # Test SpectralCentroid 015 eager
    waveform = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    error_msg = "Input hop_length is not within the required interval of \\[1, 2147483647\\]."
    with pytest.raises(ValueError, match=error_msg):
        spectral_centroid = audio.SpectralCentroid(sample_rate=44100, hop_length=-1)
        spectral_centroid(waveform)

    # Test SpectralCentroid 016 eager
    waveform = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    error_msg = "Argument hop_length with value s is not of type \\[<class 'int'>\\], but got <class 'str'>."
    with pytest.raises(TypeError, match=error_msg):
        spectral_centroid = audio.SpectralCentroid(sample_rate=44100, hop_length="s")
        spectral_centroid(waveform)

    # Test SpectralCentroid 017 eager
    waveform = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    with pytest.raises(ValueError, match="Input pad is not within the required "
                                         "interval of \\[0, 2147483647\\]."):
        spectral_centroid = audio.SpectralCentroid(sample_rate=44100, pad=-1)
        spectral_centroid(waveform)

    # Test SpectralCentroid 018 eager
    waveform = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    error_msg = "Argument pad with value s is not of type \\[<class 'int'>\\], but got <class 'str'>."
    with pytest.raises(TypeError, match=error_msg):
        spectral_centroid = audio.SpectralCentroid(sample_rate=44100, pad="s")
        spectral_centroid(waveform)

    # Test SpectralCentroid 019 eager
    waveform = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    with pytest.raises(TypeError, match="Argument window with value 0 is not of type \\[<enum 'WindowType'>\\], "
                                        "but got <class 'int'>."):
        spectral_centroid = audio.SpectralCentroid(sample_rate=44100, window=0)
        spectral_centroid(waveform)


if __name__ == "__main__":
    test_spectral_centroid_pipeline()
    test_spectral_centroid_eager()
    test_spectral_centroid_param()
    test_spectral_centroid_transform()
    test_spectral_centroid_param_check()
