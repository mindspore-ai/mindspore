# Copyright 2022-2025 Huawei Technologies Co., Ltd :
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
"""Test MFCC."""

import numpy as np
import pytest

import mindspore.dataset as ds
from mindspore.dataset import audio
from mindspore.dataset.audio import WindowType, BorderType, MelType, NormType, NormMode
from . import count_unequal_element


def test_mfcc_pipeline():
    """
    Feature: Mindspore pipeline mode normal testcase: mfcc
    Description: Input audio signal to test pipeline
    Expectation: Generate expected output after cases were executed
    """
    waveform = [
        [
            [
                1,
                1,
                2,
                2,
                3,
                3,
                4,
                4,
                5,
                5,
                4,
                4,
                3,
                3,
                2,
                2,
                1,
                1,
                0,
                0,
                1,
                1,
                2,
                2,
                3,
                3,
                4,
                4,
                5,
                5,
            ]
        ]
    ]
    dataset = ds.NumpySlicesDataset(waveform, column_names=["audio"], shuffle=False)
    output = audio.MFCC(
        sample_rate=16000,
        n_mfcc=4,
        dct_type=2,
        norm=NormMode.ORTHO,
        log_mels=True,
        melkwargs={
            "n_fft": 16,
            "win_length": 16,
            "hop_length": 8,
            "f_min": 0.0,
            "f_max": 10000.0,
            "pad": 0,
            "n_mels": 5,
            "window": WindowType.HANN,
            "power": 2.0,
            "normalized": False,
            "center": True,
            "pad_mode": BorderType.REFLECT,
            "onesided": True,
            "norm": NormType.NONE,
            "mel_scale": MelType.HTK,
        },
    )
    dataset = dataset.map(
        operations=output, input_columns=["audio"], output_columns=["MFCC"]
    )
    result = np.array(
        [
            [
                [2.7625, 5.6919, 3.6229, 3.9756],
                [0.8142, 3.2698, 1.4946, 3.0683],
                [-1.6855, -0.8312, -1.1395, 0.0481],
                [-2.1808, -2.5489, -2.3110, -3.1485],
            ]
        ]
    )
    for data1 in dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
        count_unequal_element(data1["MFCC"], result, 0.0001, 0.0001)


def test_mfcc_eager():
    """
    Feature: Mindspore eager mode normal testcase: mfcc
    Description: Input audio signal to test eager
    Expectation: Generate expected output after cases were executed
    """
    waveform = np.array(
        [
            [
                [
                    1,
                    1,
                    2,
                    2,
                    3,
                    3,
                    4,
                    4,
                    5,
                    5,
                    4,
                    4,
                    3,
                    3,
                    2,
                    2,
                    1,
                    1,
                    0,
                    0,
                    1,
                    1,
                    2,
                    2,
                    3,
                    3,
                    4,
                    4,
                    5,
                    5,
                ]
            ]
        ]
    )
    output = audio.MFCC(
        sample_rate=16000,
        n_mfcc=4,
        dct_type=2,
        norm=NormMode.ORTHO,
        log_mels=True,
        melkwargs={
            "n_fft": 16,
            "win_length": 16,
            "hop_length": 8,
            "f_min": 0.0,
            "f_max": 10000.0,
            "pad": 0,
            "n_mels": 5,
            "window": WindowType.HANN,
            "power": 2.0,
            "normalized": False,
            "center": True,
            "pad_mode": BorderType.REFLECT,
            "onesided": True,
            "norm": NormType.NONE,
            "mel_scale": MelType.HTK,
        },
    )(waveform)
    result = np.array(
        [
            [
                [
                    [2.7625, 5.6919, 3.6229, 3.9756],
                    [0.8142, 3.2698, 1.4946, 3.0683],
                    [-1.6855, -0.8312, -1.1395, 0.0481],
                    [-2.1808, -2.5489, -2.3110, -3.1485],
                ]
            ]
        ]
    )
    count_unequal_element(output, result, 0.0001, 0.0001)


def test_mfcc_param():
    """
    Feature: Test mfcc invalid parameter.
    Description: Test some invalid parameters.
    Expectation: throw ValueError, TypeError or RuntimeError exception.
    """
    try:
        audio.MFCC(sample_rate=-1)
    except ValueError as error:
        assert (
            "Input sample_rate is not within the required interval of [0, 2147483647]."
            in str(error)
        )
    try:
        audio.MFCC(log_mels=-1)
    except TypeError as error:
        assert (
            "Argument log_mels with value -1 is not of type [<class 'bool'>], but got <class 'int'>."
            in str(error)
        )
    try:
        audio.MFCC(norm="Karl Marx")
    except TypeError as error:
        assert (
            "Argument norm with value Karl Marx is not of type [<enum 'NormMode'>], but got <class 'str'>."
            in str(error)
        )
    try:
        audio.MFCC(dct_type=-1)
    except ValueError as error:
        assert "dct_type must be 2, but got : -1." in str(error)
    try:
        audio.MFCC(sample_rate=-1)
    except ValueError as error:
        assert (
            "Input sample_rate is not within the required interval of [0, 2147483647]."
            in str(error)
        )
    try:
        audio.MFCC(sample_rate="s")
    except TypeError as error:
        assert (
            "Argument sample_rate with value s is not of type [<class 'int'>], but got <class 'str'>."
            in str(error)
        )
    try:
        audio.MFCC(
            melkwargs={
                "n_fft": 16,
                "win_length": 16,
                "hop_length": 8,
                "f_min": 0.0,
                "f_max": -1,
                "pad": 0,
                "n_mels": 5,
                "window": WindowType.HANN,
                "power": 2.0,
                "normalized": True,
                "center": True,
                "pad_mode": BorderType.REFLECT,
                "onesided": True,
                "norm": NormType.NONE,
                "mel_scale": MelType.HTK,
            }
        )
    except ValueError as error:
        assert (
            "Input f_max is not within the required interval of (0, 16777216]."
            in str(error)
        )
    try:
        audio.MFCC(
            melkwargs={
                "n_fft": 16,
                "win_length": 16,
                "hop_length": 8,
                "f_min": -1,
                "f_max": 10000.0,
                "pad": 0,
                "n_mels": 5,
                "window": WindowType.HANN,
                "power": 2.0,
                "normalized": True,
                "center": True,
                "pad_mode": BorderType.REFLECT,
                "onesided": True,
                "norm": NormType.NONE,
                "mel_scale": MelType.HTK,
            }
        )
    except ValueError as error:
        assert (
            "Input n_mels should be greater than or equal to n_mfcc, but got n_mfcc: 40 and n_mels: 5."
            in str(error)
        )
    try:
        audio.MFCC(
            melkwargs={
                "n_fft": 16,
                "win_length": 16,
                "hop_length": 8,
                "f_min": 0.0,
                "f_max": 10000.0,
                "pad": 0,
                "n_mels": 5,
                "window": WindowType.HANN,
                "power": 2.0,
                "normalized": True,
                "center": True,
                "pad_mode": BorderType.REFLECT,
                "onesided": True,
                "norm": -1,
                "mel_scale": MelType.HTK,
            }
        )
    except TypeError as error:
        assert (
            "Argument norm with value -1 is not of type [<enum 'NormType'>], but got <class 'int'>."
            in str(error)
        )
    try:
        audio.MFCC(
            melkwargs={
                "n_fft": 16,
                "win_length": 16,
                "hop_length": 8,
                "f_min": 0.0,
                "f_max": 10000.0,
                "pad": 0,
                "n_mels": 5,
                "window": WindowType.HANN,
                "power": 2.0,
                "normalized": True,
                "center": True,
                "pad_mode": BorderType.REFLECT,
                "onesided": True,
                "norm": NormType.NONE,
                "mel_scale": -1,
            }
        )
    except TypeError as error:
        assert (
            "Argument mel_type with value -1 is not of type [<enum 'MelType'>], but got <class 'int'>."
            in str(error)
        )
    try:
        audio.MFCC(
            melkwargs={
                "n_fft": -1,
                "win_length": 16,
                "hop_length": 8,
                "f_min": 0.0,
                "f_max": 10000.0,
                "pad": 0,
                "n_mels": 5,
                "window": WindowType.HANN,
                "power": 2.0,
                "normalized": True,
                "center": True,
                "pad_mode": BorderType.REFLECT,
                "onesided": True,
                "norm": NormType.NONE,
                "mel_scale": MelType.HTK,
            }
        )
    except ValueError as error:
        assert (
            "Input n_fft is not within the required interval of [1, 2147483647]."
            in str(error)
        )
    try:
        audio.MFCC(
            melkwargs={
                "n_fft": 0,
                "win_length": 16,
                "hop_length": 8,
                "f_min": 0.0,
                "f_max": 10000.0,
                "pad": 0,
                "n_mels": 5,
                "window": WindowType.HANN,
                "power": 2.0,
                "normalized": True,
                "center": True,
                "pad_mode": BorderType.REFLECT,
                "onesided": True,
                "norm": NormType.NONE,
                "mel_scale": MelType.HTK,
            }
        )
    except ValueError as error:
        assert (
            "Input n_fft is not within the required interval of [1, 2147483647]."
            in str(error)
        )
    try:
        audio.MFCC(
            melkwargs={
                "n_fft": 16,
                "win_length": 0,
                "hop_length": 8,
                "f_min": 0.0,
                "f_max": 10000.0,
                "pad": 0,
                "n_mels": 50,
                "window": WindowType.HANN,
                "power": 2.0,
                "normalized": True,
                "center": True,
                "pad_mode": BorderType.REFLECT,
                "onesided": True,
                "norm": NormType.NONE,
                "mel_scale": MelType.HTK,
            }
        )
    except ValueError as error:
        assert (
            "Input win_length is not within the required interval of [0, 2147483647]."
            in str(error)
        )
    try:
        audio.MFCC(
            melkwargs={
                "n_fft": 16,
                "win_length": "s",
                "hop_length": 8,
                "f_min": 0.0,
                "f_max": 10000.0,
                "pad": 0,
                "n_mels": 5,
                "window": WindowType.HANN,
                "power": 2.0,
                "normalized": True,
                "center": True,
                "pad_mode": BorderType.REFLECT,
                "onesided": True,
                "norm": NormType.NONE,
                "mel_scale": MelType.HTK,
            }
        )
    except TypeError as error:
        assert (
            "Argument win_length with value s is not of type [<class 'int'>], but got <class 'str'>."
            in str(error)
        )
    try:
        audio.MFCC(
            melkwargs={
                "n_fft": 16,
                "win_length": 16,
                "hop_length": -1,
                "f_min": 0.0,
                "f_max": 10000.0,
                "pad": 0,
                "n_mels": 5,
                "window": WindowType.HANN,
                "power": 2.0,
                "normalized": True,
                "center": True,
                "pad_mode": BorderType.REFLECT,
                "onesided": True,
                "norm": NormType.NONE,
                "mel_scale": MelType.HTK,
            }
        )
    except ValueError as error:
        assert (
            "Input hop_length is not within the required interval of [1, 2147483647]."
            in str(error)
        )
    try:
        audio.MFCC(
            melkwargs={
                "n_fft": 200,
                "win_length": 300,
                "hop_length": 8,
                "f_min": 0.0,
                "f_max": 10000.0,
                "pad": 0,
                "n_mels": 50,
                "window": WindowType.HANN,
                "power": 2.0,
                "normalized": True,
                "center": True,
                "pad_mode": BorderType.REFLECT,
                "onesided": True,
                "norm": NormType.NONE,
                "mel_scale": MelType.HTK,
            }
        )
    except ValueError as error:
        assert (
            "Input win_length should be no more than n_fft, but got win_length: 300 and n_fft: 200."
            in str(error)
        )
    try:
        audio.MFCC(
            melkwargs={
                "n_fft": 16,
                "win_length": 16,
                "hop_length": 8,
                "f_min": 0.0,
                "f_max": 10000.0,
                "pad": -1,
                "n_mels": 5,
                "window": WindowType.HANN,
                "power": 2.0,
                "normalized": True,
                "center": True,
                "pad_mode": BorderType.REFLECT,
                "onesided": True,
                "norm": NormType.NONE,
                "mel_scale": MelType.HTK,
            }
        )
    except ValueError as error:
        assert (
            "Input pad is not within the required interval of [0, 2147483647]."
            in str(error)
        )
    try:
        audio.MFCC(
            melkwargs={
                "n_fft": 16,
                "win_length": 16,
                "hop_length": 8,
                "f_min": 0.0,
                "f_max": 10000.0,
                "pad": 0,
                "n_mels": 5,
                "window": WindowType.HANN,
                "power": -1,
                "normalized": True,
                "center": True,
                "pad_mode": BorderType.REFLECT,
                "onesided": True,
                "norm": NormType.NONE,
                "mel_scale": MelType.HTK,
            }
        )
    except ValueError as error:
        assert (
            "Input power is not within the required interval of [0, 16777216]."
            in str(error)
        )
    try:
        audio.MFCC(
            melkwargs={
                "n_fft": "XiaDanni",
                "win_length": 16,
                "hop_length": 8,
                "f_min": 0.0,
                "f_max": 10000.0,
                "pad": 0,
                "n_mels": 5,
                "window": WindowType.HANN,
                "power": 2.0,
                "normalized": True,
                "center": True,
                "pad_mode": BorderType.REFLECT,
                "onesided": True,
                "norm": NormType.NONE,
                "mel_scale": MelType.HTK,
            }
        )
    except TypeError as error:
        assert (
            "Argument n_fft with value XiaDanni is not of type [<class 'int'>], but got <class 'str'>."
            in str(error)
        )
    try:
        audio.MFCC(
            melkwargs={
                "n_fft": 16,
                "win_length": 16,
                "hop_length": 8,
                "f_min": 0.0,
                "f_max": 10000.0,
                "pad": 0,
                "n_mels": 5,
                "window": False,
                "power": 2.0,
                "normalized": True,
                "center": True,
                "pad_mode": BorderType.REFLECT,
                "onesided": True,
                "norm": NormType.NONE,
                "mel_scale": MelType.HTK,
            }
        )
    except TypeError as error:
        assert (
            "Argument window with value False is not of type [<enum 'WindowType'>], but got <class 'bool'>."
            in str(error)
        )
    try:
        audio.MFCC(
            melkwargs={
                "n_fft": 16,
                "win_length": 16,
                "hop_length": 8,
                "f_min": 0.0,
                "f_max": 10000.0,
                "pad": 0,
                "n_mels": 5,
                "window": WindowType.HANN,
                "power": 2.0,
                "normalized": True,
                "center": True,
                "pad_mode": False,
                "onesided": True,
                "norm": NormType.NONE,
                "mel_scale": MelType.HTK,
            }
        )
    except TypeError as error:
        assert (
            "Argument pad_mode with value False is not of type [<enum 'BorderType'>], but got <class 'bool'>."
            in str(error)
        )
    try:
        audio.MFCC(
            melkwargs={
                "n_fft": 16,
                "win_length": 16,
                "hop_length": 8,
                "f_min": 0.0,
                "f_max": 10000.0,
                "pad": 0,
                "n_mels": 5,
                "window": WindowType.HANN,
                "power": 2.0,
                "normalized": True,
                "center": True,
                "pad_mode": BorderType.REFLECT,
                "onesided": "LianLinghang",
                "norm": NormType.NONE,
                "mel_scale": MelType.HTK,
            }
        )
    except TypeError as error:
        assert (
            "Argument onesided with value LianLinghang is not of type [<class 'bool'>], but got <class 'str'>."
            in str(error)
        )
    try:
        audio.MFCC(
            melkwargs={
                "n_fft": 16,
                "win_length": 16,
                "hop_length": 8,
                "f_min": 0.0,
                "f_max": 10000.0,
                "pad": 0,
                "n_mels": 5,
                "window": WindowType.HANN,
                "power": 2.0,
                "normalized": True,
                "center": "XiaDanni",
                "pad_mode": BorderType.REFLECT,
                "onesided": False,
                "norm": NormType.NONE,
                "mel_scale": MelType.HTK,
            }
        )
    except TypeError as error:
        assert (
            "Argument center with value XiaDanni is not of type [<class 'bool'>], but got <class 'str'>."
            in str(error)
        )
    try:
        audio.MFCC(
            melkwargs={
                "n_fft": 16,
                "win_length": 16,
                "hop_length": 8,
                "f_min": 0.0,
                "f_max": 10000.0,
                "pad": 0,
                "n_mels": 5,
                "window": WindowType.HANN,
                "power": 2.0,
                "normalized": "s",
                "center": True,
                "pad_mode": BorderType.REFLECT,
                "onesided": False,
                "norm": NormType.NONE,
                "mel_scale": MelType.HTK,
            }
        )
    except TypeError as error:
        assert (
            "Argument normalized with value s is not of type [<class 'bool'>], but got <class 'str'>."
            in str(error)
        )
    try:
        audio.MFCC(
            melkwargs={
                "n_fft": 16,
                "win_length": 16,
                "hop_length": 8,
                "f_min": 0.0,
                "f_max": 10000.0,
                "pad": 0,
                "n_mels": 5,
                "window": WindowType.HANN,
                "power": 2.0,
                "normalized": 1,
                "center": True,
                "pad_mode": BorderType.REFLECT,
                "onesided": "LianLinghang",
                "norm": NormType.NONE,
                "mel_scale": MelType.HTK,
            }
        )
    except TypeError as error:
        assert (
            "Argument normalized with value 1 is not of type [<class 'bool'>], but got <class 'int'>."
            in str(error)
        )


def test_mfcc_transform():
    """
    Feature: MfccOps
    Description: Test MfccOps with various valid input parameters and data types
    Expectation: The operation completes successfully
    """

    waveform = [
        [
            [
                1,
                1,
                2,
                2,
                3,
                3,
                4,
                4,
                5,
                5,
                4,
                4,
                3,
                3,
                2,
                2,
                1,
                1,
                0,
                0,
                1,
                1,
                2,
                2,
                3,
                3,
                4,
                4,
                5,
                5,
            ]
        ]
    ]
    dataset = ds.NumpySlicesDataset(waveform, column_names=["audio"], shuffle=False)
    output = audio.MFCC(
        sample_rate=16000,
        n_mfcc=4,
        dct_type=2,
        norm=NormMode.ORTHO,
        log_mels=True,
        melkwargs={
            "n_fft": 16,
            "win_length": 16,
            "hop_length": 8,
            "f_min": 0.0,
            "f_max": 10000.0,
            "pad": 0,
            "n_mels": 5,
            "window": WindowType.HANN,
            "power": 2.0,
            "normalized": False,
            "center": True,
            "pad_mode": BorderType.REFLECT,
            "onesided": True,
            "norm": NormType.NONE,
            "mel_scale": MelType.HTK,
        },
    )
    dataset = dataset.map(
        operations=output, input_columns=["audio"], output_columns=["MFCC"]
    )
    dataset = dataset.project(columns=["MFCC"])
    result = np.array(
        [
            [
                [2.7625, 5.6919, 3.6229, 3.9756],
                [0.8142, 3.2698, 1.4946, 3.0683],
                [-1.6855, -0.8312, -1.1395, 0.0481],
                [-2.1808, -2.5489, -2.3110, -3.1485],
            ]
        ]
    )
    for data1 in dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
        count_unequal_element(data1["MFCC"], result, 0.0001, 0.0001)

    # Test with 3D input in eager mode
    waveform = np.array(
        [
            [
                [
                    1,
                    1,
                    2,
                    2,
                    3,
                    3,
                    4,
                    4,
                    5,
                    5,
                    4,
                    4,
                    3,
                    3,
                    2,
                    2,
                    1,
                    1,
                    0,
                    0,
                    1,
                    1,
                    2,
                    2,
                    3,
                    3,
                    4,
                    4,
                    5,
                    5,
                ]
            ]
        ]
    )
    output = audio.MFCC(
        sample_rate=16000,
        n_mfcc=4,
        dct_type=2,
        norm=NormMode.ORTHO,
        log_mels=True,
        melkwargs={
            "n_fft": 16,
            "win_length": 16,
            "hop_length": 8,
            "f_min": 0.0,
            "f_max": 10000.0,
            "pad": 0,
            "n_mels": 5,
            "window": WindowType.HANN,
            "power": 2.0,
            "normalized": False,
            "center": True,
            "pad_mode": BorderType.REFLECT,
            "onesided": True,
            "norm": NormType.NONE,
            "mel_scale": MelType.HTK,
        },
    )(waveform)
    result = np.array(
        [
            [
                [
                    [2.7625, 5.6919, 3.6229, 3.9756],
                    [0.8142, 3.2698, 1.4946, 3.0683],
                    [-1.6855, -0.8312, -1.1395, 0.0481],
                    [-2.1808, -2.5489, -2.3110, -3.1485],
                ]
            ]
        ]
    )
    count_unequal_element(output, result, 0.0001, 0.0001)

    # Test with 1D input array
    waveform = np.array(
        [
            1,
            1,
            2,
            2,
            3,
            3,
            4,
            4,
            5,
            5,
            4,
            4,
            3,
            3,
            2,
            2,
            1,
            1,
            0,
            0,
            1,
            1,
            2,
            2,
            3,
            3,
            4,
            4,
            5,
            5,
        ]
    )
    output = audio.MFCC(
        sample_rate=16000,
        n_mfcc=4,
        dct_type=2,
        norm=NormMode.ORTHO,
        log_mels=True,
        melkwargs={
            "n_fft": 16,
            "win_length": 16,
            "hop_length": 8,
            "f_min": 0.0,
            "f_max": 10000.0,
            "pad": 0,
            "n_mels": 5,
            "window": WindowType.HANN,
            "power": 2.0,
            "normalized": False,
            "center": True,
            "pad_mode": BorderType.REFLECT,
            "onesided": True,
            "norm": NormType.NONE,
            "mel_scale": MelType.HTK,
        },
    )(waveform)
    result = np.array(
        [
            [2.7625, 5.6919, 3.6229, 3.9756],
            [0.8142, 3.2698, 1.4946, 3.0683],
            [-1.6855, -0.8312, -1.1395, 0.0481],
            [-2.1808, -2.5489, -2.3110, -3.1485],
        ]
    )
    count_unequal_element(output, result, 0.0001, 0.0001)

    # Test with 2D input array
    waveform = np.array(
        [
            [
                1,
                1,
                2,
                2,
                3,
                3,
                4,
                4,
                5,
                5,
                4,
                4,
                3,
                3,
                2,
                2,
                1,
                1,
                0,
                0,
                1,
                1,
                2,
                2,
                3,
                3,
                4,
                4,
                5,
                5,
            ]
        ]
    )
    output = audio.MFCC(
        sample_rate=16000,
        n_mfcc=4,
        dct_type=2,
        norm=NormMode.ORTHO,
        log_mels=True,
        melkwargs={
            "n_fft": 16,
            "win_length": 16,
            "hop_length": 8,
            "f_min": 0.0,
            "f_max": 10000.0,
            "pad": 0,
            "n_mels": 5,
            "window": WindowType.HANN,
            "power": 2.0,
            "normalized": False,
            "center": True,
            "pad_mode": BorderType.REFLECT,
            "onesided": True,
            "norm": NormType.NONE,
            "mel_scale": MelType.HTK,
        },
    )(waveform)
    result = np.array(
        [
            [
                [2.7625, 5.6919, 3.6229, 3.9756],
                [0.8142, 3.2698, 1.4946, 3.0683],
                [-1.6855, -0.8312, -1.1395, 0.0481],
                [-2.1808, -2.5489, -2.3110, -3.1485],
            ]
        ]
    )
    count_unequal_element(output, result, 0.0001, 0.0001)

    # Test with 3D input array
    waveform = np.array(
        [
            [
                [
                    1,
                    1,
                    2,
                    2,
                    3,
                    3,
                    4,
                    4,
                    5,
                    5,
                    4,
                    4,
                    3,
                    3,
                    2,
                    2,
                    1,
                    1,
                    0,
                    0,
                    1,
                    1,
                    2,
                    2,
                    3,
                    3,
                    4,
                    4,
                    5,
                    5,
                ]
            ]
        ]
    )
    output = audio.MFCC(
        sample_rate=16000,
        n_mfcc=4,
        dct_type=2,
        norm=NormMode.ORTHO,
        log_mels=True,
        melkwargs={
            "n_fft": 16,
            "win_length": 16,
            "hop_length": 8,
            "f_min": 0.0,
            "f_max": 10000.0,
            "pad": 0,
            "n_mels": 5,
            "window": WindowType.HANN,
            "power": 2.0,
            "normalized": False,
            "center": True,
            "pad_mode": BorderType.REFLECT,
            "onesided": True,
            "norm": NormType.NONE,
            "mel_scale": MelType.HTK,
        },
    )(waveform)
    result = np.array(
        [
            [
                [
                    [2.7625, 5.6919, 3.6229, 3.9756],
                    [0.8142, 3.2698, 1.4946, 3.0683],
                    [-1.6855, -0.8312, -1.1395, 0.0481],
                    [-2.1808, -2.5489, -2.3110, -3.1485],
                ]
            ]
        ]
    )
    count_unequal_element(output, result, 0.0001, 0.0001)

    # Test with 4D input array
    waveform = np.array(
        [
            [
                [
                    [
                        1,
                        1,
                        2,
                        2,
                        3,
                        3,
                        4,
                        4,
                        5,
                        5,
                        4,
                        4,
                        3,
                        3,
                        2,
                        2,
                        1,
                        1,
                        0,
                        0,
                        1,
                        1,
                        2,
                        2,
                        3,
                        3,
                        4,
                        4,
                        5,
                        5,
                    ]
                ]
            ]
        ]
    )
    output = audio.MFCC(
        sample_rate=16000,
        n_mfcc=4,
        dct_type=2,
        norm=NormMode.ORTHO,
        log_mels=True,
        melkwargs={
            "n_fft": 16,
            "win_length": 16,
            "hop_length": 8,
            "f_min": 0.0,
            "f_max": 10000.0,
            "pad": 0,
            "n_mels": 5,
            "window": WindowType.HANN,
            "power": 2.0,
            "normalized": False,
            "center": True,
            "pad_mode": BorderType.REFLECT,
            "onesided": True,
            "norm": NormType.NONE,
            "mel_scale": MelType.HTK,
        },
    )(waveform)
    result = np.array(
        [
            [
                [
                    [
                        [2.7625, 5.6919, 3.6229, 3.9756],
                        [0.8142, 3.2698, 1.4946, 3.0683],
                        [-1.6855, -0.8312, -1.1395, 0.0481],
                        [-2.1808, -2.5489, -2.3110, -3.1485],
                    ]
                ]
            ]
        ]
    )
    count_unequal_element(output, result, 0.0001, 0.0001)

    # Test with 5D input array
    waveform = np.array(
        [
            [
                [
                    [
                        [
                            1,
                            1,
                            2,
                            2,
                            3,
                            3,
                            4,
                            4,
                            5,
                            5,
                            4,
                            4,
                            3,
                            3,
                            2,
                            2,
                            1,
                            1,
                            0,
                            0,
                            1,
                            1,
                            2,
                            2,
                            3,
                            3,
                            4,
                            4,
                            5,
                            5,
                        ]
                    ]
                ]
            ]
        ]
    )
    output = audio.MFCC(
        sample_rate=16000,
        n_mfcc=4,
        dct_type=2,
        norm=NormMode.ORTHO,
        log_mels=True,
        melkwargs={
            "n_fft": 16,
            "win_length": 16,
            "hop_length": 8,
            "f_min": 0.0,
            "f_max": 10000.0,
            "pad": 0,
            "n_mels": 5,
            "window": WindowType.HANN,
            "power": 2.0,
            "normalized": False,
            "center": True,
            "pad_mode": BorderType.REFLECT,
            "onesided": True,
            "norm": NormType.NONE,
            "mel_scale": MelType.HTK,
        },
    )(waveform)
    result = np.array(
        [
            [
                [
                    [
                        [
                            [2.7625, 5.6919, 3.6229, 3.9756],
                            [0.8142, 3.2698, 1.4946, 3.0683],
                            [-1.6855, -0.8312, -1.1395, 0.0481],
                            [-2.1808, -2.5489, -2.3110, -3.1485],
                        ]
                    ]
                ]
            ]
        ]
    )
    count_unequal_element(output, result, 0.0001, 0.0001)

    # Test with 6D input array
    waveform = np.array(
        [[[[[[1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 4, 4, 3, 3, 2, 2, 1, 1]]]]]]
    )
    output = audio.MFCC(
        sample_rate=16000,
        n_mfcc=4,
        dct_type=2,
        norm=NormMode.ORTHO,
        log_mels=True,
        melkwargs={
            "n_fft": 16,
            "win_length": 16,
            "hop_length": 8,
            "f_min": 0.0,
            "f_max": 10000.0,
            "pad": 0,
            "n_mels": 5,
            "window": WindowType.HANN,
            "power": 2.0,
            "normalized": False,
            "center": True,
            "pad_mode": BorderType.REFLECT,
            "onesided": True,
            "norm": NormType.NONE,
            "mel_scale": MelType.HTK,
        },
    )(waveform)
    result = np.array(
        [
            [
                [
                    [
                        [
                            [
                                [2.7625, 5.6919, 3.1166],
                                [0.8142, 3.2698, 1.2494],
                                [-1.6855, -0.8312, -1.4858],
                                [-2.1808, -2.5489, -2.3145],
                            ]
                        ]
                    ]
                ]
            ]
        ]
    )
    count_unequal_element(output, result, 0.0001, 0.0001)

    # Test with 7D input array
    waveform = np.array(
        [[[[[[[1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 4, 4, 3, 3, 2, 2, 1, 1]]]]]]]
    )
    output = audio.MFCC(
        sample_rate=16000,
        n_mfcc=4,
        dct_type=2,
        norm=NormMode.ORTHO,
        log_mels=True,
        melkwargs={
            "n_fft": 16,
            "win_length": 16,
            "hop_length": 8,
            "f_min": 0.0,
            "f_max": 10000.0,
            "pad": 0,
            "n_mels": 5,
            "window": WindowType.HANN,
            "power": 2.0,
            "normalized": False,
            "center": True,
            "pad_mode": BorderType.REFLECT,
            "onesided": True,
            "norm": NormType.NONE,
            "mel_scale": MelType.HTK,
        },
    )(waveform)
    result = np.array(
        [
            [
                [
                    [
                        [
                            [
                                [
                                    [2.7625, 5.6919, 3.1166],
                                    [0.8142, 3.2698, 1.2494],
                                    [-1.6855, -0.8312, -1.4858],
                                    [-2.1808, -2.5489, -2.3145],
                                ]
                            ]
                        ]
                    ]
                ]
            ]
        ]
    )
    count_unequal_element(output, result, 0.0001, 0.0001)

    # Test with log_mels set to False
    waveform = np.array(
        [
            [
                [
                    1,
                    1,
                    2,
                    2,
                    3,
                    3,
                    4,
                    4,
                    5,
                    5,
                    4,
                    4,
                    3,
                    3,
                    2,
                    2,
                    1,
                    1,
                    0,
                    0,
                    1,
                    1,
                    2,
                    2,
                    3,
                    3,
                    4,
                    4,
                    5,
                    5,
                ]
            ]
        ]
    )
    output = audio.MFCC(
        sample_rate=16000,
        n_mfcc=4,
        dct_type=2,
        norm=NormMode.ORTHO,
        log_mels=False,
        melkwargs={
            "n_fft": 16,
            "win_length": 16,
            "hop_length": 8,
            "f_min": 0.0,
            "f_max": 10000.0,
            "pad": 0,
            "n_mels": 5,
            "window": WindowType.HANN,
            "power": 2.0,
            "normalized": False,
            "center": True,
            "pad_mode": BorderType.REFLECT,
            "onesided": True,
            "norm": NormType.NONE,
            "mel_scale": MelType.HTK,
        },
    )(waveform)
    result = np.array(
        [
            [
                [
                    [11.9972, 24.7195, 15.7342, 17.2660],
                    [3.5362, 14.2004, 6.4908, 13.3254],
                    [-7.3199, -3.6097, -4.9487, 0.2088],
                    [-9.4713, -11.0696, -10.0365, -13.6739],
                ]
            ]
        ]
    )
    count_unequal_element(output, result, 0.0001, 0.0001)

    # Test with norm set to NormMode.NONE
    waveform = np.array(
        [
            [
                [
                    1,
                    1,
                    2,
                    2,
                    3,
                    3,
                    4,
                    4,
                    5,
                    5,
                    4,
                    4,
                    3,
                    3,
                    2,
                    2,
                    1,
                    1,
                    0,
                    0,
                    1,
                    1,
                    2,
                    2,
                    3,
                    3,
                    4,
                    4,
                    5,
                    5,
                ]
            ]
        ]
    )
    output = audio.MFCC(
        sample_rate=16000,
        n_mfcc=4,
        dct_type=2,
        norm=NormMode.NONE,
        log_mels=False,
        melkwargs={
            "n_fft": 16,
            "win_length": 16,
            "hop_length": 8,
            "f_min": 0.0,
            "f_max": 10000.0,
            "pad": 0,
            "n_mels": 5,
            "window": WindowType.HANN,
            "power": 2.0,
            "normalized": False,
            "center": True,
            "pad_mode": BorderType.REFLECT,
            "onesided": True,
            "norm": NormType.NONE,
            "mel_scale": MelType.HTK,
        },
    )(waveform)
    result = np.array(
        [
            [
                [
                    [53.6532, 110.5490, 70.3653, 77.2159],
                    [11.1825, 44.9055, 20.5257, 42.1386],
                    [-23.1477, -11.4149, -15.6492, 0.6602],
                    [-29.9507, -35.0053, -31.7381, -43.2408],
                ]
            ]
        ]
    )
    count_unequal_element(output, result, 0.0001, 0.0001)

    # Test with longer waveform and default melkwargs (None)
    waveform = np.array(
        [
            [
                [
                    1,
                    1,
                    2,
                    2,
                    3,
                    3,
                    4,
                    4,
                    5,
                    5,
                    4,
                    4,
                    3,
                    3,
                    2,
                    2,
                    1,
                    1,
                    0,
                    0,
                    1,
                    1,
                    2,
                    2,
                    3,
                    3,
                    4,
                    4,
                    5,
                    5,
                    4,
                    4,
                    3,
                    3,
                    2,
                    2,
                    1,
                    1,
                    0,
                    0,
                    1,
                    1,
                    2,
                    2,
                    3,
                    3,
                    4,
                    4,
                    5,
                    5,
                    4,
                    4,
                    3,
                    3,
                    2,
                    2,
                    1,
                    1,
                    0,
                    0,
                    1,
                    1,
                    2,
                    2,
                    3,
                    3,
                    4,
                    4,
                    5,
                    5,
                    4,
                    4,
                    3,
                    3,
                    2,
                    2,
                    1,
                    1,
                    0,
                    0,
                    1,
                    1,
                    2,
                    2,
                    3,
                    3,
                    4,
                    4,
                    5,
                    5,
                    4,
                    4,
                    3,
                    3,
                    2,
                    2,
                    1,
                    1,
                    0,
                    0,
                    1,
                    1,
                    2,
                    2,
                    3,
                    3,
                    4,
                    4,
                    5,
                    5,
                    4,
                    4,
                    3,
                    3,
                    2,
                    2,
                    1,
                    1,
                    0,
                    0,
                    1,
                    1,
                    2,
                    2,
                    3,
                    3,
                    4,
                    4,
                    5,
                    5,
                    4,
                    4,
                    3,
                    3,
                    2,
                    2,
                    1,
                    1,
                    0,
                    0,
                    1,
                    1,
                    2,
                    2,
                    3,
                    3,
                    4,
                    4,
                    5,
                    5,
                    4,
                    4,
                    3,
                    3,
                    2,
                    2,
                    1,
                    1,
                    0,
                    0,
                    1,
                    1,
                    2,
                    2,
                    3,
                    3,
                    4,
                    4,
                    5,
                    5,
                    4,
                    4,
                    3,
                    3,
                    2,
                    2,
                    1,
                    1,
                    0,
                    0,
                    1,
                    1,
                    2,
                    2,
                    3,
                    3,
                    4,
                    4,
                    5,
                    5,
                    4,
                    4,
                    3,
                    3,
                    2,
                    2,
                    1,
                    1,
                    0,
                    0,
                    1,
                    1,
                    2,
                    2,
                    3,
                    3,
                    4,
                    4,
                    5,
                    5,
                    4,
                    4,
                    3,
                    3,
                    2,
                    2,
                    1,
                    1,
                    0,
                    0,
                ]
            ]
        ]
    )
    output = audio.MFCC(
        sample_rate=16000,
        n_mfcc=4,
        dct_type=2,
        norm=NormMode.NONE,
        log_mels=True,
        melkwargs=None,
    )(waveform)
    result = np.array(
        [
            [
                [
                    [563.8333, 157.2625],
                    [50.7003, 15.8333],
                    [-26.0848, 43.8618],
                    [-246.8446, -248.2299],
                ]
            ]
        ]
    )
    count_unequal_element(output, result, 0.0001, 0.0001)

    # Test with default parameters to ensure no NaN values in output
    waveform = np.array(
        [
            [
                [
                    1,
                    1,
                    2,
                    2,
                    3,
                    3,
                    4,
                    4,
                    5,
                    5,
                    4,
                    4,
                    3,
                    3,
                    2,
                    2,
                    1,
                    1,
                    0,
                    0,
                    1,
                    1,
                    2,
                    2,
                    3,
                    3,
                    4,
                    4,
                    5,
                    5,
                    4,
                    4,
                    3,
                    3,
                    2,
                    2,
                    1,
                    1,
                    0,
                    0,
                    1,
                    1,
                    2,
                    2,
                    3,
                    3,
                    4,
                    4,
                    5,
                    5,
                    4,
                    4,
                    3,
                    3,
                    2,
                    2,
                    1,
                    1,
                    0,
                    0,
                    1,
                    1,
                    2,
                    2,
                    3,
                    3,
                    4,
                    4,
                    5,
                    5,
                    4,
                    4,
                    3,
                    3,
                    2,
                    2,
                    1,
                    1,
                    0,
                    0,
                    1,
                    1,
                    2,
                    2,
                    3,
                    3,
                    4,
                    4,
                    5,
                    5,
                    4,
                    4,
                    3,
                    3,
                    2,
                    2,
                    1,
                    1,
                    0,
                    0,
                    1,
                    1,
                    2,
                    2,
                    3,
                    3,
                    4,
                    4,
                    5,
                    5,
                    4,
                    4,
                    3,
                    3,
                    2,
                    2,
                    1,
                    1,
                    0,
                    0,
                    1,
                    1,
                    2,
                    2,
                    3,
                    3,
                    4,
                    4,
                    5,
                    5,
                    4,
                    4,
                    3,
                    3,
                    2,
                    2,
                    1,
                    1,
                    0,
                    0,
                    1,
                    1,
                    2,
                    2,
                    3,
                    3,
                    4,
                    4,
                    5,
                    5,
                    4,
                    4,
                    3,
                    3,
                    2,
                    2,
                    1,
                    1,
                    0,
                    0,
                    1,
                    1,
                    2,
                    2,
                    3,
                    3,
                    4,
                    4,
                    5,
                    5,
                    4,
                    4,
                    3,
                    3,
                    2,
                    2,
                    1,
                    1,
                    0,
                    0,
                    1,
                    1,
                    2,
                    2,
                    3,
                    3,
                    4,
                    4,
                    5,
                    5,
                    4,
                    4,
                    3,
                    3,
                    2,
                    2,
                    1,
                    1,
                    0,
                    0,
                    1,
                    1,
                    2,
                    2,
                    3,
                    3,
                    4,
                    4,
                    5,
                    5,
                    4,
                    4,
                    3,
                    3,
                    2,
                    2,
                    1,
                    1,
                    0,
                    0,
                ]
            ]
        ]
    )
    output = audio.MFCC()(waveform)
    result = np.array(
        [
            [
                [
                    [117.9204, 39.8853],
                    [27.2399, 17.7761],
                    [5.6903, 24.6769],
                    [-55.3415, -55.7178],
                    [-41.1866, -34.0315],
                    [-29.4843, -40.2680],
                    [39.9977, 44.8343],
                    [29.2145, 51.3140],
                    [-12.8123, 8.8970],
                    [-22.7150, -50.4240],
                    [26.8296, 18.4089],
                    [-14.7456, 12.2869],
                    [-9.9099, -22.1687],
                    [27.8783, 27.7343],
                    [16.6737, 38.5366],
                    [-12.3834, -27.5997],
                    [-7.8851, 23.0055],
                    [-4.7462, -33.6250],
                    [8.3553, 2.3019],
                    [-3.2685, 16.1033],
                    [-3.6666, 4.5545],
                    [22.7259, 22.8965],
                    [-20.4782, -25.4887],
                    [7.2899, 14.4669],
                    [-2.0913, -8.8214],
                    [9.8611, 10.9064],
                    [10.9958, -3.1751],
                    [13.2490, 29.7223],
                    [-5.6389, 5.7694],
                    [11.0445, -3.0599],
                    [-2.7316, -8.5603],
                    [1.9939, 6.0871],
                    [2.9014, -2.8322],
                    [-4.1338, -0.4745],
                    [-0.4667, 2.9039],
                    [-13.3180, -10.3825],
                    [-22.2865, -15.1669],
                    [3.5515, -4.7240],
                    [-19.0482, -17.8276],
                    [-6.7367, -12.2017],
                ]
            ]
        ]
    )
    count_unequal_element(output, result, 0.001, 0.001)

    # Test MFCC operation in pipeline mode with short waveform
    waveform = np.array(
        [
            [0.8236, 0.2049, 0.3335],
            [0.5933, 0.9911, 0.2482],
            [0.3007, 0.9054, 0.7598],
            [0.5394, 0.2842, 0.5634],
            [0.6363, 0.2226, 0.2288],
        ]
    )
    numpy_slices_dataset = ds.NumpySlicesDataset(data=waveform, column_names=["audio"])
    transforms = [audio.MFCC(4000, 1500, 2)]
    numpy_slices_dataset.map(operations=transforms, input_columns=["audio"])


def test_mfcc_param_check():
    """
    Feature: MfccOps
    Description: Test MfccOps with invalid input data types and parameters
    Expectation: Correct error types and messages are raised as expected
    """

    with pytest.raises(
        TypeError,
        match=r"Argument sample_rate with value 16000.242 is not of type "
        r"\[<class 'int'>\], but got <class 'float'>.",
    ):
        audio.MFCC(
            sample_rate=16000.242,
            n_mfcc=4,
            dct_type=2,
            norm=NormMode.ORTHO,
            log_mels=True,
            melkwargs={
                "n_fft": 16,
                "win_length": 16,
                "hop_length": 8,
                "f_min": 0.0,
                "f_max": 10000.0,
                "pad": 0,
                "n_mels": 5,
                "window": WindowType.HANN,
                "power": 2.0,
                "normalized": False,
                "center": True,
                "pad_mode": BorderType.REFLECT,
                "onesided": True,
                "norm": NormType.NONE,
                "mel_scale": MelType.HTK,
            },
        )

    # Test with negative sample_rate parameter
    with pytest.raises(
        ValueError,
        match=r"Input sample_rate is not within the required interval of "
        r"\[0, 2147483647\].",
    ):
        audio.MFCC(
            sample_rate=-16000,
            n_mfcc=4,
            dct_type=2,
            norm=NormMode.ORTHO,
            log_mels=True,
            melkwargs={
                "n_fft": 16,
                "win_length": 16,
                "hop_length": 8,
                "f_min": 0.0,
                "f_max": 10000.0,
                "pad": 0,
                "n_mels": 5,
                "window": WindowType.HANN,
                "power": 2.0,
                "normalized": False,
                "center": True,
                "pad_mode": BorderType.REFLECT,
                "onesided": True,
                "norm": NormType.NONE,
                "mel_scale": MelType.HTK,
            },
        )

    # Test with negative n_mfcc parameter
    try:
        audio.MFCC(0, -1500, 2)
    except ValueError as e:
        assert (
            "Input n_mfcc is not within the required interval of [0, 2147483647]."
            in str(e)
        )

    # Test with float type n_mfcc parameter
    try:
        audio.MFCC(0, 1500.92, 2)
    except TypeError as e:
        assert (
            "Argument n_mfcc with value 1500.92 is not of type [<class 'int'>], but got <class 'float'>."
            in str(e)
        )

    # Test with short waveform that causes padding error
    try:
        waveform = np.array(
            [
                [0.8236, 0.2049, 0.3335],
                [0.5933, 0.9911, 0.2482],
                [0.3007, 0.9054, 0.7598],
                [0.5394, 0.2842, 0.5634],
                [0.6363, 0.2226, 0.2288],
            ]
        )
        numpy_slices_dataset = ds.NumpySlicesDataset(
            data=waveform, column_names=["audio"]
        )
        transforms = audio.MFCC()
        dataset = numpy_slices_dataset.map(
            operations=transforms, input_columns=["audio"]
        )
        for _ in dataset.create_dict_iterator(output_numpy=False):
            pass
    except RuntimeError as e:
        assert (
            "map operation: [MFCC] failed. MelSpectrogram: Padding size should be "
            "less than the corresponding input dimension." in str(e)
        )

    # Test with invalid dct_type parameter (value 7)
    try:
        audio.MFCC(16000, 1500, 7)
    except ValueError as e:
        assert "dct_type must be 2, but got : 7." in str(e)

    # Test with invalid negative dct_type parameter
    try:
        audio.MFCC(16000, 1500, -7)
    except ValueError as e:
        assert "dct_type must be 2, but got : -7." in str(e)

    # Test with invalid dct_type parameter type (list)
    try:
        audio.MFCC(16000, 1500, ["322"])
    except ValueError as e:
        assert "dct_type must be 2, but got : ['322']." in str(e)

    # Test with invalid norm parameter type (string type)
    try:
        audio.MFCC(16000, 1500, 2, "skf")
    except TypeError as e:
        assert (
            "Argument norm with value skf is not of type [<enum 'NormMode'>], but got <class 'str'>."
            in str(e)
        )

    # Test with invalid log_mels parameter type (string type)
    try:
        audio.MFCC(16000, 1500, 2, NormMode.ORTHO, "true")
    except TypeError as e:
        assert (
            "Argument log_mels with value true is not of type [<class 'bool'>], but got <class 'str'>."
            in str(e)
        )

    # Test with negative n_fft in melkwargs
    try:
        audio.MFCC(
            sample_rate=16000,
            n_mfcc=4,
            dct_type=2,
            norm=NormMode.ORTHO,
            log_mels=True,
            melkwargs={
                "n_fft": -16,
                "win_length": 16,
                "hop_length": 8,
                "f_min": 0.0,
                "f_max": 10000.0,
                "pad": 0,
                "n_mels": 5,
                "window": WindowType.HANN,
                "power": 2.0,
                "normalized": False,
                "center": True,
                "pad_mode": BorderType.REFLECT,
                "onesided": True,
                "norm": NormType.NONE,
                "mel_scale": MelType.HTK,
            },
        )
    except ValueError as e:
        assert (
            "Input n_fft is not within the required interval of [1, 2147483647]."
            in str(e)
        )

    # Test with negative win_length in melkwargs
    try:
        audio.MFCC(
            sample_rate=16000,
            n_mfcc=4,
            dct_type=2,
            norm=NormMode.ORTHO,
            log_mels=True,
            melkwargs={
                "n_fft": 16,
                "win_length": -16,
                "hop_length": 8,
                "f_min": 0.0,
                "f_max": 10000.0,
                "pad": 0,
                "n_mels": 5,
                "window": WindowType.HANN,
                "power": 2.0,
                "normalized": False,
                "center": True,
                "pad_mode": BorderType.REFLECT,
                "onesided": True,
                "norm": NormType.NONE,
                "mel_scale": MelType.HTK,
            },
        )
    except ValueError as e:
        assert (
            "Input win_length is not within the required interval of [0, 2147483647]."
            in str(e)
        )

    # Test with negative hop_length in melkwargs
    try:
        audio.MFCC(
            sample_rate=16000,
            n_mfcc=4,
            dct_type=2,
            norm=NormMode.ORTHO,
            log_mels=True,
            melkwargs={
                "n_fft": 16,
                "win_length": 16,
                "hop_length": -8,
                "f_min": 0.0,
                "f_max": 10000.0,
                "pad": 0,
                "n_mels": 5,
                "window": WindowType.HANN,
                "power": 2.0,
                "normalized": False,
                "center": True,
                "pad_mode": BorderType.REFLECT,
                "onesided": True,
                "norm": NormType.NONE,
                "mel_scale": MelType.HTK,
            },
        )
    except ValueError as e:
        assert (
            "Input hop_length is not within the required interval of [1, 2147483647]"
            in str(e)
        )

    # Test with negative f_max in melkwargs
    try:
        audio.MFCC(
            sample_rate=16000,
            n_mfcc=4,
            dct_type=2,
            norm=NormMode.ORTHO,
            log_mels=True,
            melkwargs={
                "n_fft": 16,
                "win_length": 16,
                "hop_length": 8,
                "f_min": 0.0,
                "f_max": -10000.0,
                "pad": 0,
                "n_mels": 5,
                "window": WindowType.HANN,
                "power": 2.0,
                "normalized": False,
                "center": True,
                "pad_mode": BorderType.REFLECT,
                "onesided": True,
                "norm": NormType.NONE,
                "mel_scale": MelType.HTK,
            },
        )
    except ValueError as e:
        assert (
            "Input f_max is not within the required interval of (0, 16777216]."
            in str(e)
        )

    # Test with negative pad in melkwargs
    try:
        audio.MFCC(
            sample_rate=16000,
            n_mfcc=4,
            dct_type=2,
            norm=NormMode.ORTHO,
            log_mels=True,
            melkwargs={
                "n_fft": 16,
                "win_length": 16,
                "hop_length": 8,
                "f_min": 0.0,
                "f_max": 10000.0,
                "pad": -10,
                "n_mels": 5,
                "window": WindowType.HANN,
                "power": 2.0,
                "normalized": False,
                "center": True,
                "pad_mode": BorderType.REFLECT,
                "onesided": True,
                "norm": NormType.NONE,
                "mel_scale": MelType.HTK,
            },
        )
    except ValueError as e:
        assert (
            "Input pad is not within the required interval of [0, 2147483647]."
            in str(e)
        )

    # Test with negative n_mels in melkwargs
    try:
        audio.MFCC(
            sample_rate=16000,
            n_mfcc=4,
            dct_type=2,
            norm=NormMode.ORTHO,
            log_mels=True,
            melkwargs={
                "n_fft": 16,
                "win_length": 16,
                "hop_length": 8,
                "f_min": 0.0,
                "f_max": 10000.0,
                "pad": 0,
                "n_mels": -128,
                "window": WindowType.HANN,
                "power": 2.0,
                "normalized": False,
                "center": True,
                "pad_mode": BorderType.REFLECT,
                "onesided": True,
                "norm": NormType.NONE,
                "mel_scale": MelType.HTK,
            },
        )
    except ValueError as e:
        assert (
            "Input n_mels is not within the required interval of [1, 2147483647]."
            in str(e)
        )

    # Test with negative power in melkwargs
    try:
        audio.MFCC(
            sample_rate=16000,
            n_mfcc=4,
            dct_type=2,
            norm=NormMode.ORTHO,
            log_mels=True,
            melkwargs={
                "n_fft": 16,
                "win_length": 16,
                "hop_length": 8,
                "f_min": 0.0,
                "f_max": 10000.0,
                "pad": 0,
                "n_mels": 5,
                "window": WindowType.HANN,
                "power": -2.0,
                "normalized": False,
                "center": True,
                "pad_mode": BorderType.REFLECT,
                "onesided": True,
                "norm": NormType.NONE,
                "mel_scale": MelType.HTK,
            },
        )
    except ValueError as e:
        assert (
            "Input power is not within the required interval of [0, 16777216]."
            in str(e)
        )

    # Test with f_min greater than f_max in melkwargs
    try:
        audio.MFCC(
            sample_rate=16000,
            n_mfcc=4,
            dct_type=2,
            norm=NormMode.ORTHO,
            log_mels=True,
            melkwargs={
                "n_fft": 16,
                "win_length": 16,
                "hop_length": 8,
                "f_min": 20000.0,
                "f_max": 10000.0,
                "pad": 0,
                "n_mels": 5,
                "window": WindowType.HANN,
                "power": 2.0,
                "normalized": False,
                "center": True,
                "pad_mode": BorderType.REFLECT,
                "onesided": True,
                "norm": NormType.NONE,
                "mel_scale": MelType.HTK,
            },
        )
    except ValueError as e:
        assert "MelScale: f_max should be greater than f_min." in str(e)

    # Test with f_max set to 0 in melkwargs
    try:
        audio.MFCC(
            sample_rate=16000,
            n_mfcc=4,
            dct_type=2,
            norm=NormMode.ORTHO,
            log_mels=True,
            melkwargs={
                "n_fft": 16,
                "win_length": 16,
                "hop_length": 8,
                "f_min": 9000.0,
                "f_max": 0.0,
                "pad": 0,
                "n_mels": 5,
                "window": WindowType.HANN,
                "power": 2.0,
                "normalized": False,
                "center": True,
                "pad_mode": BorderType.REFLECT,
                "onesided": True,
                "norm": NormType.NONE,
                "mel_scale": MelType.HTK,
            },
        )
    except ValueError as e:
        assert (
            "Input f_max is not within the required interval of (0, 16777216]."
            in str(e)
        )

    # Test with win_length greater than n_fft in melkwargs
    try:
        audio.MFCC(
            sample_rate=16000,
            n_mfcc=4,
            dct_type=2,
            norm=NormMode.ORTHO,
            log_mels=True,
            melkwargs={
                "n_fft": 16,
                "win_length": 1000,
                "hop_length": 8,
                "f_min": 0.0,
                "f_max": 10000.0,
                "pad": 0,
                "n_mels": 5,
                "window": WindowType.HANN,
                "power": 2.0,
                "normalized": False,
                "center": True,
                "pad_mode": BorderType.REFLECT,
                "onesided": True,
                "norm": NormType.NONE,
                "mel_scale": MelType.HTK,
            },
        )
    except RuntimeError as e:
        assert (
            "MFCC: win_length must be less than or equal to n_fft, but got win_length: 1000, n_fft: 16"
            in str(e)
        )

    # Test with invalid window type (string type) in melkwargs
    try:
        audio.MFCC(
            sample_rate=16000,
            n_mfcc=4,
            dct_type=2,
            norm=NormMode.ORTHO,
            log_mels=True,
            melkwargs={
                "n_fft": 16,
                "win_length": 16,
                "hop_length": 8,
                "f_min": 0.0,
                "f_max": 10000.0,
                "pad": 0,
                "n_mels": 5,
                "window": "WindowType.HANN",
                "power": 2.0,
                "normalized": False,
                "center": True,
                "pad_mode": BorderType.REFLECT,
                "onesided": True,
                "norm": NormType.NONE,
                "mel_scale": MelType.HTK,
            },
        )
    except TypeError as e:
        assert (
            "Argument window with value WindowType.HANN is not of type [<enum 'WindowType'>], "
            "but got <class 'str'>." in str(e)
        )

    # Test with invalid pad_mode type (int type) in melkwargs
    try:
        audio.MFCC(
            sample_rate=16000,
            n_mfcc=4,
            dct_type=2,
            norm=NormMode.ORTHO,
            log_mels=True,
            melkwargs={
                "n_fft": 16,
                "win_length": 16,
                "hop_length": 8,
                "f_min": 0.0,
                "f_max": 10000.0,
                "pad": 0,
                "n_mels": 5,
                "window": WindowType.HANN,
                "power": 2.0,
                "normalized": False,
                "center": True,
                "pad_mode": 235,
                "onesided": True,
                "norm": NormType.NONE,
                "mel_scale": MelType.HTK,
            },
        )
    except TypeError as e:
        assert (
            "Argument pad_mode with value 235 is not of type [<enum 'BorderType'>], but got <class 'int'>."
            in str(e)
        )

    # Test with invalid norm type (string type) in melkwargs
    try:
        audio.MFCC(
            sample_rate=16000,
            n_mfcc=4,
            dct_type=2,
            norm=NormMode.ORTHO,
            log_mels=True,
            melkwargs={
                "n_fft": 16,
                "win_length": 16,
                "hop_length": 8,
                "f_min": 0.0,
                "f_max": 10000.0,
                "pad": 0,
                "n_mels": 5,
                "window": WindowType.HANN,
                "power": 2.0,
                "normalized": False,
                "center": True,
                "pad_mode": BorderType.REFLECT,
                "onesided": True,
                "norm": "NormType.NONE",
                "mel_scale": MelType.HTK,
            },
        )
    except TypeError as e:
        assert (
            "Argument norm with value NormType.NONE is not of type [<enum 'NormType'>], "
            "but got <class 'str'>." in str(e)
        )

    # Test with invalid mel_scale type (string type) in melkwargs
    try:
        audio.MFCC(
            sample_rate=16000,
            n_mfcc=4,
            dct_type=2,
            norm=NormMode.ORTHO,
            log_mels=True,
            melkwargs={
                "n_fft": 16,
                "win_length": 16,
                "hop_length": 8,
                "f_min": 0.0,
                "f_max": 10000.0,
                "pad": 0,
                "n_mels": 5,
                "window": WindowType.HANN,
                "power": 2.0,
                "normalized": False,
                "center": True,
                "pad_mode": BorderType.REFLECT,
                "onesided": True,
                "norm": NormType.NONE,
                "mel_scale": "MelType.HTK",
            },
        )
    except TypeError as e:
        assert (
            "Argument mel_type with value MelType.HTK is not of type [<enum 'MelType'>],"
            " but got <class 'str'>." in str(e)
        )


if __name__ == "__main__":
    test_mfcc_pipeline()
    test_mfcc_eager()
    test_mfcc_param()
    test_mfcc_transform()
    test_mfcc_param_check()
