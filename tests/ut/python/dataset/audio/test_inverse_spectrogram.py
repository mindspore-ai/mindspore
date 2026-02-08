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
"""Test InverseSpectrogram."""

import numpy as np

import mindspore.dataset as ds
from mindspore.dataset import audio
from mindspore.dataset.audio import WindowType, BorderType
from . import count_unequal_element


def test_inverse_spectrogram_pipeline():
    """
    Feature: Test pipeline mode normal testcase: InverseSpectrogram
    Description: Input audio signal to test pipeline
    Expectation: Generate expected output after cases were executed
    """
    waveform = [
        [
            [
                [[1.0, 1.0]],
                [[2.0, 2.0]],
                [[3.0, 3.0]],
                [[4.0, 4.0]],
                [[5.0, 5.0]],
                [[6.0, 6.0]],
                [[5.0, 5.0]],
                [[4.0, 4.0]],
                [[3.0, 3.0]],
            ],
            [
                [[2.0, 2.0]],
                [[1.0, 1.0]],
                [[0.0, 0.0]],
                [[1.0, 1.0]],
                [[2.0, 2.0]],
                [[3.0, 3.0]],
                [[4.0, 4.0]],
                [[5.0, 5.0]],
                [[6.0, 6.0]],
            ],
        ]
    ]
    dataset = ds.NumpySlicesDataset(waveform, column_names=["audio"], shuffle=False)
    out = audio.InverseSpectrogram(
        length=1,
        n_fft=16,
        win_length=16,
        hop_length=8,
        pad=0,
        window=WindowType.HANN,
        normalized=False,
        center=True,
        pad_mode=BorderType.REFLECT,
        onesided=True,
    )
    dataset = dataset.map(
        operations=out, input_columns=["audio"], output_columns=["InverseSpectrogram"]
    )
    result = np.array([[-0.1250], [0.0000]])
    for data1 in dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
        count_unequal_element(data1["InverseSpectrogram"], result, 0.0001, 0.0001)


def test_inverse_spectrogram_eager():
    """
    Feature: Test pipeline mode normal testcase: InverseSpectrogram
    Description: Input audio signal to test eager
    Expectation: Generate expected output after cases were executed
    """
    waveform = np.array(
        [
            [
                [[1.0, 1.0]],
                [[2.0, 2.0]],
                [[3.0, 3.0]],
                [[4.0, 4.0]],
                [[5.0, 5.0]],
                [[6.0, 6.0]],
                [[5.0, 5.0]],
                [[4.0, 4.0]],
                [[3.0, 3.0]],
            ],
            [
                [[2.0, 2.0]],
                [[1.0, 1.0]],
                [[0.0, 0.0]],
                [[1.0, 1.0]],
                [[2.0, 2.0]],
                [[3.0, 3.0]],
                [[4.0, 4.0]],
                [[5.0, 5.0]],
                [[6.0, 6.0]],
            ],
        ]
    )
    out = audio.InverseSpectrogram(
        length=1,
        n_fft=16,
        win_length=16,
        hop_length=8,
        pad=1,
        window=WindowType.HANN,
        normalized=False,
        center=True,
        pad_mode=BorderType.REFLECT,
        onesided=True,
    )(waveform)
    result = np.array([[0.1399], [0.1034]])
    count_unequal_element(out, result, 0.0001, 0.0001)


def test_inverse_spectrogram_param():
    """
    Feature: Test InverseSpectrogram invalid parameter
    Description: Test some invalid parameters
    Expectation: throw ValueError, TypeError or RuntimeError exception
    """
    try:
        audio.InverseSpectrogram(length=-1)
    except ValueError as error:
        assert (
            "Input length is not within the required interval of [0, 2147483647]."
            in str(error)
        )
    try:
        audio.InverseSpectrogram(length=1, n_fft=-1)
    except ValueError as error:
        assert (
            "Input n_fft is not within the required interval of [1, 2147483647]."
            in str(error)
        )
    try:
        audio.InverseSpectrogram(length=1, n_fft=0)
    except ValueError as error:
        assert (
            "Input n_fft is not within the required interval of [1, 2147483647]."
            in str(error)
        )
    try:
        audio.InverseSpectrogram(length=1, win_length=-1)
    except ValueError as error:
        assert (
            "Input win_length is not within the required interval of [1, 2147483647]."
            in str(error)
        )
    try:
        audio.InverseSpectrogram(length=1, win_length="s")
    except TypeError as error:
        assert (
            "Argument win_length with value s is not of type [<class 'int'>], but got <class 'str'>."
            in str(error)
        )
    try:
        audio.InverseSpectrogram(length=1, hop_length=-1)
    except ValueError as error:
        assert (
            "Input hop_length is not within the required interval of [1, 2147483647]."
            in str(error)
        )
    try:
        audio.InverseSpectrogram(length=1, hop_length=-100)
    except ValueError as error:
        assert (
            "Input hop_length is not within the required interval of [1, 2147483647]."
            in str(error)
        )
    try:
        audio.InverseSpectrogram(length=1, win_length=300, n_fft=200)
    except ValueError as error:
        assert (
            "Input win_length should be no more than n_fft, but got win_length: 300 and n_fft: 200."
            in str(error)
        )
    try:
        audio.InverseSpectrogram(length=1, pad=-1)
    except ValueError as error:
        assert (
            "Input pad is not within the required interval of [0, 2147483647]."
            in str(error)
        )
    try:
        audio.InverseSpectrogram(length=1, n_fft=False)
    except TypeError as error:
        assert "Argument n_fft with value False is not of type (<class 'int'>,)" in str(
            error
        )
    try:
        audio.InverseSpectrogram(length=1, n_fft="s")
    except TypeError as error:
        assert (
            "Argument n_fft with value s is not of type [<class 'int'>], but got <class 'str'>."
            in str(error)
        )
    try:
        audio.InverseSpectrogram(length=1, window=False)
    except TypeError as error:
        assert (
            "Argument window with value False is not of type [<enum 'WindowType'>], but got <class 'bool'>."
            in str(error)
        )
    try:
        audio.InverseSpectrogram(length=1, pad_mode=False)
    except TypeError as error:
        assert (
            "Argument pad_mode with value False is not of type [<enum 'BorderType'>], but got <class 'bool'>."
            in str(error)
        )
    try:
        audio.InverseSpectrogram(length=1, onesided="GanJisong")
    except TypeError as error:
        assert (
            "Argument onesided with value GanJisong is not of type [<class 'bool'>], but got <class 'str'>."
            in str(error)
        )
    try:
        audio.InverseSpectrogram(length=1, center="MindSpore")
    except TypeError as error:
        assert (
            "Argument center with value MindSpore is not of type [<class 'bool'>], but got <class 'str'>."
            in str(error)
        )
    try:
        audio.InverseSpectrogram(length=1, normalized="s")
    except TypeError as error:
        assert (
            "Argument normalized with value s is not of type [<class 'bool'>], but got <class 'str'>."
            in str(error)
        )
    try:
        audio.InverseSpectrogram(length=1, normalized=1)
    except TypeError as error:
        assert (
            "Argument normalized with value 1 is not of type [<class 'bool'>], but got <class 'int'>."
            in str(error)
        )


def test_inverse_spectrogram_transform():
    """
    Feature: InverseSpectrogram
    Description: Test InverseSpectrogram with various valid input parameters and data types
    Expectation: The operation completes successfully
    """
    # test InverseSpectrogram pipeline mode normal
    waveform = [
        [
            [
                [[1.0, 1.0]],
                [[2.0, 2.0]],
                [[3.0, 3.0]],
                [[4.0, 4.0]],
                [[5.0, 5.0]],
                [[6.0, 6.0]],
                [[5.0, 5.0]],
                [[4.0, 4.0]],
                [[3.0, 3.0]],
            ],
            [
                [[2.0, 2.0]],
                [[1.0, 1.0]],
                [[0.0, 0.0]],
                [[1.0, 1.0]],
                [[2.0, 2.0]],
                [[3.0, 3.0]],
                [[4.0, 4.0]],
                [[5.0, 5.0]],
                [[6.0, 6.0]],
            ],
        ]
    ]
    dataset = ds.NumpySlicesDataset(waveform, column_names=["audio"], shuffle=False)
    inverse_spectrogram = audio.InverseSpectrogram(
        n_fft=16,
        win_length=16,
        hop_length=8,
        pad=0,
        window=WindowType.HANN,
        normalized=False,
        center=True,
        pad_mode=BorderType.REFLECT,
        onesided=True,
        length=1,
    )
    dataset = dataset.map(
        operations=inverse_spectrogram,
        input_columns=["audio"],
        output_columns=["InverseSpectrogram"],
    )
    dataset = dataset.project(columns=["InverseSpectrogram"])
    result = np.array([[-0.1250], [0.0000]])
    for data1 in dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
        count_unequal_element(data1["InverseSpectrogram"], result, 0.0001, 0.0001)

    # test InverseSpectrogram eager mode
    waveform = np.array(
        [
            [
                [[1.0, 1.0]],
                [[2.0, 2.0]],
                [[3.0, 3.0]],
                [[4.0, 4.0]],
                [[5.0, 5.0]],
                [[6.0, 6.0]],
                [[5.0, 5.0]],
                [[4.0, 4.0]],
                [[3.0, 3.0]],
            ],
            [
                [[2.0, 2.0]],
                [[1.0, 1.0]],
                [[0.0, 0.0]],
                [[1.0, 1.0]],
                [[2.0, 2.0]],
                [[3.0, 3.0]],
                [[4.0, 4.0]],
                [[5.0, 5.0]],
                [[6.0, 6.0]],
            ],
        ]
    )
    inverse_spectrogram = audio.InverseSpectrogram(
        n_fft=16,
        win_length=16,
        hop_length=8,
        pad=0,
        window=WindowType.HANN,
        normalized=False,
        center=True,
        pad_mode=BorderType.REFLECT,
        onesided=True,
        length=1,
    )
    inverse_spectrogram(waveform)

    # test InverseSpectrogram normal testcase: test window hamming
    waveform = np.array(
        [
            [
                [[1.0, 1.0]],
                [[2.0, 2.0]],
                [[3.0, 3.0]],
                [[4.0, 4.0]],
                [[5.0, 5.0]],
                [[6.0, 6.0]],
                [[5.0, 5.0]],
                [[4.0, 4.0]],
                [[3.0, 3.0]],
            ],
            [
                [[2.0, 2.0]],
                [[1.0, 1.0]],
                [[0.0, 0.0]],
                [[1.0, 1.0]],
                [[2.0, 2.0]],
                [[3.0, 3.0]],
                [[4.0, 4.0]],
                [[5.0, 5.0]],
                [[6.0, 6.0]],
            ],
        ]
    )
    inverse_spectrogram = audio.InverseSpectrogram(
        n_fft=16,
        win_length=16,
        hop_length=8,
        pad=0,
        window=WindowType.HAMMING,
        normalized=False,
        center=True,
        pad_mode=BorderType.REFLECT,
        onesided=True,
        length=1,
    )
    inverse_spectrogram(waveform)

    # test InverseSpectrogram normal testcase: test window bartlett
    waveform = np.array(
        [
            [
                [[1.0, 1.0]],
                [[2.0, 2.0]],
                [[3.0, 3.0]],
                [[4.0, 4.0]],
                [[5.0, 5.0]],
                [[6.0, 6.0]],
                [[5.0, 5.0]],
                [[4.0, 4.0]],
                [[3.0, 3.0]],
            ],
            [
                [[2.0, 2.0]],
                [[1.0, 1.0]],
                [[0.0, 0.0]],
                [[1.0, 1.0]],
                [[2.0, 2.0]],
                [[3.0, 3.0]],
                [[4.0, 4.0]],
                [[5.0, 5.0]],
                [[6.0, 6.0]],
            ],
        ]
    )
    inverse_spectrogram = audio.InverseSpectrogram(
        n_fft=16,
        win_length=16,
        hop_length=8,
        pad=0,
        window=WindowType.BARTLETT,
        normalized=False,
        center=True,
        pad_mode=BorderType.REFLECT,
        onesided=True,
        length=1,
    )
    inverse_spectrogram(waveform)

    # test InverseSpectrogram normal testcase: test window blackman
    waveform = np.array(
        [
            [
                [[1.0, 1.0]],
                [[2.0, 2.0]],
                [[3.0, 3.0]],
                [[4.0, 4.0]],
                [[5.0, 5.0]],
                [[6.0, 6.0]],
                [[5.0, 5.0]],
                [[4.0, 4.0]],
                [[3.0, 3.0]],
            ],
            [
                [[2.0, 2.0]],
                [[1.0, 1.0]],
                [[0.0, 0.0]],
                [[1.0, 1.0]],
                [[2.0, 2.0]],
                [[3.0, 3.0]],
                [[4.0, 4.0]],
                [[5.0, 5.0]],
                [[6.0, 6.0]],
            ],
        ]
    )
    inverse_spectrogram = audio.InverseSpectrogram(
        n_fft=16,
        win_length=16,
        hop_length=8,
        pad=0,
        window=WindowType.BLACKMAN,
        normalized=False,
        center=True,
        pad_mode=BorderType.REFLECT,
        onesided=True,
        length=1,
    )
    inverse_spectrogram(waveform)

    # test InverseSpectrogram normal testcase: test window kaiser
    waveform = np.array(
        [
            [
                [[1.0, 1.0]],
                [[2.0, 2.0]],
                [[3.0, 3.0]],
                [[4.0, 4.0]],
                [[5.0, 5.0]],
                [[6.0, 6.0]],
                [[5.0, 5.0]],
                [[4.0, 4.0]],
                [[3.0, 3.0]],
            ],
            [
                [[2.0, 2.0]],
                [[1.0, 1.0]],
                [[0.0, 0.0]],
                [[1.0, 1.0]],
                [[2.0, 2.0]],
                [[3.0, 3.0]],
                [[4.0, 4.0]],
                [[5.0, 5.0]],
                [[6.0, 6.0]],
            ],
        ]
    )
    inverse_spectrogram = audio.InverseSpectrogram(
        n_fft=16,
        win_length=16,
        hop_length=8,
        pad=0,
        window=WindowType.KAISER,
        normalized=False,
        center=True,
        pad_mode=BorderType.REFLECT,
        onesided=True,
        length=1,
    )
    inverse_spectrogram(waveform)

    # test InverseSpectrogram normal testcase: test window edge
    waveform = np.array(
        [
            [
                [[1.0, 1.0]],
                [[2.0, 2.0]],
                [[3.0, 3.0]],
                [[4.0, 4.0]],
                [[5.0, 5.0]],
                [[6.0, 6.0]],
                [[5.0, 5.0]],
                [[4.0, 4.0]],
                [[3.0, 3.0]],
            ],
            [
                [[2.0, 2.0]],
                [[1.0, 1.0]],
                [[0.0, 0.0]],
                [[1.0, 1.0]],
                [[2.0, 2.0]],
                [[3.0, 3.0]],
                [[4.0, 4.0]],
                [[5.0, 5.0]],
                [[6.0, 6.0]],
            ],
        ]
    )
    inverse_spectrogram = audio.InverseSpectrogram(
        n_fft=16,
        win_length=16,
        hop_length=8,
        pad=0,
        window=WindowType.HANN,
        normalized=False,
        center=True,
        pad_mode=BorderType.EDGE,
        onesided=True,
        length=1,
    )
    inverse_spectrogram(waveform)

    # test InverseSpectrogram normal testcase: test window symmetric
    waveform = np.array(
        [
            [
                [[1.0, 1.0]],
                [[2.0, 2.0]],
                [[3.0, 3.0]],
                [[4.0, 4.0]],
                [[5.0, 5.0]],
                [[6.0, 6.0]],
                [[5.0, 5.0]],
                [[4.0, 4.0]],
                [[3.0, 3.0]],
            ],
            [
                [[2.0, 2.0]],
                [[1.0, 1.0]],
                [[0.0, 0.0]],
                [[1.0, 1.0]],
                [[2.0, 2.0]],
                [[3.0, 3.0]],
                [[4.0, 4.0]],
                [[5.0, 5.0]],
                [[6.0, 6.0]],
            ],
        ]
    )
    inverse_spectrogram = audio.InverseSpectrogram(
        n_fft=16,
        win_length=16,
        hop_length=8,
        pad=0,
        window=WindowType.HANN,
        normalized=False,
        center=True,
        pad_mode=BorderType.SYMMETRIC,
        onesided=True,
        length=1,
    )
    inverse_spectrogram(waveform)

    # test InverseSpectrogram normal testcase: test pad mode constant
    waveform = np.array(
        [
            [
                [[1.0, 1.0]],
                [[2.0, 2.0]],
                [[3.0, 3.0]],
                [[4.0, 4.0]],
                [[5.0, 5.0]],
                [[6.0, 6.0]],
                [[5.0, 5.0]],
                [[4.0, 4.0]],
                [[3.0, 3.0]],
            ],
            [
                [[2.0, 2.0]],
                [[1.0, 1.0]],
                [[0.0, 0.0]],
                [[1.0, 1.0]],
                [[2.0, 2.0]],
                [[3.0, 3.0]],
                [[4.0, 4.0]],
                [[5.0, 5.0]],
                [[6.0, 6.0]],
            ],
        ]
    )
    inverse_spectrogram = audio.InverseSpectrogram(
        n_fft=16,
        win_length=16,
        hop_length=8,
        pad=0,
        window=WindowType.HANN,
        normalized=False,
        center=True,
        pad_mode=BorderType.CONSTANT,
        onesided=True,
        length=1,
    )
    inverse_spectrogram(waveform)

    # test InverseSpectrogram normal testcase: test normalized true
    waveform = np.array(
        [
            [
                [[1.0, 1.0]],
                [[2.0, 2.0]],
                [[3.0, 3.0]],
                [[4.0, 4.0]],
                [[5.0, 5.0]],
                [[6.0, 6.0]],
                [[5.0, 5.0]],
                [[4.0, 4.0]],
                [[3.0, 3.0]],
            ],
            [
                [[2.0, 2.0]],
                [[1.0, 1.0]],
                [[0.0, 0.0]],
                [[1.0, 1.0]],
                [[2.0, 2.0]],
                [[3.0, 3.0]],
                [[4.0, 4.0]],
                [[5.0, 5.0]],
                [[6.0, 6.0]],
            ],
        ]
    )
    inverse_spectrogram = audio.InverseSpectrogram(
        n_fft=16,
        win_length=16,
        hop_length=8,
        pad=0,
        window=WindowType.HANN,
        normalized=True,
        center=True,
        pad_mode=BorderType.REFLECT,
        onesided=True,
        length=1,
    )
    inverse_spectrogram(waveform)

    # test InverseSpectrogram normal testcase: test center false
    waveform = np.array(
        [
            [
                [[1.0, 1.0]],
                [[2.0, 2.0]],
                [[3.0, 3.0]],
                [[4.0, 4.0]],
                [[5.0, 5.0]],
                [[6.0, 6.0]],
                [[5.0, 5.0]],
                [[4.0, 4.0]],
                [[3.0, 3.0]],
            ],
            [
                [[2.0, 2.0]],
                [[1.0, 1.0]],
                [[0.0, 0.0]],
                [[1.0, 1.0]],
                [[2.0, 2.0]],
                [[3.0, 3.0]],
                [[4.0, 4.0]],
                [[5.0, 5.0]],
                [[6.0, 6.0]],
            ],
        ]
    )
    inverse_spectrogram = audio.InverseSpectrogram(
        n_fft=16,
        win_length=16,
        hop_length=8,
        pad=0,
        window=WindowType.HAMMING,
        normalized=False,
        center=False,
        pad_mode=BorderType.REFLECT,
        onesided=True,
        length=1,
    )
    inverse_spectrogram(waveform)

    # test InverseSpectrogram normal testcase: test onesided false
    waveform = np.array(
        [
            [
                [[1.0, 1.0]],
                [[2.0, 2.0]],
                [[3.0, 3.0]],
                [[4.0, 4.0]],
                [[5.0, 5.0]],
                [[6.0, 6.0]],
                [[5.0, 5.0]],
                [[4.0, 4.0]],
                [[3.0, 3.0]],
                [[2.0, 2.0]],
                [[1.0, 1.0]],
                [[0.0, 0.0]],
                [[1.0, 1.0]],
                [[2.0, 2.0]],
                [[3.0, 3.0]],
                [[4.0, 4.0]],
            ]
        ]
    )
    inverse_spectrogram = audio.InverseSpectrogram(
        n_fft=16,
        win_length=16,
        hop_length=8,
        pad=0,
        window=WindowType.HANN,
        normalized=False,
        center=True,
        pad_mode=BorderType.REFLECT,
        onesided=False,
        length=1,
    )
    inverse_spectrogram(waveform)

    # test InverseSpectrogram normal testcase: test other size 21x
    waveform = np.array(
        [
            [
                [
                    [[1.0, 1.0]],
                    [[2.0, 2.0]],
                    [[3.0, 3.0]],
                    [[4.0, 4.0]],
                    [[5.0, 5.0]],
                    [[6.0, 6.0]],
                    [[5.0, 5.0]],
                    [[4.0, 4.0]],
                    [[3.0, 3.0]],
                ]
            ],
            [
                [
                    [[2.0, 2.0]],
                    [[1.0, 1.0]],
                    [[0.0, 0.0]],
                    [[1.0, 1.0]],
                    [[2.0, 2.0]],
                    [[3.0, 3.0]],
                    [[4.0, 4.0]],
                    [[5.0, 5.0]],
                    [[6.0, 6.0]],
                ]
            ],
        ]
    )
    inverse_spectrogram = audio.InverseSpectrogram(
        n_fft=16,
        win_length=16,
        hop_length=8,
        pad=0,
        window=WindowType.HANN,
        normalized=False,
        center=True,
        pad_mode=BorderType.REFLECT,
        onesided=True,
        length=1,
    )
    inverse_spectrogram(waveform)

    # test InverseSpectrogram normal testcase: test other size 12x
    waveform = np.array(
        [
            [
                [
                    [[1.0, 1.0]],
                    [[2.0, 2.0]],
                    [[3.0, 3.0]],
                    [[4.0, 4.0]],
                    [[5.0, 5.0]],
                    [[6.0, 6.0]],
                    [[5.0, 5.0]],
                    [[4.0, 4.0]],
                    [[3.0, 3.0]],
                ],
                [
                    [[2.0, 2.0]],
                    [[1.0, 1.0]],
                    [[0.0, 0.0]],
                    [[1.0, 1.0]],
                    [[2.0, 2.0]],
                    [[3.0, 3.0]],
                    [[4.0, 4.0]],
                    [[5.0, 5.0]],
                    [[6.0, 6.0]],
                ],
            ]
        ]
    )
    inverse_spectrogram = audio.InverseSpectrogram(
        n_fft=16,
        win_length=16,
        hop_length=8,
        pad=0,
        window=WindowType.HANN,
        normalized=False,
        center=True,
        pad_mode=BorderType.REFLECT,
        onesided=True,
        length=1,
    )
    inverse_spectrogram(waveform)

    # test InverseSpectrogram normal testcase: test 3 deminsions
    waveform = np.array(
        [
            [[1.0, 1.0], [2.0, 2.0]],
            [[3.0, 3.0], [4.0, 4.0]],
            [[5.0, 5.0], [6.0, 6.0]],
            [[5.0, 5.0], [4.0, 4.0]],
            [[3.0, 3.0], [2.0, 2.0]],
            [[1.0, 1.0], [0.0, 0.0]],
            [[1.0, 1.0], [2.0, 2.0]],
            [[3.0, 3.0], [4.0, 4.0]],
            [[5.0, 5.0], [6.0, 6.0]],
        ]
    )
    inverse_spectrogram = audio.InverseSpectrogram(
        n_fft=16,
        win_length=16,
        hop_length=8,
        pad=0,
        window=WindowType.HANN,
        normalized=False,
        center=True,
        pad_mode=BorderType.REFLECT,
        onesided=True,
        length=1,
    )
    inverse_spectrogram(waveform)

    # test InverseSpectrogram normal testcase: test 4 deminsions
    waveform = np.array(
        [
            [
                [[1.0, 1.0]],
                [[2.0, 2.0]],
                [[3.0, 3.0]],
                [[4.0, 4.0]],
                [[5.0, 5.0]],
                [[6.0, 6.0]],
                [[5.0, 5.0]],
                [[4.0, 4.0]],
                [[3.0, 3.0]],
            ],
            [
                [[2.0, 2.0]],
                [[1.0, 1.0]],
                [[0.0, 0.0]],
                [[1.0, 1.0]],
                [[2.0, 2.0]],
                [[3.0, 3.0]],
                [[4.0, 4.0]],
                [[5.0, 5.0]],
                [[6.0, 6.0]],
            ],
        ]
    )
    inverse_spectrogram = audio.InverseSpectrogram(
        n_fft=16,
        win_length=16,
        hop_length=8,
        pad=0,
        window=WindowType.HANN,
        normalized=False,
        center=True,
        pad_mode=BorderType.REFLECT,
        onesided=True,
        length=1,
    )
    inverse_spectrogram(waveform)

    # test InverseSpectrogram normal testcase: test 5 deminsions
    waveform = np.array(
        [
            [
                [
                    [[1.0, 1.0]],
                    [[2.0, 2.0]],
                    [[3.0, 3.0]],
                    [[4.0, 4.0]],
                    [[5.0, 5.0]],
                    [[6.0, 6.0]],
                    [[5.0, 5.0]],
                    [[4.0, 4.0]],
                    [[3.0, 3.0]],
                ],
                [
                    [[2.0, 2.0]],
                    [[1.0, 1.0]],
                    [[0.0, 0.0]],
                    [[1.0, 1.0]],
                    [[2.0, 2.0]],
                    [[3.0, 3.0]],
                    [[4.0, 4.0]],
                    [[5.0, 5.0]],
                    [[6.0, 6.0]],
                ],
            ]
        ]
    )
    inverse_spectrogram = audio.InverseSpectrogram(
        n_fft=16,
        win_length=16,
        hop_length=8,
        pad=0,
        window=WindowType.HANN,
        normalized=False,
        center=True,
        pad_mode=BorderType.REFLECT,
        onesided=True,
        length=1,
    )
    inverse_spectrogram(waveform)

    # test InverseSpectrogram normal testcase: test 6 deminsions
    waveform = np.array(
        [
            [
                [
                    [
                        [[1.0, 1.0]],
                        [[2.0, 2.0]],
                        [[3.0, 3.0]],
                        [[4.0, 4.0]],
                        [[5.0, 5.0]],
                        [[6.0, 6.0]],
                        [[5.0, 5.0]],
                        [[4.0, 4.0]],
                        [[3.0, 3.0]],
                    ],
                    [
                        [[2.0, 2.0]],
                        [[1.0, 1.0]],
                        [[0.0, 0.0]],
                        [[1.0, 1.0]],
                        [[2.0, 2.0]],
                        [[3.0, 3.0]],
                        [[4.0, 4.0]],
                        [[5.0, 5.0]],
                        [[6.0, 6.0]],
                    ],
                ]
            ]
        ]
    )
    inverse_spectrogram = audio.InverseSpectrogram(
        n_fft=16,
        win_length=16,
        hop_length=8,
        pad=0,
        window=WindowType.HANN,
        normalized=False,
        center=True,
        pad_mode=BorderType.REFLECT,
        onesided=True,
        length=1,
    )
    inverse_spectrogram(waveform)

    # test InverseSpectrogram normal testcase: test 7 deminsions
    waveform = np.array(
        [
            [
                [
                    [
                        [
                            [[1.0, 1.0]],
                            [[2.0, 2.0]],
                            [[3.0, 3.0]],
                            [[4.0, 4.0]],
                            [[5.0, 5.0]],
                            [[6.0, 6.0]],
                            [[5.0, 5.0]],
                            [[4.0, 4.0]],
                            [[3.0, 3.0]],
                        ],
                        [
                            [[2.0, 2.0]],
                            [[1.0, 1.0]],
                            [[0.0, 0.0]],
                            [[1.0, 1.0]],
                            [[2.0, 2.0]],
                            [[3.0, 3.0]],
                            [[4.0, 4.0]],
                            [[5.0, 5.0]],
                            [[6.0, 6.0]],
                        ],
                    ]
                ]
            ]
        ]
    )
    inverse_spectrogram = audio.InverseSpectrogram(
        n_fft=16,
        win_length=16,
        hop_length=8,
        pad=0,
        window=WindowType.HANN,
        normalized=False,
        center=True,
        pad_mode=BorderType.REFLECT,
        onesided=True,
        length=1,
    )
    inverse_spectrogram(waveform)


def test_inverse_spectrogram_param_check():
    """
    Feature: InverseSpectrogram
    Description: Test InverseSpectrogram with invalid input data types and parameters
    Expectation: Correct error types and messages are raised as expected
    """
    # Description: test InverseSpectrogram invalid parameter: n_fft is -1
    try:
        audio.InverseSpectrogram(n_fft=-1)
    except ValueError as error:
        assert (
            "Input n_fft is not within the required interval of [1, 2147483647]."
            in str(error)
        )

    # Description: test InverseSpectrogram invalid parameter: n_fft is 0
    try:
        audio.InverseSpectrogram(n_fft=0)
    except ValueError as error:
        assert (
            "Input n_fft is not within the required interval of [1, 2147483647]."
            in str(error)
        )

    # Description:test InverseSpectrogram invalid parameter: win_length is -1
    try:
        audio.InverseSpectrogram(win_length=-1)
    except ValueError as error:
        assert (
            "Input win_length is not within the required interval of [1, 2147483647]."
            in str(error)
        )

    # Description:test InverseSpectrogram invalid parameter: win_length is s
    try:
        audio.InverseSpectrogram(win_length="s")
    except TypeError as error:
        assert (
            "Argument win_length with value s is not of type [<class 'int'>], but got <class 'str'>."
            in str(error)
        )

    # Description:test InverseSpectrogram invalid parameter: hop_length is -1
    try:
        audio.InverseSpectrogram(hop_length=-1)
    except ValueError as error:
        assert (
            "Input hop_length is not within the required interval of [1, 2147483647]."
            in str(error)
        )

    # Description:test test InverseSpectrogram invalid parameter: hop_length is -100
    try:
        audio.InverseSpectrogram(hop_length=-100)
    except ValueError as error:
        assert (
            "Input hop_length is not within the required interval of [1, 2147483647]."
            in str(error)
        )

    # Description:test test InverseSpectrogram invalid parameter: win_length large n_fft
    try:
        audio.InverseSpectrogram(win_length=300, n_fft=200)
    except ValueError as error:
        assert (
            "Input win_length should be no more than n_fft, but got win_length: 300 and n_fft: 200."
            in str(error)
        )

    # Description:test InverseSpectrogram invalid parameter: pad is -1
    try:
        audio.InverseSpectrogram(pad=-1)
    except ValueError as error:
        assert (
            "Input pad is not within the required interval of [0, 2147483647]."
            in str(error)
        )

    # Description:test InverseSpectrogram invalid parameter: n_fft is False
    try:
        audio.InverseSpectrogram(n_fft=False)
    except TypeError as error:
        assert "Argument n_fft with value False is not of type (<class 'int'>,)" in str(
            error
        )

    # Description:test InverseSpectrogram invalid parameter: n_fft is s
    try:
        audio.InverseSpectrogram(n_fft="s")
    except TypeError as error:
        assert (
            "Argument n_fft with value s is not of type [<class 'int'>], but got <class 'str'>."
            in str(error)
        )

    # Description:test InverseSpectrogram invalid parameter: window is False
    try:
        audio.InverseSpectrogram(window=False)
    except TypeError as error:
        assert (
            "Argument window with value False is not of type [<enum 'WindowType'>], but got <class 'bool'>."
            in str(error)
        )

    # Description:test InverseSpectrogram invalid parameter: pad_mode is False
    try:
        audio.InverseSpectrogram(pad_mode=False)
    except TypeError as error:
        assert (
            "Argument pad_mode with value False is not of type [<enum 'BorderType'>], but got <class 'bool'>."
            in str(error)
        )

    # Description:test InverseSpectrogram invalid parameter: onesided is GanJisong
    try:
        audio.InverseSpectrogram(onesided="GanJisong")
    except TypeError as error:
        assert (
            "Argument onesided with value GanJisong is not of type [<class 'bool'>], but got <class 'str'>."
            in str(error)
        )

    # Description:test InverseSpectrogram invalid parameter: center is MindSpore
    try:
        audio.InverseSpectrogram(center="MindSpore")
    except TypeError as error:
        assert (
            "Argument center with value MindSpore is not of type [<class 'bool'>], but got <class 'str'>."
            in str(error)
        )

    # Description:test InverseSpectrogram invalid parameter: normalized is s
    try:
        audio.InverseSpectrogram(normalized="s")
    except TypeError as error:
        assert (
            "Argument normalized with value s is not of type [<class 'bool'>], but got <class 'str'>."
            in str(error)
        )

    # Description:test InverseSpectrogram invalid parameter: normalized is 1
    try:
        audio.InverseSpectrogram(normalized=1)
    except TypeError as error:
        assert (
            "Argument normalized with value 1 is not of type [<class 'bool'>], but got <class 'int'>."
            in str(error)
        )

    # test InverseSpectrogram invalid parameter: win_length large n_fft
    try:
        waveform = np.array(
            [
                [
                    [[1.0, 1.0]],
                    [[2.0, 2.0]],
                    [[3.0, 3.0]],
                    [[4.0, 4.0]],
                    [[5.0, 5.0]],
                    [[6.0, 6.0]],
                    [[5.0, 5.0]],
                    [[4.0, 4.0]],
                    [[3.0, 3.0]],
                ],
                [
                    [[2.0, 2.0]],
                    [[1.0, 1.0]],
                    [[0.0, 0.0]],
                    [[1.0, 1.0]],
                    [[2.0, 2.0]],
                    [[3.0, 3.0]],
                    [[4.0, 4.0]],
                    [[5.0, 5.0]],
                    [[6.0, 6.0]],
                ],
            ]
        )
        audio.InverseSpectrogram(n_fft=100, win_length=400)(waveform)
    except ValueError as error:
        assert (
            "Input win_length should be no more than n_fft, but got win_length: 400 and n_fft: 100."
            in str(error)
        )


if __name__ == "__main__":
    test_inverse_spectrogram_pipeline()
    test_inverse_spectrogram_eager()
    test_inverse_spectrogram_param()
    test_inverse_spectrogram_transform()
    test_inverse_spectrogram_param_check()
