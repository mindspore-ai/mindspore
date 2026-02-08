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
"""Test PitchShift."""

import numpy as np

import mindspore.dataset as ds
from mindspore.dataset import audio
from mindspore.dataset.audio import WindowType
from . import count_unequal_element


def test_pitch_shift_pipeline():
    """
    Feature: Test pipeline mode normal testcase: PitchShift
    Description: Input audio signal to test pipeline
    Expectation: Generate expected output after cases were executed
    """
    waveform = [
        [
            [
                1,
                1,
                2,
                3,
                2,
                3,
                4,
                5,
                1,
                2,
                3,
                4,
                5,
                2,
                3,
                2,
                1,
                2,
                3,
                0,
                1,
                0,
                2,
                4,
                5,
                3,
                1,
                2,
                3,
                4,
            ]
        ]
    ]
    dataset = ds.NumpySlicesDataset(waveform, column_names=["audio"], shuffle=False)
    out = audio.PitchShift(
        sample_rate=16000,
        n_steps=4,
        bins_per_octave=12,
        n_fft=16,
        win_length=16,
        hop_length=4,
        window=WindowType.HANN,
    )

    dataset = dataset.map(
        operations=out, input_columns=["audio"], output_columns=["PitchShift"]
    )
    result = np.array(
        [
            [
                0.8897,
                1.0983,
                2.4355,
                1.8842,
                2.2082,
                3.6461,
                2.4232,
                1.7691,
                3.2835,
                3.3354,
                2.1773,
                3.3544,
                4.0488,
                3.1631,
                1.9124,
                2.2346,
                2.2417,
                3.6008,
                1.9539,
                1.3373,
                0.4311,
                2.0768,
                2.6538,
                1.5035,
                1.5668,
                2.3749,
                3.9702,
                3.5922,
                1.7618,
                1.2730,
            ]
        ]
    )
    for data1 in dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
        count_unequal_element(data1["PitchShift"], result, 0.0001, 0.0001)


def test_pitch_shift_eager():
    """
    Feature: Mindspore eager mode normal testcase: PitchShift
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
                    3,
                    2,
                    3,
                    4,
                    5,
                    1,
                    2,
                    3,
                    4,
                    5,
                    2,
                    3,
                    2,
                    1,
                    2,
                    3,
                    0,
                    1,
                    0,
                    2,
                    4,
                    5,
                    3,
                    1,
                    2,
                    3,
                    4,
                ]
            ]
        ]
    )
    out = audio.PitchShift(
        sample_rate=16000,
        n_steps=4,
        bins_per_octave=12,
        n_fft=16,
        win_length=16,
        hop_length=4,
        window=WindowType.HANN,
    )(waveform)
    result = np.array(
        [
            [
                [
                    0.8897,
                    1.0983,
                    2.4355,
                    1.8842,
                    2.2082,
                    3.6461,
                    2.4232,
                    1.7691,
                    3.2835,
                    3.3354,
                    2.1773,
                    3.3544,
                    4.0488,
                    3.1631,
                    1.9124,
                    2.2346,
                    2.2417,
                    3.6008,
                    1.9539,
                    1.3373,
                    0.4311,
                    2.0768,
                    2.6538,
                    1.5035,
                    1.5668,
                    2.3749,
                    3.9702,
                    3.5922,
                    1.7618,
                    1.2730,
                ]
            ]
        ]
    )
    count_unequal_element(out, result, 0.0001, 0.0001)


def test_pitch_shift_param():
    """
    Feature: Test PitchShift invalid parameter
    Description: Test some invalid parameters
    Expectation: throw ValueError, TypeError or RuntimeError exception
    """
    try:
        audio.PitchShift(sample_rate="s", n_steps=4)
    except TypeError as error:
        assert (
            "Argument sample_rate with value s is not of type [<class 'int'>], but got <class 'str'>."
            in str(error)
        )

    try:
        audio.PitchShift(sample_rate=-1, n_steps=4)
    except ValueError as error:
        assert (
            "Input sample_rate is not within the required interval of [0, 2147483647]."
            in str(error)
        )
    try:
        audio.PitchShift(n_steps="s", sample_rate=16)
    except TypeError as error:
        assert (
            "Argument n_steps with value s is not of type [<class 'int'>], but got <class 'str'>."
            in str(error)
        )
    try:
        audio.PitchShift(bins_per_octave=0, sample_rate=16, n_steps=4)
    except ValueError as error:
        assert (
            "Input bins_per_octave is not within the required interval of [-2147483648, 0) and (0, 2147483647]."
            in str(error)
        )
    try:
        audio.PitchShift(bins_per_octave="s", sample_rate=16, n_steps=4)
    except TypeError as error:
        assert (
            "Argument bins_per_octave  with value s is not of type [<class 'int'>], but got <class 'str'>."
            in str(error)
        )

    try:
        audio.PitchShift(n_fft=-1, sample_rate=16, n_steps=4)
    except ValueError as error:
        assert (
            "Input n_fft is not within the required interval of [1, 2147483647]."
            in str(error)
        )
    try:
        audio.PitchShift(n_fft=0, sample_rate=16, n_steps=4)
    except ValueError as error:
        assert (
            "Input n_fft is not within the required interval of [1, 2147483647]."
            in str(error)
        )
    try:
        audio.PitchShift(win_length=-1, sample_rate=16, n_steps=4)
    except ValueError as error:
        assert (
            "Input win_length is not within the required interval of [1, 2147483647]."
            in str(error)
        )
    try:
        audio.PitchShift(win_length="s", sample_rate=16, n_steps=4)
    except TypeError as error:
        assert (
            "Argument win_length with value s is not of type [<class 'int'>], but got <class 'str'>."
            in str(error)
        )
    try:
        audio.PitchShift(hop_length=-1, sample_rate=16, n_steps=4)
    except ValueError as error:
        assert (
            "Input hop_length is not within the required interval of [1, 2147483647]."
            in str(error)
        )
    try:
        audio.PitchShift(hop_length=-100, sample_rate=16, n_steps=4)
    except ValueError as error:
        assert (
            "Input hop_length is not within the required interval of [1, 2147483647]."
            in str(error)
        )
    try:
        audio.PitchShift(win_length=300, n_fft=200, sample_rate=16, n_steps=4)
    except ValueError as error:
        assert (
            "Input win_length should be no more than n_fft, but got win_length: 300 and n_fft: 200."
            in str(error)
        )
    try:
        audio.PitchShift(window=False, sample_rate=16, n_steps=4)
    except TypeError as error:
        assert (
            "Argument window with value False is not of type [<enum 'WindowType'>], but got <class 'bool'>."
            in str(error)
        )


def test_pitch_shift_transform():
    """
    Feature: Pitchshift
    Description: Test Pitchshift with various valid input parameters and data types
    Expectation: The operation completes successfully
    """
    waveform = [
        [
            [
                1,
                1,
                2,
                3,
                2,
                3,
                4,
                5,
                1,
                2,
                3,
                4,
                5,
                2,
                3,
                2,
                1,
                2,
                3,
                0,
                1,
                0,
                2,
                4,
                5,
                3,
                1,
                2,
                3,
                4,
            ]
        ]
    ]
    dataset = ds.NumpySlicesDataset(waveform, column_names=["audio"], shuffle=False)
    pitch_shift = audio.PitchShift(
        sample_rate=16000,
        n_steps=4,
        bins_per_octave=12,
        n_fft=16,
        win_length=16,
        hop_length=4,
        window=WindowType.HANN,
    )
    dataset = dataset.map(
        operations=pitch_shift, input_columns=["audio"], output_columns=["pitch_shift"]
    )
    dataset = dataset.project(columns=["pitch_shift"])
    result = np.array(
        [
            [
                0.8897,
                1.0983,
                2.4355,
                1.8842,
                2.2082,
                3.6461,
                2.4232,
                1.7691,
                3.2835,
                3.3354,
                2.1773,
                3.3544,
                4.0488,
                3.1631,
                1.9124,
                2.2346,
                2.2417,
                3.6008,
                1.9539,
                1.3373,
                0.4311,
                2.0768,
                2.6538,
                1.5035,
                1.5668,
                2.3749,
                3.9702,
                3.5922,
                1.7618,
                1.2730,
            ]
        ]
    )
    for data1 in dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
        count_unequal_element(data1["pitch_shift"], result, 0.0001, 0.0001)

    # test PitchShift eager mode normal
    waveform = np.array(
        [
            [
                [
                    1,
                    1,
                    2,
                    3,
                    2,
                    3,
                    4,
                    5,
                    1,
                    2,
                    3,
                    4,
                    5,
                    2,
                    3,
                    2,
                    1,
                    2,
                    3,
                    0,
                    1,
                    0,
                    2,
                    4,
                    5,
                    3,
                    1,
                    2,
                    3,
                    4,
                ]
            ]
        ]
    )
    pitch_shift = audio.PitchShift(
        sample_rate=16000,
        n_steps=4,
        bins_per_octave=12,
        n_fft=16,
        win_length=16,
        hop_length=4,
        window=WindowType.HANN,
    )
    pitch_shift(waveform)

    # test PitchShift normal testcase: test window hamming
    waveform = np.array(
        [
            [
                [
                    1,
                    1,
                    2,
                    3,
                    2,
                    3,
                    4,
                    5,
                    1,
                    2,
                    3,
                    4,
                    5,
                    2,
                    3,
                    2,
                    1,
                    2,
                    3,
                    0,
                    1,
                    0,
                    2,
                    4,
                    5,
                    3,
                    1,
                    2,
                    3,
                    4,
                ]
            ]
        ]
    )
    pitch_shift = audio.PitchShift(
        sample_rate=16000,
        n_steps=4,
        bins_per_octave=12,
        n_fft=16,
        win_length=16,
        hop_length=4,
        window=WindowType.HAMMING,
    )
    pitch_shift(waveform)

    # test PitchShift normal testcase: test window bartlett
    waveform = np.array(
        [
            [
                [
                    1,
                    1,
                    2,
                    3,
                    2,
                    3,
                    4,
                    5,
                    1,
                    2,
                    3,
                    4,
                    5,
                    2,
                    3,
                    2,
                    1,
                    2,
                    3,
                    0,
                    1,
                    0,
                    2,
                    4,
                    5,
                    3,
                    1,
                    2,
                    3,
                    4,
                ]
            ]
        ]
    )
    pitch_shift = audio.PitchShift(
        sample_rate=16000,
        n_steps=4,
        bins_per_octave=12,
        n_fft=16,
        win_length=16,
        hop_length=4,
        window=WindowType.BARTLETT,
    )
    pitch_shift(waveform)

    # test PitchShift normal testcase: window blackman
    waveform = np.array(
        [
            [
                [
                    1,
                    1,
                    2,
                    3,
                    2,
                    3,
                    4,
                    5,
                    1,
                    2,
                    3,
                    4,
                    5,
                    2,
                    3,
                    2,
                    1,
                    2,
                    3,
                    0,
                    1,
                    0,
                    2,
                    4,
                    5,
                    3,
                    1,
                    2,
                    3,
                    4,
                ]
            ]
        ]
    )
    pitch_shift = audio.PitchShift(
        sample_rate=16000,
        n_steps=4,
        bins_per_octave=12,
        n_fft=16,
        win_length=16,
        hop_length=4,
        window=WindowType.BLACKMAN,
    )
    pitch_shift(waveform)

    # test PitchShift normal testcase: test window kaiser
    waveform = np.array(
        [
            [
                [
                    1,
                    1,
                    2,
                    3,
                    2,
                    3,
                    4,
                    5,
                    1,
                    2,
                    3,
                    4,
                    5,
                    2,
                    3,
                    2,
                    1,
                    2,
                    3,
                    0,
                    1,
                    0,
                    2,
                    4,
                    5,
                    3,
                    1,
                    2,
                    3,
                    4,
                ]
            ]
        ]
    )
    pitch_shift = audio.PitchShift(
        sample_rate=16000,
        n_steps=4,
        bins_per_octave=12,
        n_fft=16,
        win_length=16,
        hop_length=4,
        window=WindowType.KAISER,
    )
    pitch_shift(waveform)

    # test PitchShift normal testcase: test other size 21x
    waveform = np.array(
        [
            [
                [
                    1,
                    1,
                    2,
                    3,
                    2,
                    3,
                    4,
                    5,
                    1,
                    2,
                    3,
                    4,
                    5,
                    2,
                    3,
                    2,
                    1,
                    2,
                    3,
                    0,
                    1,
                    0,
                    2,
                    4,
                    5,
                    3,
                    1,
                    2,
                    3,
                    4,
                ]
            ],
            [
                [
                    1,
                    1,
                    2,
                    3,
                    2,
                    3,
                    4,
                    5,
                    1,
                    2,
                    3,
                    4,
                    5,
                    2,
                    3,
                    2,
                    1,
                    2,
                    3,
                    0,
                    1,
                    0,
                    2,
                    4,
                    5,
                    3,
                    1,
                    2,
                    3,
                    4,
                ]
            ],
        ]
    )
    pitch_shift = audio.PitchShift(
        sample_rate=16000,
        n_steps=4,
        bins_per_octave=12,
        n_fft=16,
        win_length=16,
        hop_length=4,
        window=WindowType.HANN,
    )
    pitch_shift(waveform)

    # test PitchShift normal testcase: test other size 12x
    waveform = np.array(
        [
            [
                [
                    1,
                    1,
                    2,
                    3,
                    2,
                    3,
                    4,
                    5,
                    1,
                    2,
                    3,
                    4,
                    5,
                    2,
                    3,
                    2,
                    1,
                    2,
                    3,
                    0,
                    1,
                    0,
                    2,
                    4,
                    5,
                    3,
                    1,
                    2,
                    3,
                    4,
                ],
                [
                    1,
                    1,
                    2,
                    3,
                    2,
                    3,
                    4,
                    5,
                    1,
                    2,
                    3,
                    4,
                    5,
                    2,
                    3,
                    2,
                    1,
                    2,
                    3,
                    0,
                    1,
                    0,
                    2,
                    4,
                    5,
                    3,
                    1,
                    2,
                    3,
                    4,
                ],
            ]
        ]
    )
    pitch_shift = audio.PitchShift(
        sample_rate=16000,
        n_steps=4,
        bins_per_octave=12,
        n_fft=16,
        win_length=16,
        hop_length=4,
        window=WindowType.HANN,
    )
    pitch_shift(waveform)

    # test PitchShift normal testcase: test 1 deminsions
    waveform = np.array(
        [
            1,
            1,
            2,
            3,
            2,
            3,
            4,
            5,
            1,
            2,
            3,
            4,
            5,
            2,
            3,
            2,
            1,
            2,
            3,
            0,
            1,
            0,
            2,
            4,
            5,
            3,
            1,
            2,
            3,
            4,
        ]
    )
    pitch_shift = audio.PitchShift(
        sample_rate=16000,
        n_steps=4,
        bins_per_octave=12,
        n_fft=16,
        win_length=16,
        hop_length=4,
        window=WindowType.HANN,
    )
    pitch_shift(waveform)

    # test PitchShift normal testcase: test 2 deminsions
    waveform = np.array(
        [
            [
                1,
                1,
                2,
                3,
                2,
                3,
                4,
                5,
                1,
                2,
                3,
                4,
                5,
                2,
                3,
                2,
                1,
                2,
                3,
                0,
                1,
                0,
                2,
                4,
                5,
                3,
                1,
                2,
                3,
                4,
            ]
        ]
    )
    pitch_shift = audio.PitchShift(
        sample_rate=16000,
        n_steps=4,
        bins_per_octave=12,
        n_fft=16,
        win_length=16,
        hop_length=4,
        window=WindowType.HANN,
    )
    pitch_shift(waveform)

    # test PitchShift normal testcase: test 4 deminsions
    waveform = np.array(
        [
            [
                [
                    [
                        1,
                        1,
                        2,
                        3,
                        2,
                        3,
                        4,
                        5,
                        1,
                        2,
                        3,
                        4,
                        5,
                        2,
                        3,
                        2,
                        1,
                        2,
                        3,
                        0,
                        1,
                        0,
                        2,
                        4,
                        5,
                        3,
                        1,
                        2,
                        3,
                        4,
                    ]
                ]
            ]
        ]
    )
    pitch_shift = audio.PitchShift(
        sample_rate=16000,
        n_steps=4,
        bins_per_octave=12,
        n_fft=16,
        win_length=16,
        hop_length=4,
        window=WindowType.HANN,
    )
    pitch_shift(waveform)

    # test PitchShift normal testcase: test 5 deminsions
    waveform = np.array(
        [
            [
                [
                    [
                        [
                            1,
                            1,
                            2,
                            3,
                            2,
                            3,
                            4,
                            5,
                            1,
                            2,
                            3,
                            4,
                            5,
                            2,
                            3,
                            2,
                            1,
                            2,
                            3,
                            0,
                            1,
                            0,
                            2,
                            4,
                            5,
                            3,
                            1,
                            2,
                            3,
                            4,
                        ]
                    ]
                ]
            ]
        ]
    )
    pitch_shift = audio.PitchShift(
        sample_rate=16000,
        n_steps=4,
        bins_per_octave=12,
        n_fft=16,
        win_length=16,
        hop_length=4,
        window=WindowType.HANN,
    )
    pitch_shift(waveform)

    # test PitchShift normal testcase: test 6 deminsions
    waveform = np.array(
        [[[[[[1, 1, 2, 3, 2, 3, 4, 5, 1, 2, 3, 4, 5, 2, 3, 2, 1, 2]]]]]]
    )
    pitch_shift = audio.PitchShift(
        sample_rate=16000,
        n_steps=4,
        bins_per_octave=12,
        n_fft=16,
        win_length=16,
        hop_length=4,
        window=WindowType.HANN,
    )
    pitch_shift(waveform)

    # test PitchShift normal testcase: test 7 deminsions
    waveform = np.array(
        [[[[[[[1, 1, 2, 3, 2, 3, 4, 5, 1, 2, 3, 4, 5, 2, 3, 2, 1, 2]]]]]]]
    )
    pitch_shift = audio.PitchShift(
        sample_rate=16000,
        n_steps=4,
        bins_per_octave=12,
        n_fft=16,
        win_length=16,
        hop_length=4,
        window=WindowType.HANN,
    )
    pitch_shift(waveform)


def test_pitch_shift_param_check():
    """
    Feature: Pitchshift
    Description: Test Pitchshift with invalid input data types and parameters
    Expectation: Correct error types and messages are raised as expected
    """
    try:
        audio.PitchShift(sample_rate="s", n_steps=4)
    except TypeError as error:
        assert (
            "Argument sample_rate with value s is not of type [<class 'int'>], but got <class 'str'>."
            in str(error)
        )

    # test PitchShift invalid parameter: sample_rate is -1
    try:
        audio.PitchShift(sample_rate=-1, n_steps=4)
    except ValueError as error:
        assert (
            "Input sample_rate is not within the required interval of [0, 2147483647]."
            in str(error)
        )

    # test PitchShift invalid parameter: n_steps is s
    try:
        audio.PitchShift(n_steps="s", sample_rate=16)
    except TypeError as error:
        assert (
            "Argument n_steps with value s is not of type [<class 'int'>], but got <class 'str'>."
            in str(error)
        )

    # test PitchShift invalid parameter: bins_per_octave is 0
    try:
        audio.PitchShift(bins_per_octave=0, sample_rate=16, n_steps=4)
    except ValueError as error:
        assert (
            r"Input bins_per_octave is not within the required interval of [-2147483648, 0) "
            r"and (0, 2147483647]." in str(error)
        )

    # test PitchShift invalid parameter: bins_per_octave is s
    try:
        audio.PitchShift(bins_per_octave="s", sample_rate=16, n_steps=4)
    except TypeError as error:
        assert (
            "Argument bins_per_octave  with value s is not of type [<class 'int'>], but got <class 'str'>."
            in str(error)
        )

    # test PitchShift invalid parameter: n_fft is -1
    try:
        audio.PitchShift(n_fft=-1, sample_rate=16, n_steps=4)
    except ValueError as error:
        assert (
            "Input n_fft is not within the required interval of [1, 2147483647]."
            in str(error)
        )

    # test PitchShift invalid parameter: n_fft is 0
    try:
        audio.PitchShift(n_fft=0, sample_rate=16, n_steps=4)
    except ValueError as error:
        assert (
            "Input n_fft is not within the required interval of [1, 2147483647]."
            in str(error)
        )

    # test PitchShift invalid parameter: win_length is -1
    try:
        audio.PitchShift(win_length=-1, sample_rate=16, n_steps=4)
    except ValueError as error:
        assert (
            "Input win_length is not within the required interval of [1, 2147483647]."
            in str(error)
        )

    # test PitchShift invalid parameter: win_length is s
    try:
        audio.PitchShift(win_length="s", sample_rate=16, n_steps=4)
    except TypeError as error:
        assert (
            "Argument win_length with value s is not of type [<class 'int'>], but got <class 'str'>."
            in str(error)
        )

    # test PitchShift invalid parameter: hop_length is -1
    try:
        audio.PitchShift(hop_length=-1, sample_rate=16, n_steps=4)
    except ValueError as error:
        assert (
            "Input hop_length is not within the required interval of [1, 2147483647]."
            in str(error)
        )

    # test PitchShift invalid parameter: hop_length is -100
    try:
        audio.PitchShift(hop_length=-100, sample_rate=16, n_steps=4)
    except ValueError as error:
        assert (
            "Input hop_length is not within the required interval of [1, 2147483647]."
            in str(error)
        )

    # test PitchShift invalid parameter: win_length large n_fft
    try:
        audio.PitchShift(win_length=300, n_fft=200, sample_rate=16, n_steps=4)
    except ValueError as error:
        assert (
            "Input win_length should be no more than n_fft, but got win_length: 300 and n_fft: 200."
            in str(error)
        )

    # test PitchShift invalid parameter: window is False
    try:
        audio.PitchShift(window=False, sample_rate=16, n_steps=4)
    except TypeError as error:
        assert (
            "Argument window with value False is not of type [<enum 'WindowType'>], but got <class 'bool'>."
            in str(error)
        )


if __name__ == "__main__":
    test_pitch_shift_pipeline()
    test_pitch_shift_eager()
    test_pitch_shift_param()
    test_pitch_shift_transform()
    test_pitch_shift_param_check()
