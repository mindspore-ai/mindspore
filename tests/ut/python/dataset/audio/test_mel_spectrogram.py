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
"""Test MelSpectrogram."""

import numpy as np

import mindspore.dataset as ds
from mindspore.dataset import audio
from mindspore.dataset.audio import WindowType, BorderType, NormType, MelType
from . import count_unequal_element


def test_mel_spectrogram_pipeline():
    """
    Feature: Test pipeline mode normal testcase: MelSpectrogram
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
    output = audio.MelSpectrogram(
        sample_rate=16000,
        n_fft=16,
        win_length=16,
        hop_length=8,
        f_min=0.0,
        f_max=10000.0,
        pad=0,
        n_mels=8,
        window=WindowType.HANN,
        power=2.0,
        normalized=False,
        center=True,
        pad_mode=BorderType.REFLECT,
        onesided=True,
        norm=NormType.NONE,
        mel_scale=MelType.HTK,
    )
    dataset = dataset.map(
        operations=output, input_columns=["audio"], output_columns=["MelSpectrogram"]
    )
    result = np.array(
        [
            [
                [0.0000e00, 0.0000e00, 0.0000e00, 0.0000e00],
                [1.6105e00, 2.8416e01, 4.0224e00, 1.4698e01],
                [1.8027e01, 3.1808e02, 4.5026e01, 1.6452e02],
                [7.9213e00, 8.4180e00, 5.6739e00, 2.2122e00],
                [6.0452e00, 6.5609e00, 4.5775e00, 1.8347e00],
                [5.6763e-01, 9.4627e-01, 6.4849e-01, 3.0038e-01],
                [3.1647e-01, 1.2753e00, 7.9531e-01, 1.7264e-01],
                [2.6995e00, 2.0453e00, 2.6940e00, 3.5556e00],
            ]
        ]
    )
    for data1 in dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
        count_unequal_element(data1["MelSpectrogram"], result, 0.0001, 0.0001)


def test_mel_spectrogram_eager():
    """
    Feature: Test eager mode normal testcase: MelSpectrogram
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
    output = audio.MelSpectrogram(
        sample_rate=16000,
        n_fft=16,
        win_length=16,
        hop_length=8,
        f_min=0.0,
        f_max=5000.0,
        pad=0,
        n_mels=8,
        window=WindowType.HANN,
        power=2.0,
        normalized=False,
        center=True,
        pad_mode=BorderType.REFLECT,
        onesided=True,
        norm=NormType.NONE,
        mel_scale=MelType.HTK,
    )(waveform)
    result = np.array(
        [
            [
                [
                    [0.0000e00, 0.0000e00, 0.0000e00, 0.0000e00],
                    [0.0000e00, 0.0000e00, 0.0000e00, 0.0000e00],
                    [4.1355e00, 7.2968e01, 1.0329e01, 3.7741e01],
                    [1.5502e01, 2.7353e02, 3.8720e01, 1.4148e02],
                    [3.0792e00, 3.2723e00, 2.2056e00, 8.5993e-01],
                    [1.0531e01, 1.1192e01, 7.5435e00, 2.9411e00],
                    [5.6983e-01, 8.2424e-01, 8.0414e-01, 3.9367e-01],
                    [3.7583e-01, 6.6152e-01, 3.4257e-01, 1.6248e-01],
                ]
            ]
        ]
    )
    count_unequal_element(output, result, 0.0001, 0.0001)


def test_mel_spectrogram_param():
    """
    Feature: Test MelSpectrogram invalid parameter
    Description: Test some invalid parameters
    Expectation: throw ValueError, TypeError or RuntimeError exception
    """
    try:
        audio.MelSpectrogram(sample_rate=-1)
    except ValueError as error:
        assert (
            "Input sample_rate is not within the required interval of [0, 2147483647]."
            in str(error)
        )
    try:
        audio.MelSpectrogram(sample_rate="s")
    except TypeError as error:
        assert (
            "Argument sample_rate with value s is not of type [<class 'int'>], but got <class 'str'>."
            in str(error)
        )
    try:
        audio.MelSpectrogram(f_max=-1)
    except ValueError as error:
        assert (
            "Input f_max is not within the required interval of [0, 16777216]."
            in str(error)
        )
    try:
        audio.MelSpectrogram(f_min=-1.0)
    except ValueError as error:
        assert (
            "Input f_min is not within the required interval of (0, 16777216]."
            in str(error)
        )
    try:
        audio.MelSpectrogram(norm=-1)
    except TypeError as error:
        assert (
            "Argument norm with value -1 is not of type [<enum 'NormType'>], but got <class 'int'>."
            in str(error)
        )
    try:
        audio.MelSpectrogram(mel_scale=-1)
    except TypeError as error:
        assert (
            "Argument mel_type with value -1 is not of type [<enum 'MelType'>], but got <class 'int'>."
            in str(error)
        )
    try:
        audio.MelSpectrogram(n_fft=-1)
    except ValueError as error:
        assert (
            "Input n_fft is not within the required interval of [1, 2147483647]."
            in str(error)
        )
    try:
        audio.MelSpectrogram(n_fft=0)
    except ValueError as error:
        assert (
            "Input n_fft is not within the required interval of [1, 2147483647]."
            in str(error)
        )
    try:
        audio.MelSpectrogram(win_length=-1)
    except ValueError as error:
        assert (
            "Input win_length is not within the required interval of [1, 2147483647]."
            in str(error)
        )
    try:
        audio.MelSpectrogram(win_length="s")
    except TypeError as error:
        assert (
            "Argument win_length with value s is not of type [<class 'int'>], but got <class 'str'>."
            in str(error)
        )
    try:
        audio.MelSpectrogram(hop_length=-1)
    except ValueError as error:
        assert (
            "Input hop_length is not within the required interval of [1, 2147483647]."
            in str(error)
        )
    try:
        audio.MelSpectrogram(hop_length=-100)
    except ValueError as error:
        assert (
            "Input hop_length is not within the required interval of [1, 2147483647]."
            in str(error)
        )
    try:
        audio.MelSpectrogram(win_length=300, n_fft=200)
    except ValueError as error:
        assert (
            "Input win_length should be no more than n_fft, but got win_length: 300 and n_fft: 200."
            in str(error)
        )
    try:
        audio.MelSpectrogram(pad=-1)
    except ValueError as error:
        assert (
            "Input pad is not within the required interval of [0, 2147483647]."
            in str(error)
        )
    try:
        audio.MelSpectrogram(power=-1)
    except ValueError as error:
        assert (
            "Input power is not within the required interval of (0, 16777216]."
            in str(error)
        )
    try:
        audio.MelSpectrogram(n_fft=False)
    except TypeError as error:
        assert "Argument n_fft with value False is not of type (<class 'int'>,)" in str(
            error
        )
    try:
        audio.MelSpectrogram(n_fft="s")
    except TypeError as error:
        assert (
            "Argument n_fft with value s is not of type [<class 'int'>], but got <class 'str'>."
            in str(error)
        )
    try:
        audio.MelSpectrogram(window=False)
    except TypeError as error:
        assert (
            "Argument window with value False is not of type [<enum 'WindowType'>], but got <class 'bool'>."
            in str(error)
        )
    try:
        audio.MelSpectrogram(pad_mode=False)
    except TypeError as error:
        assert (
            "Argument pad_mode with value False is not of type [<enum 'BorderType'>], but got <class 'bool'>."
            in str(error)
        )
    try:
        audio.MelSpectrogram(onesided="LianLinghang")
    except TypeError as error:
        assert (
            "Argument onesided with value LianLinghang is not of type [<class 'bool'>], but got <class 'str'>."
            in str(error)
        )
    try:
        audio.MelSpectrogram(center="XiaDanni")
    except TypeError as error:
        assert (
            "Argument center with value XiaDanni is not of type [<class 'bool'>], but got <class 'str'>."
            in str(error)
        )
    try:
        audio.MelSpectrogram(normalized="s")
    except TypeError as error:
        assert (
            "Argument normalized with value s is not of type [<class 'bool'>], but got <class 'str'>."
            in str(error)
        )
    try:
        audio.MelSpectrogram(normalized=1)
    except TypeError as error:
        assert (
            "Argument normalized with value 1 is not of type [<class 'bool'>], but got <class 'int'>."
            in str(error)
        )
    try:
        audio.MelSpectrogram(f_max=1.0, f_min=2.0, sample_rate=16000)
    except ValueError as error:
        assert (
            "f_max should be greater than or equal to f_min, but got f_min: 2.0 and f_max: 1.0."
            in str(error)
        )
    try:
        audio.MelSpectrogram(f_min=60.0, f_max=None, sample_rate=100)
    except ValueError as error:
        assert (
            "MelSpectrogram: sample_rate // 2 should be greater than f_min when f_max is set to None, "
            "but got f_min: 60.0." in str(error)
        )


def test_mel_spectrogram_transform():
    """
    Feature: MelSpectrogramOps
    Description: Test MelSpectrogramOps with various valid input parameters and data types
    Expectation: The operation completes successfully
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
    output = audio.MelSpectrogram(
        sample_rate=16000,
        n_fft=16,
        win_length=16,
        hop_length=8,
        f_min=0.0,
        f_max=5000.0,
        pad=0,
        n_mels=8,
        window=WindowType.HANN,
        power=2.0,
        normalized=False,
        center=True,
        pad_mode=BorderType.REFLECT,
        onesided=True,
        norm=NormType.NONE,
        mel_scale=MelType.HTK,
    )(waveform)
    expected_result = np.array(
        [
            [
                [
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [4.1355, 72.968216, 10.32906, 37.741383],
                    [15.502378, 273.52942, 38.71962, 141.47775],
                    [3.079225, 3.2723322, 2.2056212, 0.85993207],
                    [10.531313, 11.191762, 7.543485, 2.9410691],
                    [0.56982964, 0.8242375, 0.804139, 0.3936723],
                    [0.3758252, 0.66152155, 0.34256887, 0.16248375],
                ]
            ]
        ],
        dtype=np.float32,
    )
    count_unequal_element(output, expected_result, 0.0001, 0.0001)

    # Description: Input audio signal to test eager.
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
    output = audio.MelSpectrogram(
        sample_rate=16000,
        n_fft=16,
        win_length=16,
        hop_length=8,
        f_min=0.0,
        f_max=5000.0,
        pad=0,
        n_mels=8,
        window=WindowType.HAMMING,
        power=2.0,
        normalized=False,
        center=True,
        pad_mode=BorderType.REFLECT,
        onesided=True,
        norm=NormType.NONE,
        mel_scale=MelType.HTK,
    )(waveform)
    expected_result = np.array(
        [
            [
                [
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [1.9282618, 69.6696, 9.697615, 37.459522],
                    [7.228302, 261.16418, 36.35258, 140.42116],
                    [2.606256, 2.769702, 1.6024503, 0.4044047],
                    [8.913703, 9.472707, 5.4805684, 1.3831117],
                    [0.6646495, 0.96139014, 0.68523324, 0.20461725],
                    [0.34568316, 0.59981084, 0.5114038, 0.10528136],
                ]
            ]
        ],
        dtype=np.float32,
    )
    count_unequal_element(output, expected_result, 0.0001, 0.0001)

    # Description: Input audio signal to test window bartlett.
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
    output = audio.MelSpectrogram(
        sample_rate=16000,
        n_fft=16,
        win_length=16,
        hop_length=8,
        f_min=0.0,
        f_max=5000.0,
        pad=0,
        n_mels=8,
        window=WindowType.BARTLETT,
        power=2.0,
        normalized=False,
        center=True,
        pad_mode=BorderType.REFLECT,
        onesided=True,
        norm=NormType.NONE,
        mel_scale=MelType.HTK,
    )(waveform)
    expected_result = np.array(
        [
            [
                [
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [1.3116374, 57.146717, 8.19836, 31.084433],
                    [4.9168177, 214.22078, 30.73246, 116.52343],
                    [2.637227, 2.7503462, 1.1162192, 0.08241335],
                    [9.019628, 9.406508, 3.8176012, 0.2818634],
                    [0.10575772, 3.8158412, 2.5755224, 0.93568146],
                    [0.86002606, 1.6322781, 0.44236544, 0.40530437],
                ]
            ]
        ],
        dtype=np.float32,
    )
    count_unequal_element(output, expected_result, 0.0001, 0.0001)

    # Description: Input audio signal to test window blackman.
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
    output = audio.MelSpectrogram(
        sample_rate=16000,
        n_fft=16,
        win_length=16,
        hop_length=8,
        f_min=0.0,
        f_max=5000.0,
        pad=0,
        n_mels=8,
        window=WindowType.BLACKMAN,
        power=2.0,
        normalized=False,
        center=True,
        pad_mode=BorderType.REFLECT,
        onesided=True,
        norm=NormType.NONE,
        mel_scale=MelType.HTK,
    )(waveform)
    expected_result = np.array(
        [
            [
                [
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [5.04003, 69.27671, 8.326068, 31.834158],
                    [18.89311, 259.6914, 31.211185, 119.333855],
                    [0.84205693, 7.3113894, 1.8502499, 2.0212405],
                    [2.8799338, 25.005814, 6.328073, 6.912881],
                    [1.324332, 1.6395636, 0.49636307, 0.00890297],
                    [0.489963, 0.7848614, 0.27410823, 0.01202468],
                ]
            ]
        ],
        dtype=np.float32,
    )
    count_unequal_element(output, expected_result, 0.0001, 0.0001)

    # Description: Input audio signal to test other size 21x.
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
            ],
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
            ],
        ]
    )
    output = audio.MelSpectrogram(
        sample_rate=16000,
        n_fft=16,
        win_length=16,
        hop_length=8,
        f_min=0.0,
        f_max=5000.0,
        pad=0,
        n_mels=8,
        window=WindowType.HANN,
        power=2.0,
        normalized=False,
        center=True,
        pad_mode=BorderType.REFLECT,
        onesided=True,
        norm=NormType.NONE,
        mel_scale=MelType.HTK,
    )(waveform)
    expected_result = np.array(
        [
            [
                [
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [4.1355, 72.968216, 10.32906, 37.741383],
                    [15.502378, 273.52942, 38.71962, 141.47775],
                    [3.079225, 3.2723322, 2.2056212, 0.85993207],
                    [10.531313, 11.191762, 7.543485, 2.9410691],
                    [0.56982964, 0.8242375, 0.804139, 0.3936723],
                    [0.3758252, 0.66152155, 0.34256887, 0.16248375],
                ]
            ],
            [
                [
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [4.1355, 72.968216, 10.32906, 37.741383],
                    [15.502378, 273.52942, 38.71962, 141.47775],
                    [3.079225, 3.2723322, 2.2056212, 0.85993207],
                    [10.531313, 11.191762, 7.543485, 2.9410691],
                    [0.56982964, 0.8242375, 0.804139, 0.3936723],
                    [0.3758252, 0.66152155, 0.34256887, 0.16248375],
                ]
            ],
        ],
        dtype=np.float32,
    )
    count_unequal_element(output, expected_result, 0.0001, 0.0001)

    # Description: Input audio signal to test other size 12x.
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
                ],
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
                ],
            ]
        ]
    )
    output = audio.MelSpectrogram(
        sample_rate=16000,
        n_fft=16,
        win_length=16,
        hop_length=8,
        f_min=0.0,
        f_max=5000.0,
        pad=0,
        n_mels=8,
        window=WindowType.HANN,
        power=2.0,
        normalized=False,
        center=True,
        pad_mode=BorderType.REFLECT,
        onesided=True,
        norm=NormType.NONE,
        mel_scale=MelType.HTK,
    )(waveform)
    expected_result = np.array(
        [
            [
                [
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [4.1355, 72.968216, 10.32906, 37.741383],
                    [15.502378, 273.52942, 38.71962, 141.47775],
                    [3.079225, 3.2723322, 2.2056212, 0.85993207],
                    [10.531313, 11.191762, 7.543485, 2.9410691],
                    [0.56982964, 0.8242375, 0.804139, 0.3936723],
                    [0.3758252, 0.66152155, 0.34256887, 0.16248375],
                ],
                [
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [4.1355, 72.968216, 10.32906, 37.741383],
                    [15.502378, 273.52942, 38.71962, 141.47775],
                    [3.079225, 3.2723322, 2.2056212, 0.85993207],
                    [10.531313, 11.191762, 7.543485, 2.9410691],
                    [0.56982964, 0.8242375, 0.804139, 0.3936723],
                    [0.3758252, 0.66152155, 0.34256887, 0.16248375],
                ],
            ]
        ],
        dtype=np.float32,
    )
    count_unequal_element(output, expected_result, 0.0001, 0.0001)

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
    output = audio.MelSpectrogram(
        sample_rate=16000,
        n_fft=16,
        win_length=16,
        hop_length=8,
        f_min=0.0,
        f_max=5000.0,
        pad=0,
        n_mels=8,
        window=WindowType.HANN,
        power=2.0,
        normalized=False,
        center=True,
        pad_mode=BorderType.REFLECT,
        onesided=True,
        norm=NormType.NONE,
        mel_scale=MelType.HTK,
    )(waveform)
    expected_result = np.array(
        [
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [4.1355, 72.968216, 10.32906, 37.741383],
            [15.502378, 273.52942, 38.71962, 141.47775],
            [3.079225, 3.2723322, 2.2056212, 0.85993207],
            [10.531313, 11.191762, 7.543485, 2.9410691],
            [0.56982964, 0.8242375, 0.804139, 0.3936723],
            [0.3758252, 0.66152155, 0.34256887, 0.16248375],
        ],
        dtype=np.float32,
    )
    count_unequal_element(output, expected_result, 0.0001, 0.0001)

    # Description: Input audio signal to test 2 deminsions.
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
    output = audio.MelSpectrogram(
        sample_rate=16000,
        n_fft=16,
        win_length=16,
        hop_length=8,
        f_min=0.0,
        f_max=5000.0,
        pad=0,
        n_mels=8,
        window=WindowType.HANN,
        power=2.0,
        normalized=False,
        center=True,
        pad_mode=BorderType.REFLECT,
        onesided=True,
        norm=NormType.NONE,
        mel_scale=MelType.HTK,
    )(waveform)
    expected_result = np.array(
        [
            [
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [4.1355, 72.968216, 10.32906, 37.741383],
                [15.502378, 273.52942, 38.71962, 141.47775],
                [3.079225, 3.2723322, 2.2056212, 0.85993207],
                [10.531313, 11.191762, 7.543485, 2.9410691],
                [0.56982964, 0.8242375, 0.804139, 0.3936723],
                [0.3758252, 0.66152155, 0.34256887, 0.16248375],
            ]
        ],
        dtype=np.float32,
    )
    count_unequal_element(output, expected_result, 0.0001, 0.0001)

    # Description: Input audio signal to test 6 deminsions.
    waveform = np.array(
        [[[[[[1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 4, 4, 3, 3, 2, 2, 1, 1]]]]]]
    )
    output = audio.MelSpectrogram(
        sample_rate=16000,
        n_fft=16,
        win_length=16,
        hop_length=8,
        f_min=0.0,
        f_max=5000.0,
        pad=0,
        n_mels=8,
        window=WindowType.HANN,
        power=2.0,
        normalized=False,
        center=True,
        pad_mode=BorderType.REFLECT,
        onesided=True,
        norm=NormType.NONE,
        mel_scale=MelType.HTK,
    )(waveform)
    expected_result = np.array(
        [
            [
                [
                    [
                        [
                            [
                                [0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0],
                                [4.1355, 72.968216, 6.4519672],
                                [15.502378, 273.52942, 24.185913],
                                [3.079225, 3.2723322, 2.9029555],
                                [10.531313, 11.191762, 9.928451],
                                [0.56982964, 0.8242375, 0.56982976],
                                [0.3758252, 0.66152155, 0.33962244],
                            ]
                        ]
                    ]
                ]
            ]
        ],
        dtype=np.float32,
    )
    count_unequal_element(output, expected_result, 0.0001, 0.0001)

    # Description: Input audio signal to test 7 deminsions.
    waveform = np.array(
        [[[[[[[1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 4, 4, 3, 3, 2, 2, 1, 1]]]]]]]
    )
    output = audio.MelSpectrogram(
        sample_rate=16000,
        n_fft=16,
        win_length=16,
        hop_length=8,
        f_min=0.0,
        f_max=5000.0,
        pad=0,
        n_mels=8,
        window=WindowType.HANN,
        power=2.0,
        normalized=False,
        center=True,
        pad_mode=BorderType.REFLECT,
        onesided=True,
        norm=NormType.NONE,
        mel_scale=MelType.HTK,
    )(waveform)
    expected_result = np.array(
        [
            [
                [
                    [
                        [
                            [
                                [
                                    [0.0, 0.0, 0.0],
                                    [0.0, 0.0, 0.0],
                                    [4.1355, 72.968216, 6.4519672],
                                    [15.502378, 273.52942, 24.185913],
                                    [3.079225, 3.2723322, 2.9029555],
                                    [10.531313, 11.191762, 9.928451],
                                    [0.56982964, 0.8242375, 0.56982976],
                                    [0.3758252, 0.66152155, 0.33962244],
                                ]
                            ]
                        ]
                    ]
                ]
            ]
        ],
        dtype=np.float32,
    )
    count_unequal_element(output, expected_result, 0.0001, 0.0001)

    # Description: Input audio signal to test pipeline.
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
    output = audio.MelSpectrogram(
        sample_rate=16000,
        n_fft=16,
        win_length=16,
        hop_length=8,
        f_min=0.0,
        f_max=10000.0,
        pad=0,
        n_mels=8,
        window=WindowType.HANN,
        power=2.0,
        normalized=False,
        center=True,
        pad_mode=BorderType.REFLECT,
        onesided=True,
        norm=NormType.NONE,
        mel_scale=MelType.HTK,
    )
    dataset = dataset.map(
        operations=output, input_columns=["audio"], output_columns=["MelSpectrogram"]
    )
    dataset = dataset.project(columns=["MelSpectrogram"])
    result = np.array(
        [
            [
                [0.0000e00, 0.0000e00, 0.0000e00, 0.0000e00],
                [1.6105e00, 2.8416e01, 4.0224e00, 1.4698e01],
                [1.8027e01, 3.1808e02, 4.5026e01, 1.6452e02],
                [7.9213e00, 8.4180e00, 5.6739e00, 2.2122e00],
                [6.0452e00, 6.5609e00, 4.5775e00, 1.8347e00],
                [5.6763e-01, 9.4627e-01, 6.4849e-01, 3.0038e-01],
                [3.1647e-01, 1.2753e00, 7.9531e-01, 1.7264e-01],
                [2.6995e00, 2.0453e00, 2.6940e00, 3.5556e00],
            ]
        ]
    )
    for data1 in dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
        count_unequal_element(data1["MelSpectrogram"], result, 0.0001, 0.0001)

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
    output = audio.MelSpectrogram(
        sample_rate=16000,
        n_fft=16,
        win_length=16,
        hop_length=8,
        f_min=0.0,
        f_max=10000.0,
        pad=0,
        n_mels=8,
        power=10.0,
    )(waveform)
    expected_result = np.array(
        [
            [
                [
                    [0.00000000e00, 0.00000000e00, 0.00000000e00, 0.00000000e00],
                    [2.39516203e05, 4.09603146e11, 2.32808360e07, 1.51629527e10],
                    [2.68108925e06, 4.58500342e12, 2.60600320e08, 1.69730605e11],
                    [2.71828875e05, 3.68447406e05, 5.12558359e04, 4.61752350e02],
                    [1.95235156e05, 2.64629625e05, 3.68137422e04, 3.31653839e02],
                    [5.91365881e-02, 4.30786401e-01, 3.12383413e-01, 8.76839459e-03],
                    [6.09893771e-03, 2.22022939e00, 1.28320563e00, 2.64250100e-01],
                    [1.82062180e02, 1.61950150e01, 1.55059185e01, 3.44718353e02],
                ]
            ]
        ],
        dtype=np.float32,
    )
    count_unequal_element(output, expected_result, 0.0001, 0.0001)

    # Description: power equal to  0.1
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
    output = audio.MelSpectrogram(
        sample_rate=16000,
        n_fft=16,
        win_length=16,
        hop_length=8,
        f_min=0.0,
        f_max=10000.0,
        pad=0,
        n_mels=8,
        power=0.1,
    )(waveform)
    expected_result = np.array(
        [
            [
                [
                    [0.0, 0.0, 0.0, 0.0],
                    [0.09517365, 0.10986188, 0.09963074, 0.1062995],
                    [1.065352, 1.2297686, 1.1152437, 1.1898922],
                    [0.6631518, 0.66517174, 0.65218, 0.62217724],
                    [1.0075192, 1.0188653, 1.0088668, 0.96835756],
                    [1.2357031, 1.2711056, 1.2378613, 1.1852659],
                    [1.6769525, 1.8028057, 1.7598753, 1.322085],
                    [2.0023227, 1.9694152, 2.0535903, 1.8422422],
                ]
            ]
        ],
        dtype=np.float32,
    )
    count_unequal_element(output, expected_result, 0.0001, 0.0001)

    waveform = np.array(
        [[[[[[[1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 4, 4, 3, 3, 2, 2, 1, 1]]]]]]]
    )
    output = audio.MelSpectrogram(f_max=2.0, f_min=2.0, n_fft=16)(waveform)
    expected_result = np.zeros((1, 1, 1, 1, 1, 1, 128, 3), dtype=np.float32)
    count_unequal_element(output, expected_result, 0.0001, 0.0001)

    # Description: n_mels (int,optional)
    # Supplement scenario with n_mels=0
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
    output = audio.MelSpectrogram(n_fft=16, n_mels=0)(waveform)
    # For n_mels=0, result is an empty array
    assert output.size == 0

    # Description: pad_mode equal to
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
    output = audio.MelSpectrogram(
        sample_rate=16000,
        n_fft=16,
        win_length=16,
        hop_length=8,
        f_min=0.0,
        f_max=5000.0,
        pad=0,
        n_mels=8,
        window=WindowType.BARTLETT,
        power=2.0,
        normalized=False,
        center=True,
        pad_mode=BorderType.EDGE,
        onesided=True,
        norm=NormType.NONE,
        mel_scale=MelType.HTK,
    )(waveform)
    expected_result = np.array(
        [
            [
                [
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [4.3010736, 57.146717, 8.19836, 30.797155],
                    [16.12305, 214.22078, 30.73246, 115.44654],
                    [0.68758655, 2.7503462, 1.1162192, 0.06181001],
                    [2.351627, 9.406508, 3.8176012, 0.21139751],
                    [0.13574125, 3.8158412, 2.5755224, 0.8002972],
                    [0.28429294, 1.6322781, 0.44236544, 0.29250854],
                ]
            ]
        ],
        dtype=np.float32,
    )
    count_unequal_element(output, expected_result, 0.0001, 0.0001)

    # Description: pad_mode equal to
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
    output = audio.MelSpectrogram(
        sample_rate=16000,
        n_fft=16,
        win_length=16,
        hop_length=8,
        f_min=0.0,
        f_max=5000.0,
        pad=0,
        n_mels=8,
        window=WindowType.BARTLETT,
        power=2.0,
        normalized=False,
        center=True,
        pad_mode=BorderType.CONSTANT,
        onesided=True,
        norm=NormType.NONE,
        mel_scale=MelType.HTK,
    )(waveform)
    expected_result = np.array(
        [
            [
                [
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [8.194638, 57.146717, 8.19836, 33.375484],
                    [30.718506, 214.22078, 30.73246, 125.11169],
                    [0.49448004, 2.7503462, 1.1162192, 1.130359],
                    [1.6911801, 9.406508, 3.8176012, 3.8659613],
                    [0.7390366, 3.8158412, 2.5755224, 2.5287364],
                    [0.16454992, 1.6322781, 0.44236544, 3.1256256],
                ]
            ]
        ],
        dtype=np.float32,
    )
    count_unequal_element(output, expected_result, 0.0001, 0.0001)

    # Description: Input audio signal to test norm_type_slaney.
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
    output = audio.MelSpectrogram(
        sample_rate=16000,
        n_fft=16,
        win_length=16,
        hop_length=8,
        f_min=0.0,
        f_max=5000.0,
        pad=0,
        n_mels=8,
        window=WindowType.HANN,
        power=2.0,
        normalized=False,
        center=True,
        pad_mode=BorderType.REFLECT,
        onesided=True,
        norm=NormType.SLANEY,
        mel_scale=MelType.HTK,
    )(waveform)
    expected_result = np.array(
        [
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.01248906, 0.22036129, 0.03119338, 0.11397758],
            [0.03708536, 0.6543471, 0.0926265, 0.3384483],
            [0.0058351, 0.00620103, 0.00417963, 0.00162956],
            [0.01580854, 0.01679994, 0.01132352, 0.00441484],
            [0.00067757, 0.00098009, 0.00095619, 0.00046811],
            [0.000354, 0.0006231, 0.00032267, 0.00015305],
        ],
        dtype=np.float32,
    )
    count_unequal_element(output, expected_result, 0.0001, 0.0001)

    # Description: Input audio signal to test norm_type_slaney.
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
    output = audio.MelSpectrogram(
        sample_rate=16000,
        n_fft=16,
        win_length=16,
        hop_length=8,
        f_min=0.0,
        f_max=5000.0,
        pad=0,
        n_mels=8,
        window=WindowType.HANN,
        power=2.0,
        normalized=False,
        center=True,
        pad_mode=BorderType.REFLECT,
        onesided=True,
        norm=NormType.NONE,
        mel_scale=MelType.SLANEY,
    )(waveform)
    expected_result = np.array(
        [
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [10.033709, 177.03828, 25.06076, 91.569595],
            [9.604169, 169.45934, 23.987917, 87.64953],
            [1.8953271, 2.0141885, 1.3576059, 0.5293061],
            [11.71521, 12.449905, 8.3915, 3.271695],
            [0.50411767, 0.7291876, 0.7114069, 0.34827456],
            [0.42180264, 0.71999246, 0.42024773, 0.20086795],
        ],
        dtype=np.float32,
    )
    count_unequal_element(output, expected_result, 0.0001, 0.0001)


def test_mel_spectrogram_invalid_sample_rate():
    """
    Feature: MelSpectrogram
    Description: Test MelSpectrogram with invalid sample_rate parameter
    Expectation: Correct error types and messages are raised as expected
    """
    # Description: sample_rate (int,optional)
    try:
        audio.MelSpectrogram(sample_rate=-1)
    except ValueError as error:
        assert (
            "Input sample_rate is not within the required interval of [0, 2147483647]."
            in str(error)
        )
    try:
        audio.MelSpectrogram(sample_rate="s")
    except TypeError as error:
        assert (
            "Argument sample_rate with value s is not of type [<class 'int'>], but got <class 'str'>."
            in str(error)
        )
    try:
        audio.MelSpectrogram(
            sample_rate=2147483648,
            n_fft=16,
            win_length=16,
            hop_length=8,
            f_min=0.0,
            f_max=5000.0,
            pad=0,
            n_mels=8,
        )
    except ValueError as error:
        assert (
            "Input sample_rate is not within the required interval of [0, 2147483647]."
            in str(error)
        )
    try:
        audio.MelSpectrogram(sample_rate=1500.52)
    except TypeError as error:
        assert (
            r"Argument sample_rate with value 1500.52 is not of type [<class 'int'>], "
            r"but got <class 'float'>." in str(error)
        )
    try:
        audio.MelSpectrogram(sample_rate=True)
    except TypeError as error:
        assert (
            "Argument sample_rate with value True is not of type (<class 'int'>,), but got <class 'bool'>."
            in str(error)
        )


def test_mel_spectrogram_invalid_n_fft():
    """
    Feature: MelSpectrogram
    Description: Test MelSpectrogram with invalid n_fft parameter
    Expectation: Correct error types and messages are raised as expected
    """
    # Description: n_fft (int,optional)
    try:
        audio.MelSpectrogram(n_fft=-1)
    except ValueError as error:
        assert (
            "Input n_fft is not within the required interval of [1, 2147483647]."
            in str(error)
        )
    try:
        audio.MelSpectrogram(n_fft=0)
    except ValueError as error:
        assert (
            "Input n_fft is not within the required interval of [1, 2147483647]."
            in str(error)
        )
    try:
        audio.MelSpectrogram(n_fft=1.0)
    except TypeError as error:
        assert (
            "Argument n_fft with value 1.0 is not of type [<class 'int'>], but got <class 'float'>."
            in str(error)
        )
    try:
        audio.MelSpectrogram(n_fft=False)
    except TypeError as error:
        assert (
            "Argument n_fft with value False is not of type (<class 'int'>,), but got <class 'bool'>."
            in str(error)
        )
    try:
        audio.MelSpectrogram(n_fft="s")
    except TypeError as error:
        assert (
            "Argument n_fft with value s is not of type [<class 'int'>], but got <class 'str'>."
            in str(error)
        )
    try:
        audio.MelSpectrogram(n_fft=2147483648)
    except ValueError as error:
        assert (
            "Input n_fft is not within the required interval of [1, 2147483647]."
            in str(error)
        )

    waveform = np.array(
        [[[[[[[1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 4, 4, 3, 3, 2, 2, 1, 1]]]]]]]
    )  # waveform.shape：(1, 1, 1, 1, 1, 1, 18)
    try:
        audio.MelSpectrogram(n_fft=1)(waveform)
    except RuntimeError as error:
        assert "MelSpectrogram: 'hop_length' must be greater than 0, got: 0" in str(
            error
        )

    # n_fft constraint: less than twice the size of the last dimension of the input tensor
    waveform = np.array(
        [[[[[[[1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 4, 4, 3, 3, 2, 2, 1, 1]]]]]]]
    )  # waveform.shape：(1, 1, 1, 1, 1, 1, 18)
    try:
        audio.MelSpectrogram(n_fft=400)(waveform)
    except RuntimeError as error:
        assert (
            "MelSpectrogram: Padding size should be less than the corresponding input dimension."
            in str(error)
        )


def test_mel_spectrogram_invalid_win_length():
    """
    Feature: MelSpectrogram
    Description: Test MelSpectrogram with invalid win_length parameter
    Expectation: Correct error types and messages are raised as expected
    """
    # Description: win_length (int,optional),win_length should be no more than n_fft
    try:
        audio.MelSpectrogram(win_length=-1)
    except ValueError as error:
        assert (
            "Input win_length is not within the required interval of [1, 2147483647]."
            in str(error)
        )
    try:
        audio.MelSpectrogram(win_length="s")
    except TypeError as error:
        assert (
            "Argument win_length with value s is not of type [<class 'int'>], but got <class 'str'>."
            in str(error)
        )

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
    try:
        audio.MelSpectrogram(win_length=0)(waveform)
    except ValueError as error:
        assert (
            "Input win_length is not within the required interval of [1, 2147483647]."
            in str(error)
        )

    try:
        audio.MelSpectrogram(win_length=0.1)
    except TypeError as error:
        assert (
            "Argument win_length with value 0.1 is not of type [<class 'int'>], but got <class 'float'>."
            in str(error)
        )
    try:
        audio.MelSpectrogram(win_length=True)
    except TypeError as error:
        assert (
            "Argument win_length with value True is not of type (<class 'int'>,), but got <class 'bool'>."
            in str(error)
        )
    try:
        audio.MelSpectrogram(win_length=300, n_fft=200)
    except ValueError as error:
        assert (
            "Input win_length should be no more than n_fft, but got win_length: 300 and n_fft: 200."
            in str(error)
        )
    try:
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
        audio.MelSpectrogram(win_length=160, n_fft=100, center=False)(waveform)
    except ValueError as error:
        assert (
            "Input win_length should be no more than n_fft, but got win_length: 160 and n_fft: 100."
            in str(error)
        )


def test_mel_spectrogram_invalid_hop_length():
    """
    Feature: MelSpectrogram
    Description: Test MelSpectrogram with invalid hop_length parameter
    Expectation: Correct error types and messages are raised as expected
    """
    # Description: hop_length (int,optional)
    try:
        audio.MelSpectrogram(hop_length=-1)
    except ValueError as error:
        assert (
            "Input hop_length is not within the required interval of [1, 2147483647]."
            in str(error)
        )
    try:
        audio.MelSpectrogram(hop_length=0)
    except ValueError as error:
        assert (
            "Input hop_length is not within the required interval of [1, 2147483647]."
            in str(error)
        )

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
    try:
        audio.MelSpectrogram(hop_length=0)(waveform)
    except ValueError as error:
        assert (
            "Input hop_length is not within the required interval of [1, 2147483647]."
            in str(error)
        )

    try:
        audio.MelSpectrogram(hop_length=2147483648)
    except ValueError as error:
        assert (
            "Input hop_length is not within the required interval of [1, 2147483647]."
            in str(error)
        )
    try:
        audio.MelSpectrogram(hop_length=1.5)
    except TypeError as error:
        assert (
            "Argument hop_length with value 1.5 is not of type [<class 'int'>], but got <class 'float'>."
            in str(error)
        )
    try:
        audio.MelSpectrogram(hop_length="1")
    except TypeError as error:
        assert (
            "Argument hop_length with value 1 is not of type [<class 'int'>], but got <class 'str'>."
            in str(error)
        )
    try:
        audio.MelSpectrogram(hop_length=True)
    except TypeError as error:
        assert (
            "Argument hop_length with value True is not of type (<class 'int'>,), but got <class 'bool'>."
            in str(error)
        )


def test_mel_spectrogram_invalid_f_max():
    """
    Feature: MelSpectrogram
    Description: Test MelSpectrogram with invalid f_max parameter
    Expectation: Correct error types and messages are raised as expected
    """
    # Description: f_max (float,optional)
    try:
        audio.MelSpectrogram(f_max=-1)  # f_min defaults to 0.0
    except ValueError as error:
        assert (
            "Input f_max is not within the required interval of [0, 16777216]."
            in str(error)
        )
    try:
        audio.MelSpectrogram(f_max=0)
    except TypeError as error:
        assert (
            "Argument f_max with value 0 is not of type [<class 'float'>], but got <class 'int'>."
            in str(error)
        )
    try:
        audio.MelSpectrogram(f_max=16777216.1)
    except ValueError as error:
        assert (
            "Input f_max is not within the required interval of [0, 16777216]."
            in str(error)
        )
    try:
        audio.MelSpectrogram(f_min=-2.0, f_max=-1.0)
    except ValueError as error:
        assert (
            "Input f_max is not within the required interval of [0, 16777216]."
            in str(error)
        )
    try:
        audio.MelSpectrogram(f_max=1.0, f_min=2.0, sample_rate=16000)
    except ValueError as error:
        assert "f_max should be greater than or equal to f_min" in str(error)

    audio.MelSpectrogram(f_max=2, f_min=1.0, sample_rate=16000)

    try:
        audio.MelSpectrogram(f_max="2", f_min=1.0, sample_rate=16000)
    except TypeError as error:
        assert (
            "Argument f_max with value 2 is not of type [<class 'float'>, <class 'int'>], "
            "but got <class 'str'>." in str(error)
        )
    try:
        audio.MelSpectrogram(f_max=False, f_min=1.0, sample_rate=16000)
    except TypeError as error:
        assert (
            "Argument f_max with value False is not of type (<class 'float'>, <class 'int'>),"
            " but got <class 'bool'>." in str(error)
        )


def test_mel_spectrogram_invalid_f_min():
    """
    Feature: MelSpectrogram
    Description: Test MelSpectrogram with invalid f_min parameter
    Expectation: Correct error types and messages are raised as expected
    """
    # Description: half_of_sample_rate_greater_than_f_min
    try:
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
        audio.MelSpectrogram(sample_rate=100, f_min=60.0)(waveform)
    except ValueError as error:
        assert (
            "MelSpectrogram: sample_rate // 2 should be greater than f_min when f_max is set to None"
            in str(error)
        )

    # Description: f_min (float,optional)
    try:
        audio.MelSpectrogram(
            win_length=16, hop_length=8, f_max=2.0, f_min=1, sample_rate=16000
        )
    except TypeError as error:
        assert (
            "Argument f_min with value 1 is not of type [<class 'float'>], but got <class 'int'>."
            in str(error)
        )
    try:
        audio.MelSpectrogram(win_length=16, hop_length=8, f_min=16777216.1)
    except ValueError as error:
        assert (
            "MelSpectrogram: sample_rate // 2 should be greater than f_min when f_max is set to None"
            in str(error)
        )
    try:
        audio.MelSpectrogram(win_length=16, hop_length=8, f_min=-16777216.1)
    except ValueError as error:
        assert (
            "Input f_min is not within the required interval of [-16777216, 16777216]"
            in str(error)
        )
    try:
        audio.MelSpectrogram(f_max=2.0, f_min="1.0", sample_rate=16000)
    except TypeError as error:
        assert (
            "Argument f_min with value 1.0 is not of type [<class 'float'>], but got <class 'str'>."
            in str(error)
        )
    try:
        audio.MelSpectrogram(f_max=2.0, f_min=False, sample_rate=16000)
    except TypeError as error:
        assert (
            "Argument f_min with value False is not of type [<class 'float'>], but got <class 'bool'>."
            in str(error)
        )
    try:
        audio.MelSpectrogram(f_max=2.0, f_min=None, sample_rate=16000)
    except TypeError as error:
        assert (
            "Argument f_min with value None is not of type [<class 'float'>], but got <class 'NoneType'>."
            in str(error)
        )


def test_mel_spectrogram_invalid_pad():
    """
    Feature: MelSpectrogram
    Description: Test MelSpectrogram with invalid pad parameter
    Expectation: Correct error types and messages are raised as expected
    """
    # Description: pad (int,optional)
    try:
        audio.MelSpectrogram(pad=-1)
    except ValueError as error:
        assert (
            "Input pad is not within the required interval of [0, 2147483647]."
            in str(error)
        )
    try:
        audio.MelSpectrogram(pad=2147483648)
    except ValueError as error:
        assert (
            "Input pad is not within the required interval of [0, 2147483647]."
            in str(error)
        )
    try:
        audio.MelSpectrogram(pad=1.5)
    except TypeError as error:
        assert (
            "Argument pad with value 1.5 is not of type [<class 'int'>], but got <class 'float'>."
            in str(error)
        )
    try:
        audio.MelSpectrogram(pad="1")
    except TypeError as error:
        assert (
            "Argument pad with value 1 is not of type [<class 'int'>], but got <class 'str'>."
            in str(error)
        )
    try:
        audio.MelSpectrogram(pad=False)
    except TypeError as error:
        assert (
            "Argument pad with value False is not of type (<class 'int'>,), but got <class 'bool'>."
            in str(error)
        )


def test_mel_spectrogram_invalid_n_mels():
    """
    Feature: MelSpectrogram
    Description: Test MelSpectrogram with invalid n_mels parameter
    Expectation: Correct error types and messages are raised as expected
    """
    try:
        audio.MelSpectrogram(n_mels=2147483648)
    except ValueError as error:
        assert (
            "Input n_mels is not within the required interval of [0, 2147483647]."
            in str(error)
        )
    try:
        audio.MelSpectrogram(n_mels=-1)
    except ValueError as error:
        assert (
            "Input n_mels is not within the required interval of [0, 2147483647]."
            in str(error)
        )
    try:
        audio.MelSpectrogram(n_mels=1.5)
    except TypeError as error:
        assert (
            "Argument n_mels with value 1.5 is not of type [<class 'int'>], but got <class 'float'>."
            in str(error)
        )
    try:
        audio.MelSpectrogram(n_mels="1")
    except TypeError as error:
        assert (
            "Argument n_mels with value 1 is not of type [<class 'int'>], but got <class 'str'>."
            in str(error)
        )
    try:
        audio.MelSpectrogram(n_mels=False)
    except TypeError as error:
        assert (
            "Argument n_mels with value False is not of type (<class 'int'>,), but got <class 'bool'>."
            in str(error)
        )

    waveform = np.array(
        [[[[[[[1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 4, 4, 3, 3, 2, 2, 1, 1]]]]]]]
    )
    try:
        audio.MelSpectrogram(n_fft=16, n_mels=2147483647)(waveform)
    except RuntimeError as error:
        assert "Linspace: input param n must be non-negative." in str(error)


def test_mel_spectrogram_invalid_window():
    """
    Feature: MelSpectrogram
    Description: Test MelSpectrogram with invalid window parameter
    Expectation: Correct error types and messages are raised as expected
    """
    # Description: window (WindowType,optional)
    try:
        audio.MelSpectrogram(window="WindowType.KAISER")
    except TypeError as error:
        assert (
            r"Argument window with value WindowType.KAISER is not of type "
            r"[<enum 'WindowType'>], but got <class 'str'>." in str(error)
        )
    try:
        audio.MelSpectrogram(window=123)
    except TypeError as error:
        assert (
            "Argument window with value 123 is not of type [<enum 'WindowType'>], but got <class 'int'>."
            in str(error)
        )
    try:
        audio.MelSpectrogram(window=12.3)
    except TypeError as error:
        assert (
            "Argument window with value 12.3 is not of type [<enum 'WindowType'>], but got <class 'float'>."
            in str(error)
        )
    try:
        audio.MelSpectrogram(window=False)
    except TypeError as error:
        assert (
            "Argument window with value False is not of type [<enum 'WindowType'>], but got <class 'bool'>."
            in str(error)
        )


def test_mel_spectrogram_invalid_power():
    """
    Feature: MelSpectrogram
    Description: Test MelSpectrogram with invalid power parameter
    Expectation: Correct error types and messages are raised as expected
    """
    # Description: power (float,optional)
    try:
        audio.MelSpectrogram(power=1)
    except TypeError as error:
        assert (
            "Argument power with value -1 is not of type [<class 'float'>], but got <class 'int'>."
            in str(error)
        )
    try:
        audio.MelSpectrogram(
            sample_rate=16000,
            n_fft=16,
            win_length=None,
            hop_length=None,
            f_min=0.0,
            f_max=10000.0,
            pad=0,
            n_mels=8,
            window=WindowType.HANN,
            power=0,
        )
    except ValueError as error:
        assert (
            "Input power is not within the required interval of (0, 16777216]."
            in str(error)
        )

    try:
        audio.MelSpectrogram(power=16777216.1)
    except ValueError as error:
        assert (
            "Input power is not within the required interval of (0, 16777216]."
            in str(error)
        )
    try:
        audio.MelSpectrogram(power=True)
    except TypeError as error:
        assert (
            "Argument power with value True is not of type (<class 'float'>, <class 'int'>), "
            "but got <class 'bool'>." in str(error)
        )
    try:
        audio.MelSpectrogram(win_length=16, hop_length=8, power="10")
    except TypeError as error:
        assert (
            "Argument power with value 10 is not of type [<class 'float'>, <class 'int'>], "
            "but got <class 'str'>." in str(error)
        )

    # Description: power equal to  0
    try:
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
        audio.MelSpectrogram(
            sample_rate=16000,
            n_fft=16,
            win_length=16,
            hop_length=8,
            f_min=0.0,
            f_max=10000.0,
            pad=0,
            n_mels=8,
            power=0.0,
        )(waveform)
    except ValueError as error:
        assert (
            "Input power is not within the required interval of (0, 16777216]."
            in str(error)
        )


def test_mel_spectrogram_invalid_normalized():
    """
    Feature: MelSpectrogram
    Description: Test MelSpectrogram with invalid normalized parameter
    Expectation: Correct error types and messages are raised as expected
    """
    # Description: normalized (bool,optional)
    try:
        audio.MelSpectrogram(normalized="s")
    except TypeError as error:
        assert (
            "Argument normalized with value s is not of type [<class 'bool'>], but got <class 'str'>."
            in str(error)
        )
    try:
        audio.MelSpectrogram(normalized=1)
    except TypeError as error:
        assert (
            "Argument normalized with value 1 is not of type [<class 'bool'>], but got <class 'int'>."
            in str(error)
        )
    try:
        audio.MelSpectrogram(normalized=1.0)
    except TypeError as error:
        assert (
            "Argument normalized with value 1.0 is not of type [<class 'bool'>], but got <class 'float'>."
            in str(error)
        )


def test_mel_spectrogram_invalid_center():
    """
    Feature: MelSpectrogram
    Description: Test MelSpectrogram with invalid center parameter
    Expectation: Correct error types and messages are raised as expected
    """
    # Description: center (bool,optional)
    try:
        audio.MelSpectrogram(center="XiaDanni")
    except TypeError as error:
        assert (
            "Argument center with value XiaDanni is not of type [<class 'bool'>], but got <class 'str'>."
            in str(error)
        )
    try:
        audio.MelSpectrogram(center=1)
    except TypeError as error:
        assert (
            "Argument center with value 1 is not of type [<class 'bool'>], but got <class 'int'>."
            in str(error)
        )
    try:
        audio.MelSpectrogram(center=0.0)
    except TypeError as error:
        assert (
            "Argument center with value 0.0 is not of type [<class 'bool'>], but got <class 'float'>."
            in str(error)
        )


def test_mel_spectrogram_invalid_pad_mode():
    """
    Feature: MelSpectrogram
    Description: Test MelSpectrogram with invalid pad_mode parameter
    Expectation: Correct error types and messages are raised as expected
    """
    # Description: pad_mode (BoederType,optional)
    try:
        audio.MelSpectrogram(pad_mode=False)
    except TypeError as error:
        assert (
            "Argument pad_mode with value False is not of type [<enum 'BorderType'>], but got <class 'bool'>."
            in str(error)
        )
    try:
        audio.MelSpectrogram(pad_mode=1)
    except TypeError as error:
        assert (
            "Argument pad_mode with value 1 is not of type [<enum 'BorderType'>], but got <class 'int'>."
            in str(error)
        )
    try:
        audio.MelSpectrogram(pad_mode=1.0)
    except TypeError as error:
        assert (
            "Argument pad_mode with value 1.0 is not of type [<enum 'BorderType'>], but got <class 'float'>."
            in str(error)
        )
    try:
        audio.MelSpectrogram(pad_mode="BorderType.EDGE")
    except TypeError as error:
        assert (
            r"Argument pad_mode with value BorderType.EDGE is not of type "
            r"[<enum 'BorderType'>], but got <class 'str'>." in str(error)
        )


def test_mel_spectrogram_invalid_onesided():
    """
    Feature: MelSpectrogram
    Description: Test MelSpectrogram with invalid onesided parameter
    Expectation: Correct error types and messages are raised as expected
    """
    # Description: onesided (bool,optional)
    try:
        audio.MelSpectrogram(onesided="LianLinghang")
    except TypeError as error:
        assert (
            "Argument onesided with value LianLinghang is not of type [<class 'bool'>], but got <class 'str'>."
            in str(error)
        )
    try:
        audio.MelSpectrogram(onesided=1)
    except TypeError as error:
        assert (
            "Argument onesided with value 1 is not of type [<class 'bool'>], but got <class 'int'>."
            in str(error)
        )
    try:
        audio.MelSpectrogram(onesided=1.0)
    except TypeError as error:
        assert (
            "Argument onesided with value 1.0 is not of type [<class 'bool'>], but got <class 'float'>."
            in str(error)
        )


def test_mel_spectrogram_invalid_norm():
    """
    Feature: MelSpectrogram
    Description: Test MelSpectrogram with invalid norm parameter
    Expectation: Correct error types and messages are raised as expected
    """
    # Description: norm (NormType,optional)
    try:
        audio.MelSpectrogram(norm=-1)
    except TypeError as error:
        assert (
            "Argument norm with value -1 is not of type [<enum 'NormType'>], but got <class 'int'>."
            in str(error)
        )
    try:
        audio.MelSpectrogram(norm=1.0)
    except TypeError as error:
        assert (
            "Argument norm with value 1.0 is not of type [<enum 'NormType'>], but got <class 'float'>."
            in str(error)
        )
    try:
        audio.MelSpectrogram(norm="NormType.SLANEY")
    except TypeError as error:
        assert (
            r"Argument norm with value NormType.SLANEY is not of type "
            r"[<enum 'NormType'>], but got <class 'str'>." in str(error)
        )


def test_mel_spectrogram_invalid_mel_scale():
    """
    Feature: MelSpectrogram
    Description: Test MelSpectrogram with invalid mel_scale parameter
    Expectation: Correct error types and messages are raised as expected
    """
    # Description: mel_scale (MelType,optional)
    try:
        audio.MelSpectrogram(mel_scale=-1)
    except TypeError as error:
        assert (
            "Argument mel_type with value -1 is not of type [<enum 'MelType'>], but got <class 'int'>."
            in str(error)
        )
    try:
        audio.MelSpectrogram(mel_scale=1.0)
    except TypeError as error:
        assert (
            "Argument mel_type with value 1.0 is not of type [<enum 'MelType'>], but got <class 'float'>."
            in str(error)
        )
    try:
        audio.MelSpectrogram(mel_scale="MelType.HTK")
    except TypeError as error:
        assert (
            "Argument mel_type with value MelType.HTK is not of type "
            "[<enum 'MelType'>], but got <class 'str'>." in str(error)
        )
    try:
        audio.MelSpectrogram(mel_scale=True)
    except TypeError as error:
        assert (
            "Argument mel_type with value True is not of type [<enum 'MelType'>], but got <class 'bool'>."
            in str(error)
        )


if __name__ == "__main__":
    test_mel_spectrogram_pipeline()
    test_mel_spectrogram_eager()
    test_mel_spectrogram_param()
    test_mel_spectrogram_transform()
    test_mel_spectrogram_invalid_sample_rate()
    test_mel_spectrogram_invalid_n_fft()
    test_mel_spectrogram_invalid_win_length()
    test_mel_spectrogram_invalid_hop_length()
    test_mel_spectrogram_invalid_f_max()
    test_mel_spectrogram_invalid_f_min()
    test_mel_spectrogram_invalid_pad()
    test_mel_spectrogram_invalid_n_mels()
    test_mel_spectrogram_invalid_window()
    test_mel_spectrogram_invalid_power()
    test_mel_spectrogram_invalid_normalized()
    test_mel_spectrogram_invalid_center()
    test_mel_spectrogram_invalid_pad_mode()
    test_mel_spectrogram_invalid_onesided()
    test_mel_spectrogram_invalid_norm()
    test_mel_spectrogram_invalid_mel_scale()
