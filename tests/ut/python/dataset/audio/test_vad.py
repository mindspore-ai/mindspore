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
"""Test Vad."""

import numpy as np
import pytest

import mindspore.dataset as ds
import mindspore.dataset.audio.transforms as audio
from . import count_unequal_element

DATA_DIR = "../data/dataset/audiorecord/"


def allclose_nparray(data_expected, data_me, rtol, atol, equal_nan=True):
    if np.any(np.isnan(data_expected)):
        assert np.allclose(data_me, data_expected, rtol, atol, equal_nan=equal_nan)
    elif not np.allclose(data_me, data_expected, rtol, atol, equal_nan=equal_nan):
        count_unequal_element(data_expected, data_me, rtol, atol)


def test_vad_pipeline():
    """
    Feature: Vad
    Description: Test Vad in pipeline
    Expectation: Equal results from Mindspore and benchmark
    """
    # <1000>
    dataset = ds.NumpySlicesDataset(
        np.load(DATA_DIR + "single_channel.npy")[np.newaxis, :],
        column_names=["multi_dimensional_data"],
        shuffle=False,
    )
    dataset = dataset.map(
        operations=[audio.Vad(sample_rate=600)],
        input_columns=["multi_dimensional_data"],
    )
    for item in dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
        allclose_nparray(
            item["multi_dimensional_data"],
            np.load(DATA_DIR + "single_channel_res.npy"),
            0.001,
            0.001,
        )

    # <2, 1000>
    dataset = ds.NumpySlicesDataset(
        np.load(DATA_DIR + "double_channel.npy")[np.newaxis, :],
        column_names=["multi_dimensional_data"],
        shuffle=False,
    )
    dataset = dataset.map(
        operations=[audio.Vad(sample_rate=1600)],
        input_columns=["multi_dimensional_data"],
    )
    for item in dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
        allclose_nparray(
            item["multi_dimensional_data"],
            np.load(DATA_DIR + "double_channel_res.npy"),
            0.001,
            0.001,
        )

    # <1, 1000>
    dataset = ds.NumpySlicesDataset(
        np.load(DATA_DIR + "single_channel.npy")[np.newaxis, np.newaxis, :],
        column_names=["multi_dimensional_data"],
        shuffle=False,
    )
    transforms = [audio.Vad(sample_rate=600)]
    dataset = dataset.map(
        operations=transforms, input_columns=["multi_dimensional_data"]
    )
    for item in dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
        allclose_nparray(
            item["multi_dimensional_data"],
            np.load(DATA_DIR + "single_channel_res.npy"),
            0.001,
            0.001,
        )

    # <1, 1000> trigger level and time
    dataset = ds.NumpySlicesDataset(
        np.load(DATA_DIR + "single_channel.npy")[np.newaxis, np.newaxis, :],
        column_names=["multi_dimensional_data"],
        shuffle=False,
    )
    dataset = dataset.map(
        operations=[audio.Vad(sample_rate=700, trigger_level=14.0, trigger_time=1.0)],
        input_columns=["multi_dimensional_data"],
    )
    for item in dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
        allclose_nparray(
            item["multi_dimensional_data"],
            np.load(DATA_DIR + "single_channel_trigger_res.npy"),
            0.001,
            0.001,
        )

    # <1, 1000> search time
    dataset = ds.NumpySlicesDataset(
        np.load(DATA_DIR + "single_channel.npy")[np.newaxis, np.newaxis, :],
        column_names=["multi_dimensional_data"],
        shuffle=False,
    )
    dataset = dataset.map(
        operations=[
            audio.Vad(
                sample_rate=750, trigger_level=14.0, trigger_time=1.0, search_time=2.0
            )
        ],
        input_columns=["multi_dimensional_data"],
    )
    for item in dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
        allclose_nparray(
            item["multi_dimensional_data"],
            np.load(DATA_DIR + "single_channel_search_res.npy"),
            0.001,
            0.001,
        )

    # <1, 1000> allowed gap
    dataset = ds.NumpySlicesDataset(
        np.load(DATA_DIR + "single_channel.npy")[np.newaxis, np.newaxis, :],
        column_names=["multi_dimensional_data"],
        shuffle=False,
    )
    dataset = dataset.map(
        operations=[
            audio.Vad(
                sample_rate=750,
                trigger_level=14.0,
                trigger_time=1.0,
                search_time=2.0,
                allowed_gap=0.125,
            )
        ],
        input_columns=["multi_dimensional_data"],
    )
    for item in dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
        allclose_nparray(
            item["multi_dimensional_data"],
            np.load(DATA_DIR + "single_channel_allowed_gap_res.npy"),
            0.001,
            0.001,
        )

    # <1, 1000> boot time
    dataset = ds.NumpySlicesDataset(
        np.load(DATA_DIR + "single_channel.npy")[np.newaxis, np.newaxis, :],
        column_names=["multi_dimensional_data"],
        shuffle=False,
    )
    dataset = dataset.map(
        operations=[
            audio.Vad(
                sample_rate=750,
                trigger_level=14.0,
                trigger_time=1.0,
                search_time=2.0,
                allowed_gap=0.125,
                boot_time=0.7,
            )
        ],
        input_columns=["multi_dimensional_data"],
    )
    for item in dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
        allclose_nparray(
            item["multi_dimensional_data"],
            np.load(DATA_DIR + "single_channel_boot_time_res.npy"),
            0.001,
            0.001,
        )

    # <1, 1000> noise
    dataset = ds.NumpySlicesDataset(
        np.load(DATA_DIR + "single_channel.npy")[np.newaxis, np.newaxis, :],
        column_names=["multi_dimensional_data"],
        shuffle=False,
    )
    dataset = dataset.map(
        operations=[
            audio.Vad(
                sample_rate=750,
                trigger_level=14.0,
                trigger_time=1.0,
                search_time=2.0,
                allowed_gap=0.125,
                boot_time=0.7,
                noise_up_time=0.5,
                noise_down_time=0.1,
                noise_reduction_amount=2.7,
            )
        ],
        input_columns=["multi_dimensional_data"],
    )
    for item in dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
        allclose_nparray(
            item["multi_dimensional_data"],
            np.load(DATA_DIR + "single_channel_noise_res.npy"),
            0.001,
            0.001,
        )

    # <1, 1000> measure
    dataset = ds.NumpySlicesDataset(
        np.load(DATA_DIR + "single_channel.npy")[np.newaxis, np.newaxis, :],
        column_names=["multi_dimensional_data"],
        shuffle=False,
    )
    dataset = dataset.map(
        operations=[
            audio.Vad(
                sample_rate=800,
                trigger_level=14.0,
                trigger_time=1.0,
                search_time=2.0,
                allowed_gap=0.125,
                boot_time=0.7,
                noise_up_time=0.5,
                noise_down_time=0.1,
                noise_reduction_amount=2.7,
                measure_freq=40,
                measure_duration=0.05,
                measure_smooth_time=1.0,
            )
        ],
        input_columns=["multi_dimensional_data"],
    )
    for item in dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
        allclose_nparray(
            item["multi_dimensional_data"],
            np.load(DATA_DIR + "single_channel_measure_res.npy"),
            0.001,
            0.001,
        )

    # <1, 1000> filter freq
    dataset = ds.NumpySlicesDataset(
        np.load(DATA_DIR + "single_channel.npy")[np.newaxis, np.newaxis, :],
        column_names=["multi_dimensional_data"],
        shuffle=False,
    )
    dataset = dataset.map(
        operations=[
            audio.Vad(
                sample_rate=800,
                trigger_level=14.0,
                trigger_time=1.0,
                search_time=2.0,
                allowed_gap=0.125,
                boot_time=0.7,
                measure_freq=40,
                measure_duration=0.05,
                measure_smooth_time=1.0,
                hp_filter_freq=20.0,
                lp_filter_freq=3000.0,
                hp_lifter_freq=75.0,
                lp_lifter_freq=1000.0,
                noise_up_time=0.5,
                noise_down_time=0.1,
                noise_reduction_amount=2.7,
            )
        ],
        input_columns=["multi_dimensional_data"],
    )
    for item in dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
        allclose_nparray(
            item["multi_dimensional_data"],
            np.load(DATA_DIR + "single_channel_filter_res.npy"),
            0.001,
            0.001,
        )


def test_vad_pipeline_invalid_param():
    """
    Feature: Vad
    Description: Test Vad with invalid input parameters
    Expectation: Throw ValueError or TypeError
    """
    with pytest.raises(
        ValueError,
        match=r"Input sample_rate is not within the required interval of \[1, 2147483647\].",
    ):
        audio.Vad(sample_rate=-10)

    with pytest.raises(
        ValueError,
        match=r"Input search_time is not within the required interval of \[0, 16777216\].",
    ):
        audio.Vad(sample_rate=1000, search_time=-10)

    with pytest.raises(
        ValueError,
        match=r"Input allowed_gap is not within the required interval of \[0, 16777216\].",
    ):
        audio.Vad(sample_rate=1000, allowed_gap=-10)

    with pytest.raises(
        ValueError,
        match=r"Input pre_trigger_time is not within the required interval of \[0, 16777216\].",
    ):
        audio.Vad(sample_rate=1000, pre_trigger_time=-10)

    with pytest.raises(
        ValueError,
        match=r"Input boot_time is not within the required interval of \[0, 16777216\].",
    ):
        audio.Vad(sample_rate=1000, boot_time=-10)

    with pytest.raises(
        ValueError,
        match=r"Input noise_up_time is not within the required interval of \[0, 16777216\].",
    ):
        audio.Vad(sample_rate=1000, noise_up_time=-10)

    with pytest.raises(
        ValueError,
        match=r"Input noise_down_time is not within the required interval of \[0, 16777216\].",
    ):
        audio.Vad(sample_rate=1000, noise_down_time=-10)

    with pytest.raises(
        ValueError,
        match=r"Input noise_up_time should be greater than noise_down_time, but got noise_up_time: 1 and"
        + r" noise_down_time: 3.",
    ):
        audio.Vad(sample_rate=1000, noise_up_time=1, noise_down_time=3)

    with pytest.raises(
        ValueError,
        match=r"Input noise_reduction_amount is not within the required interval of \[0, 16777216\].",
    ):
        audio.Vad(sample_rate=1000, noise_reduction_amount=-10)

    with pytest.raises(
        ValueError,
        match=r"Input measure_freq is not within the required interval of \(0, 16777216\].",
    ):
        audio.Vad(sample_rate=1000, measure_freq=-10)

    with pytest.raises(
        ValueError,
        match=r"Input measure_duration is not within the required interval of \[0, 16777216\].",
    ):
        audio.Vad(sample_rate=1000, measure_duration=-10)

    with pytest.raises(
        ValueError,
        match=r"Input measure_smooth_time is not within the required interval of \[0, 16777216\].",
    ):
        audio.Vad(sample_rate=1000, measure_smooth_time=-10)

    with pytest.raises(
        ValueError,
        match=r"Input hp_filter_freq is not within the required interval of \(0, 16777216\].",
    ):
        audio.Vad(sample_rate=1000, hp_filter_freq=-10)

    with pytest.raises(
        ValueError,
        match=r"Input lp_filter_freq is not within the required interval of \(0, 16777216\].",
    ):
        audio.Vad(sample_rate=1000, lp_filter_freq=-10)

    with pytest.raises(
        ValueError,
        match=r"Input hp_lifter_freq is not within the required interval of \(0, 16777216\].",
    ):
        audio.Vad(sample_rate=1000, hp_lifter_freq=-10)

    with pytest.raises(
        ValueError,
        match=r"Input lp_lifter_freq is not within the required interval of \(0, 16777216\].",
    ):
        audio.Vad(sample_rate=1000, lp_lifter_freq=-10)


def test_vad_eager():
    """
    Feature: Vad
    Description: Test Vad with eager mode
    Expectation: Equal results from Mindspore and benchmark
    """
    spectrogram = np.load(DATA_DIR + "single_channel.npy")
    output = audio.Vad(sample_rate=600)(spectrogram)
    out_expect = np.load(DATA_DIR + "single_channel_res.npy")
    allclose_nparray(output, out_expect, 0.001, 0.001)

    spectrogram = np.load(DATA_DIR + "double_channel.npy")
    output = audio.Vad(sample_rate=1600)(spectrogram)
    out_expect = np.load(DATA_DIR + "double_channel_res.npy")
    allclose_nparray(output, out_expect, 0.001, 0.001)

    # benchmark trigger warning
    spectrogram = np.load(DATA_DIR + "three_channel.npy")
    output = audio.Vad(sample_rate=1600)(spectrogram)
    out_expect = np.load(DATA_DIR + "three_channel_res.npy")
    allclose_nparray(output, out_expect, 0.001, 0.001)


def test_vad_transform():
    """
    Feature: Vad
    Description: Test Vad with various valid input parameters and data types
    Expectation: The operation completes successfully
    """

    waveform = np.random.randn(10, 2, 5, 10).astype(np.float32)
    dataset = ds.NumpySlicesDataset(waveform, column_names=["column1"])
    vad = audio.Vad(
        8000,
        1.0,
        1.0,
        1.0,
        1.125,
        2.0,
        0.2,
        10.6,
        0.6,
        2.68,
        40,
        80,
        0.4,
        30.0,
        3000.0,
        175.0,
        1000.0,
    )
    dataset = dataset.map(operations=vad)
    for _ in dataset.create_dict_iterator(output_numpy=True):
        pass

    # VAD test, sample_rate=500，dtype=float64
    waveform = np.random.randn(100).astype(np.float64)
    vad = audio.Vad(
        500,
        7,
        10.0,
        0.85,
        0.3,
        0.0,
        1.56,
        5,
        1.85,
        2.7,
        38.6,
        2,
        1,
        20.0,
        1000.0,
        30,
        800,
    )
    vad(waveform)

    # VAD test, sample_rate=1024，dtype=float16
    waveform = np.random.randn(10, 20).astype(np.float16)
    vad = audio.Vad(
        1024,
        100.0,
        0.12,
        12.6,
        1,
        1,
        3.05,
        0.12,
        0.08,
        1.3,
        10,
        32.04,
        0.05,
        20.0,
        15000,
        25.0,
        2000,
    )
    vad(waveform)

    # VAD test, sample_rate=80，dtype=int64
    waveform = np.random.randint(-1000, 1000, (5, 8, 6))
    vad = audio.Vad(
        80,
        0.5,
        0.05,
        0.25,
        0.08,
        0.08,
        0.01,
        1.4,
        1,
        5,
        0.8,
        0.5,
        0.8,
        20.0,
        10000,
        1.5,
        1000,
    )
    vad(waveform)

    # VAD test, sample_rate=2000
    waveform = np.random.randn(10, 1000)
    vad = audio.Vad(2000, -2, 18.0, 2, 0.99, 1.06, 10.8, 0.86, 0.5, 1.6, 200)
    vad(waveform)

    # VAD test, noise_up_time less than noise_down_time
    waveform = np.random.randn(10, 20).astype(np.float16)
    with pytest.raises(
        ValueError,
        match="Input noise_up_time should be greater than noise_down_time,"
        " but got noise_up_time: 0.12 and noise_down_time: 1.08.",
    ):
        vad = audio.Vad(
            1024, 100.0, 0.12, 12.6, 1, 1, 3.05, 0.12, 1.08, 1.3, 10, 32.04, 0.05
        )
        vad(waveform)

    # VAD test, spectrum less than start
    waveform = np.random.randn(10, 20).astype(np.float16)
    with pytest.raises(
        RuntimeError,
        match="Vad: the end of spectrum must be greater than the start. Check "
        "if `hp_filter_freq` is too large or `lp_filter_freq` is too small.",
    ):
        vad = audio.Vad(
            1024,
            100.0,
            0.12,
            12.6,
            1,
            5,
            3.05,
            0.12,
            0.08,
            10.3,
            10,
            32.04,
            0.05,
            20.0,
            10,
            25.0,
            2000,
        )
        vad(waveform)


def test_vad_invalid_input():
    """
    Feature: Vad
    Description: Test Vad with invalid input
    Expectation: Error is raised as expected
    """
    # VAD test, input=list
    waveform = np.random.randn(10, 10).tolist()
    vad = audio.Vad(2000)
    with pytest.raises(
        TypeError, match="Input should be NumPy audio, got <class 'list'>."
    ):
        vad(waveform)

    # VAD test, input=int
    waveform = 10
    vad = audio.Vad(2000)
    with pytest.raises(
        TypeError, match="Input should be NumPy audio, got <class 'int'>."
    ):
        vad(waveform)

    # VAD test, shape equals 0d
    waveform = np.array(10)
    vad = audio.Vad(2000)
    with pytest.raises(
        RuntimeError,
        match="Vad: the shape of input tensor does not match the requirement of operator. "
        "Expecting tensor in shape of <..., time>. But got tensor with dimension 0",
    ):
        vad(waveform)

    # VAD test, shape equals 0d
    waveform = np.array(["1", "2", "3"])
    vad = audio.Vad(2000)
    with pytest.raises(
        RuntimeError,
        match="Vad: the data type of input tensor does not match the requirement"
        " of operator. Expecting tensor in type of \\[int, float, double\\]."
        " But got type string.",
    ):
        vad(waveform)


def test_vad_invalid_sample_rate():
    """
    Feature: Vad
    Description: Test Vad with invalid sample rate
    Expectation: Error is raised as expected
    """
    # VAD test, sample_rate equals float
    with pytest.raises(
        TypeError,
        match="Argument sample_rate with value 2000.0 is not of type"
        " \\[<class 'int'>\\], but got <class 'float'>.",
    ):
        audio.Vad(2000.0)

    # VAD test, sample_rate equals 0
    with pytest.raises(
        ValueError,
        match=r"Input sample_rate is not within the required interval of \[1, 2147483647\]",
    ):
        audio.Vad(0)

    # VAD test, sample_rate equals 2147483648
    with pytest.raises(
        ValueError,
        match=r"Input sample_rate is not within the required interval of \[1, 2147483647\]",
    ):
        audio.Vad(2147483648)

    # VAD test, sample_rate equals bool
    with pytest.raises(
        TypeError,
        match="Argument sample_rate with value True is not of type"
        " \\(<class 'int'>,\\), but got <class 'bool'>.",
    ):
        audio.Vad(True)

    # VAD test, sample_rate too small
    waveform = np.random.randn(10, 100)
    vad = audio.Vad(20)
    with pytest.raises(
        RuntimeError,
        match="Vad: the end of spectrum must be greater than the start. Check if"
        " `hp_filter_freq` is too large or `lp_filter_freq` is too small.",
    ):
        vad(waveform)


def test_vad_invalid_trigger_level():
    """
    Feature: Vad
    Description: Test Vad with invalid trigger level
    Expectation: Error is raised as expected
    """
    # VAD test, trigger_level too large
    with pytest.raises(
        ValueError,
        match="Input trigger_level is not within the required interval of \\[-16777216, 16777216\\].",
    ):
        audio.Vad(800, 16777216.1)

    # VAD test, trigger_level too small
    with pytest.raises(
        ValueError,
        match="Input trigger_level is not within the required interval of \\[-16777216, 16777216\\].",
    ):
        audio.Vad(800, -16777216.1)

    # VAD test, trigger_level equals bool
    with pytest.raises(
        TypeError,
        match="Argument trigger_level with value True is not of type"
        " \\(<class 'int'>, <class 'float'>\\), but got <class 'bool'>.",
    ):
        audio.Vad(800, True)

    # VAD test, trigger_level equals str
    with pytest.raises(
        TypeError,
        match="Argument trigger_level with value 2 is not of type \\[<class"
        " 'int'>, <class 'float'>\\], but got <class 'str'>.",
    ):
        audio.Vad(800, "2")

    # VAD test, trigger_level equals None
    with pytest.raises(
        TypeError,
        match="Argument trigger_level with value None is not of type \\[<class"
        " 'int'>, <class 'float'>\\], but got <class 'NoneType'>.",
    ):
        audio.Vad(800, None)


def test_vad_invalid_trigger_time():
    """
    Feature: Vad
    Description: Test Vad with invalid trigger time
    Expectation: Error is raised as expected
    """
    # VAD test, trigger_time too large
    with pytest.raises(
        ValueError,
        match="Input trigger_time is not within the required interval of \\[0, 16777216\\]",
    ):
        audio.Vad(800, trigger_time=16777216.1)

    # VAD test, trigger_time equals -1
    with pytest.raises(
        ValueError,
        match="Input trigger_time is not within the required interval of \\[0, 16777216\\]",
    ):
        audio.Vad(800, trigger_time=-1)

    # VAD test, trigger_time equals bool
    with pytest.raises(
        TypeError,
        match="Argument trigger_time with value True is not of type \\(<class"
        " 'int'>, <class 'float'>\\), but got <class 'bool'>.",
    ):
        audio.Vad(800, trigger_time=True)

    # VAD test, trigger_time equals str
    with pytest.raises(
        TypeError,
        match="Argument trigger_time with value 1 is not of type \\[<class"
        " 'int'>, <class 'float'>\\], but got <class 'str'>.",
    ):
        audio.Vad(800, trigger_time="1")

    # VAD test, trigger_time equals None
    with pytest.raises(
        TypeError,
        match="Argument trigger_time with value None is not of type"
        " \\[<class 'int'>, <class 'float'>\\], but got <class 'NoneType'>.",
    ):
        audio.Vad(800, trigger_time=None)

    # VAD test, trigger_time equals -0.5
    with pytest.raises(
        ValueError,
        match="Input search_time is not within the required interval of \\[0, 16777216\\].",
    ):
        audio.Vad(800, search_time=-0.5)


def test_vad_invalid_search_time():
    """
    Feature: Vad
    Description: Test Vad with invalid search time
    Expectation: Error is raised as expected
    """
    # VAD test, search_time equals 16777216.1
    with pytest.raises(
        ValueError,
        match="Input search_time is not within the required interval of \\[0, 16777216\\].",
    ):
        audio.Vad(800, search_time=16777216.1)

    # VAD test, search_time equals False
    with pytest.raises(
        TypeError,
        match="Argument search_time with value False is not of type \\(<class"
        " 'int'>, <class 'float'>\\), but got <class 'bool'>.",
    ):
        audio.Vad(800, search_time=False)

    # VAD test, search_time equals str
    with pytest.raises(
        TypeError,
        match="Argument search_time with value 2 is not of type \\[<class"
        " 'int'>, <class 'float'>\\], but got <class 'str'>.",
    ):
        audio.Vad(800, search_time="2")

    # VAD test, search_time equals list
    with pytest.raises(
        TypeError,
        match="Argument search_time with value \\[1\\] is not of"
        " type \\[<class 'int'>, <class 'float'>\\], but got <class 'list'>.",
    ):
        audio.Vad(800, search_time=[1])


def test_vad_invalid_allowed_gap():
    """
    Feature: Vad
    Description: Test Vad with invalid allowed gap
    Expectation: Error is raised as expected
    """
    # VAD test, allowed_gap too large
    with pytest.raises(
        ValueError,
        match="Input allowed_gap is not within the required interval of \\[0, 16777216\\].",
    ):
        audio.Vad(800, allowed_gap=16777216.1)

    # VAD test, allowed_gap too small
    with pytest.raises(
        ValueError,
        match="Input allowed_gap is not within the required interval of \\[0, 16777216\\].",
    ):
        audio.Vad(800, allowed_gap=-1)

    # VAD test, allowed_gap equals str
    with pytest.raises(
        TypeError,
        match="Argument allowed_gap with value 1 is not of type \\[<class"
        " 'int'>, <class 'float'>\\], but got <class 'str'>.",
    ):
        audio.Vad(800, allowed_gap="1")

    # VAD test, allowed_gap equals None
    with pytest.raises(
        TypeError,
        match="Argument allowed_gap with value None is not of type \\[<class"
        " 'int'>, <class 'float'>\\], but got <class 'NoneType'>.",
    ):
        audio.Vad(800, allowed_gap=None)


def test_vad_invalid_pre_trigger_time():
    """
    Feature: Vad
    Description: Test Vad with invalid pre trigger time
    Expectation: Error is raised as expected
    """
    # VAD test, pre_trigger_time too large
    with pytest.raises(
        ValueError,
        match="Input pre_trigger_time is not within the required interval of \\[0, 16777216\\].",
    ):
        audio.Vad(800, pre_trigger_time=16777217)

    # VAD test, pre_trigger_time equals negative number
    with pytest.raises(
        ValueError,
        match="Input pre_trigger_time is not within the required interval of \\[0, 16777216\\].",
    ):
        audio.Vad(800, pre_trigger_time=-0.01)

    # VAD test, pre_trigger_time equals None
    with pytest.raises(
        TypeError,
        match="Argument pre_trigger_time with value None is not of type \\[<class"
        " 'int'>, <class 'float'>\\], but got <class 'NoneType'>.",
    ):
        audio.Vad(800, pre_trigger_time=None)

    # VAD test, pre_trigger_time equals tuple
    with pytest.raises(
        TypeError,
        match="Argument pre_trigger_time with value \\(1,\\) is not of"
        " type \\[<class 'int'>, <class 'float'>\\], but got <class 'tuple'>.",
    ):
        audio.Vad(800, pre_trigger_time=(1,))

    # VAD test, pre_trigger_time equals str
    with pytest.raises(
        TypeError,
        match="Argument pre_trigger_time with value 10 is not of type"
        " \\[<class 'int'>, <class 'float'>\\], but got <class 'str'>.",
    ):
        audio.Vad(800, pre_trigger_time="10")


def test_vad_invalid_boot_time():
    """
    Feature: Vad
    Description: Test Vad with invalid boot time
    Expectation: Error is raised as expected
    """
    # VAD test, boot_time too large
    with pytest.raises(
        ValueError,
        match="Input boot_time is not within the required interval of \\[0, 16777216\\].",
    ):
        audio.Vad(800, boot_time=16777217)

    # VAD test, boot_time equals -0.01
    with pytest.raises(
        ValueError,
        match="Input boot_time is not within the required interval of \\[0, 16777216\\].",
    ):
        audio.Vad(800, boot_time=-0.01)

    # VAD test, boot_time equals None
    with pytest.raises(
        TypeError,
        match="Argument boot_time with value None is not of type \\[<class"
        " 'int'>, <class 'float'>\\], but got <class 'NoneType'>.",
    ):
        audio.Vad(800, boot_time=None)

    # VAD test, boot_time equals bool
    with pytest.raises(
        TypeError,
        match="Argument boot_time with value True is not of type \\("
        "<class 'int'>, <class 'float'>\\), but got <class 'bool'>.",
    ):
        audio.Vad(800, boot_time=True)

    # VAD test, boot_time equals str
    with pytest.raises(
        TypeError,
        match="Argument boot_time with value 10 is not of type"
        " \\[<class 'int'>, <class 'float'>\\], but got <class 'str'>.",
    ):
        audio.Vad(800, boot_time="10")


def test_vad_invalid_noise_up_time():
    """
    Feature: Vad
    Description: Test Vad with invalid noise up time
    Expectation: Error is raised as expected
    """
    # VAD test, noise_up_time too large
    with pytest.raises(
        ValueError,
        match="Input noise_up_time is not within the required interval of \\[0, 16777216\\].",
    ):
        audio.Vad(800, noise_up_time=16777217)

    # VAD test, noise_up_time equals -0.01
    with pytest.raises(
        ValueError,
        match="Input noise_up_time is not within the required interval of \\[0, 16777216\\].",
    ):
        audio.Vad(800, noise_up_time=-0.01)

    # VAD test, noise_up_time equals None
    with pytest.raises(
        TypeError,
        match="Argument noise_up_time with value None is not of type \\[<class"
        " 'int'>, <class 'float'>\\], but got <class 'NoneType'>.",
    ):
        audio.Vad(800, noise_up_time=None)

    # VAD test, noise_up_time equals tuple
    with pytest.raises(
        TypeError,
        match="Argument noise_up_time with value \\(1,\\) is not of"
        " type \\[<class 'int'>, <class 'float'>\\], but got <class 'tuple'>.",
    ):
        audio.Vad(800, noise_up_time=(1,))

    # VAD test, noise_up_time equals str
    with pytest.raises(
        TypeError,
        match="Argument noise_up_time with value 10 is not of type"
        " \\[<class 'int'>, <class 'float'>\\], but got <class 'str'>.",
    ):
        audio.Vad(800, noise_up_time="10")

    # VAD test, noise_up_time equals bool
    with pytest.raises(
        TypeError,
        match="Argument noise_up_time with value True is not of type \\("
        "<class 'int'>, <class 'float'>\\), but got <class 'bool'>.",
    ):
        audio.Vad(800, noise_up_time=True)


def test_vad_invalid_noise_down_time():
    """
    Feature: Vad
    Description: Test Vad with invalid noise down time
    Expectation: Error is raised as expected
    """
    # VAD test, noise_up_time too large
    with pytest.raises(
        ValueError,
        match="Input noise_down_time is not within the required interval of \\[0, 16777216\\].",
    ):
        audio.Vad(800, noise_down_time=16777217)

    # VAD test, noise_up_time equals -0.01
    with pytest.raises(
        ValueError,
        match="Input noise_down_time is not within the required interval of \\[0, 16777216\\].",
    ):
        audio.Vad(800, noise_down_time=-0.01)

    # VAD test, noise_up_time equals None
    with pytest.raises(
        TypeError,
        match="Argument noise_down_time with value None is not of type \\[<class"
        " 'int'>, <class 'float'>\\], but got <class 'NoneType'>.",
    ):
        audio.Vad(800, noise_down_time=None)

    # VAD test, noise_down_time equals list
    with pytest.raises(
        TypeError,
        match="Argument noise_down_time with value \\[1\\] is not of"
        " type \\[<class 'int'>, <class 'float'>\\], but got <class 'list'>.",
    ):
        audio.Vad(800, noise_down_time=[1])

    # VAD test, noise_down_time equals str
    with pytest.raises(
        TypeError,
        match="Argument noise_down_time with value 10 is not of type"
        " \\[<class 'int'>, <class 'float'>\\], but got <class 'str'>.",
    ):
        audio.Vad(800, noise_down_time="10")

    # VAD test, noise_down_time equals bool
    with pytest.raises(
        TypeError,
        match="Argument noise_down_time with value True is not of type \\("
        "<class 'int'>, <class 'float'>\\), but got <class 'bool'>.",
    ):
        audio.Vad(800, noise_down_time=True)


def test_vad_invalid_noise_reduction_amount():
    """
    Feature: Vad
    Description: Test Vad with invalid noise reduction amount
    Expectation: Error is raised as expected
    """
    # VAD test, noise_reduction_amount too large
    with pytest.raises(
        ValueError,
        match="Input noise_reduction_amount is not within the required interval of \\[0, 16777216\\].",
    ):
        audio.Vad(800, noise_reduction_amount=16777217)

    # VAD test, noise_reduction_amount equals -0.01
    with pytest.raises(
        ValueError,
        match="Input noise_reduction_amount is not within the required interval of \\[0, 16777216\\].",
    ):
        audio.Vad(800, noise_reduction_amount=-0.01)

    # VAD test, noise_reduction_amount equals None
    with pytest.raises(
        TypeError,
        match="Argument noise_reduction_amount with value None is not of type \\[<class"
        " 'int'>, <class 'float'>\\], but got <class 'NoneType'>.",
    ):
        audio.Vad(800, noise_reduction_amount=None)

    # VAD test, noise_reduction_amount equals tuple
    with pytest.raises(
        TypeError,
        match="Argument noise_reduction_amount with value \\(1,\\) is not of"
        " type \\[<class 'int'>, <class 'float'>\\], but got <class 'tuple'>.",
    ):
        audio.Vad(800, noise_reduction_amount=(1,))

    # VAD test, noise_reduction_amount equals str
    with pytest.raises(
        TypeError,
        match="Argument noise_reduction_amount with value 10 is not of type"
        " \\[<class 'int'>, <class 'float'>\\], but got <class 'str'>.",
    ):
        audio.Vad(800, noise_reduction_amount="10")

    # VAD test, noise_reduction_amount equals bool
    with pytest.raises(
        TypeError,
        match="Argument noise_reduction_amount with value True is not of type \\("
        "<class 'int'>, <class 'float'>\\), but got <class 'bool'>.",
    ):
        audio.Vad(800, noise_reduction_amount=True)


def test_vad_invalid_measure_freq():
    """
    Feature: Vad
    Description: Test Vad with invalid measure freq
    Expectation: Error is raised as expected
    """
    # VAD test, measure_freq too large
    with pytest.raises(
        ValueError,
        match="Input measure_freq is not within the required interval of \\(0, 16777216\\].",
    ):
        audio.Vad(800, measure_freq=16777217)

    # VAD test, measure_freq equals 0
    with pytest.raises(
        ValueError,
        match="Input measure_freq is not within the required interval of \\(0, 16777216\\].",
    ):
        audio.Vad(800, measure_freq=0)

    # VAD test, measure_freq equals None
    with pytest.raises(
        TypeError,
        match="Argument measure_freq with value None is not of type \\[<class"
        " 'int'>, <class 'float'>\\], but got <class 'NoneType'>.",
    ):
        audio.Vad(800, measure_freq=None)

    # VAD test, measure_freq equals tuple
    with pytest.raises(
        TypeError,
        match="Argument measure_freq with value \\(1,\\) is not of"
        " type \\[<class 'int'>, <class 'float'>\\], but got <class 'tuple'>.",
    ):
        audio.Vad(800, measure_freq=(1,))

    # VAD test, measure_freq equals str
    with pytest.raises(
        TypeError,
        match="Argument measure_freq with value 10 is not of type"
        " \\[<class 'int'>, <class 'float'>\\], but got <class 'str'>.",
    ):
        audio.Vad(800, measure_freq="10")

    # VAD test, measure_freq equals bool
    with pytest.raises(
        TypeError,
        match="Argument measure_freq with value True is not of type \\("
        "<class 'int'>, <class 'float'>\\), but got <class 'bool'>.",
    ):
        audio.Vad(800, measure_freq=True)


def test_vad_invalid_measure_duration():
    """
    Feature: Vad
    Description: Test Vad with invalid measure duration
    Expectation: Error is raised as expected
    """
    # VAD test, measure_duration too large
    with pytest.raises(
        ValueError,
        match="Input measure_duration is not within the required interval of \\[0, 16777216\\].",
    ):
        audio.Vad(800, measure_duration=16777217)

    # VAD test, measure_duration too small
    with pytest.raises(
        ValueError,
        match="Input measure_duration is not within the required interval of \\[0, 16777216\\].",
    ):
        audio.Vad(800, measure_duration=-0.01)

    # VAD test, measure_duration equals tuple
    with pytest.raises(
        TypeError,
        match="Argument measure_duration with value \\(1,\\) is not of"
        " type \\[<class 'int'>, <class 'float'>\\], but got <class 'tuple'>.",
    ):
        audio.Vad(800, measure_duration=(1,))

    # VAD test, measure_duration equals bool
    with pytest.raises(
        TypeError,
        match="Argument measure_duration with value True is not of type \\("
        "<class 'int'>, <class 'float'>\\), but got <class 'bool'>.",
    ):
        audio.Vad(800, measure_duration=True)


def test_vad_invalid_measure_smooth_time():
    """
    Feature: Vad
    Description: Test Vad with invalid measure smooth time
    Expectation: Error is raised as expected
    """
    # VAD test, measure_smooth_time too large
    with pytest.raises(
        ValueError,
        match="Input measure_smooth_time is not within the required interval of \\[0, 16777216\\].",
    ):
        audio.Vad(800, measure_smooth_time=16777217)

    # VAD test, measure_smooth_time too small
    with pytest.raises(
        ValueError,
        match="Input measure_smooth_time is not within the required interval of \\[0, 16777216\\].",
    ):
        audio.Vad(800, measure_smooth_time=-0.01)

    # VAD test, measure_smooth_time equals None
    with pytest.raises(
        TypeError,
        match="Argument measure_smooth_time with value None is not of type \\[<class"
        " 'int'>, <class 'float'>\\], but got <class 'NoneType'>.",
    ):
        audio.Vad(800, measure_smooth_time=None)

    # VAD test, measure_smooth_time equals tuple
    with pytest.raises(
        TypeError,
        match="Argument measure_smooth_time with value \\(1,\\) is not of"
        " type \\[<class 'int'>, <class 'float'>\\], but got <class 'tuple'>.",
    ):
        audio.Vad(800, measure_smooth_time=(1,))

    # VAD test, measure_smooth_time equals str
    with pytest.raises(
        TypeError,
        match="Argument measure_smooth_time with value 10 is not of type"
        " \\[<class 'int'>, <class 'float'>\\], but got <class 'str'>.",
    ):
        audio.Vad(800, measure_smooth_time="10")

    # VAD test, measure_smooth_time equals bool
    with pytest.raises(
        TypeError,
        match="Argument measure_smooth_time with value True is not of type \\("
        "<class 'int'>, <class 'float'>\\), but got <class 'bool'>.",
    ):
        audio.Vad(800, measure_smooth_time=True)


def test_vad_invalid_hp_filter_freq():
    """
    Feature: Vad
    Description: Test Vad with invalid hp filter freq
    Expectation: Error is raised as expected
    """
    # VAD test, hp_filter_freq too large
    with pytest.raises(
        ValueError,
        match="Input hp_filter_freq is not within the required interval of \\(0, 16777216\\].",
    ):
        audio.Vad(800, hp_filter_freq=16777217)

    # VAD test, hp_filter_freq too small
    with pytest.raises(
        ValueError,
        match="Input hp_filter_freq is not within the required interval of \\(0, 16777216\\].",
    ):
        audio.Vad(800, hp_filter_freq=0)

    # VAD test, hp_filter_freq equals None
    with pytest.raises(
        TypeError,
        match="Argument hp_filter_freq with value None is not of type \\[<class"
        " 'int'>, <class 'float'>\\], but got <class 'NoneType'>.",
    ):
        audio.Vad(800, hp_filter_freq=None)

    # VAD test, hp_filter_freq equals str
    with pytest.raises(
        TypeError,
        match="Argument hp_filter_freq with value 10 is not of type"
        " \\[<class 'int'>, <class 'float'>\\], but got <class 'str'>.",
    ):
        audio.Vad(800, hp_filter_freq="10")

    # VAD test, hp_filter_freq equals True
    with pytest.raises(
        TypeError,
        match="Argument hp_filter_freq with value True is not of type \\("
        "<class 'int'>, <class 'float'>\\), but got <class 'bool'>.",
    ):
        audio.Vad(800, hp_filter_freq=True)

    # VAD test, lp_filter_freq too large
    with pytest.raises(
        ValueError,
        match="Input lp_filter_freq is not within the required interval of \\(0, 16777216\\].",
    ):
        audio.Vad(800, lp_filter_freq=16777217)

    # VAD test, lp_filter_freq equals 0
    with pytest.raises(
        ValueError,
        match="Input lp_filter_freq is not within the required interval of \\(0, 16777216\\].",
    ):
        audio.Vad(800, lp_filter_freq=0)

    # VAD test, lp_filter_freq equals None
    with pytest.raises(
        TypeError,
        match="Argument lp_filter_freq with value None is not of type \\[<class"
        " 'int'>, <class 'float'>\\], but got <class 'NoneType'>.",
    ):
        audio.Vad(800, lp_filter_freq=None)

    # VAD test, lp_filter_freq equals True
    with pytest.raises(
        TypeError,
        match="Argument lp_filter_freq with value True is not of type \\("
        "<class 'int'>, <class 'float'>\\), but got <class 'bool'>.",
    ):
        audio.Vad(800, lp_filter_freq=True)

    # VAD test, hp_lifter_freq equals 16777217
    with pytest.raises(
        ValueError,
        match="Input hp_lifter_freq is not within the required interval of \\(0, 16777216\\].",
    ):
        audio.Vad(800, hp_lifter_freq=16777217)

    # VAD test, hp_lifter_freq equals 0
    with pytest.raises(
        ValueError,
        match="Input hp_lifter_freq is not within the required interval of \\(0, 16777216\\].",
    ):
        audio.Vad(800, hp_lifter_freq=0)

    # VAD test, hp_lifter_freq equals None
    with pytest.raises(
        TypeError,
        match="Argument hp_lifter_freq with value None is not of type \\[<class"
        " 'int'>, <class 'float'>\\], but got <class 'NoneType'>.",
    ):
        audio.Vad(800, hp_lifter_freq=None)

    # VAD test, hp_lifter_freq equals tuple
    with pytest.raises(
        TypeError,
        match="Argument hp_lifter_freq with value \\(1,\\) is not of"
        " type \\[<class 'int'>, <class 'float'>\\], but got <class 'tuple'>.",
    ):
        audio.Vad(800, hp_lifter_freq=(1,))

    # VAD test, hp_lifter_freq equals str
    with pytest.raises(
        TypeError,
        match="Argument hp_lifter_freq with value 10 is not of type"
        " \\[<class 'int'>, <class 'float'>\\], but got <class 'str'>.",
    ):
        audio.Vad(800, hp_lifter_freq="10")

    # VAD test, hp_lifter_freq equals True
    with pytest.raises(
        TypeError,
        match="Argument hp_lifter_freq with value True is not of type \\("
        "<class 'int'>, <class 'float'>\\), but got <class 'bool'>.",
    ):
        audio.Vad(800, hp_lifter_freq=True)


def test_vad_invalid_lp_lifter_freq():
    """
    Feature: Vad
    Description: Test Vad with invalid lp lifter freq
    Expectation: Error is raised as expected
    """
    # VAD test, lp_lifter_freq too large
    with pytest.raises(
        ValueError,
        match="Input lp_lifter_freq is not within the required interval of \\(0, 16777216\\].",
    ):
        audio.Vad(800, lp_lifter_freq=16777217)

    # VAD test, lp_lifter_freq equals 0
    with pytest.raises(
        ValueError,
        match="Input lp_lifter_freq is not within the required interval of \\(0, 16777216\\].",
    ):
        audio.Vad(800, lp_lifter_freq=0)

    # VAD test, lp_lifter_freq equals None
    with pytest.raises(
        TypeError,
        match="Argument lp_lifter_freq with value None is not of type \\[<class"
        " 'int'>, <class 'float'>\\], but got <class 'NoneType'>.",
    ):
        audio.Vad(800, lp_lifter_freq=None)

    # VAD test, lp_lifter_freq equals tuple
    with pytest.raises(
        TypeError,
        match="Argument lp_lifter_freq with value \\(1,\\) is not of"
        " type \\[<class 'int'>, <class 'float'>\\], but got <class 'tuple'>.",
    ):
        audio.Vad(800, lp_lifter_freq=(1,))

    # VAD test, lp_lifter_freq equals str
    with pytest.raises(
        TypeError,
        match="Argument lp_lifter_freq with value 10 is not of type"
        " \\[<class 'int'>, <class 'float'>\\], but got <class 'str'>.",
    ):
        audio.Vad(800, lp_lifter_freq="10")

    # VAD test, lp_lifter_freq equals bool
    with pytest.raises(
        TypeError,
        match="Argument lp_lifter_freq with value True is not of type \\("
        "<class 'int'>, <class 'float'>\\), but got <class 'bool'>.",
    ):
        audio.Vad(800, lp_lifter_freq=True)


if __name__ == "__main__":
    test_vad_pipeline()
    test_vad_pipeline_invalid_param()
    test_vad_eager()
    test_vad_transform()
    test_vad_invalid_input()
    test_vad_invalid_sample_rate()
    test_vad_invalid_trigger_level()
    test_vad_invalid_trigger_time()
    test_vad_invalid_search_time()
    test_vad_invalid_allowed_gap()
    test_vad_invalid_pre_trigger_time()
    test_vad_invalid_boot_time()
    test_vad_invalid_noise_up_time()
    test_vad_invalid_noise_down_time()
    test_vad_invalid_noise_reduction_amount()
    test_vad_invalid_measure_freq()
    test_vad_invalid_measure_duration()
    test_vad_invalid_measure_smooth_time()
    test_vad_invalid_hp_filter_freq()
    test_vad_invalid_lp_lifter_freq()
