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
"""Test GriffinLim."""

import numpy as np
import pytest

import mindspore.dataset as ds
from mindspore.dataset import audio
from . import count_unequal_element

DATA_DIR = "../data/dataset/audiorecord/"


def allclose_nparray(data_expected, data_me, rtol, atol, equal_nan=True):
    if np.any(np.isnan(data_expected)):
        assert np.allclose(data_me, data_expected, rtol, atol, equal_nan=equal_nan)
    elif not np.allclose(data_me, data_expected, rtol, atol, equal_nan=equal_nan):
        count_unequal_element(data_expected, data_me, rtol, atol)


def test_griffin_lim_pipeline():
    """
    Feature: GriffinLim
    Description: Test GriffinLim cpp in pipeline
    Expectation: Equal results from Mindspore and benchmark
    """
    # <101, 6>
    in_data = np.load(DATA_DIR + "griffinlim_101x6.npy")[np.newaxis, :]
    out_expect = np.load(DATA_DIR + "griffinlim_101x6_out.npy")
    dataset = ds.NumpySlicesDataset(in_data, column_names=["multi_dimensional_data"], shuffle=False)
    transforms = [audio.GriffinLim(n_fft=200, rand_init=False)]
    dataset = dataset.map(operations=transforms, input_columns=["multi_dimensional_data"])
    for item in dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
        output = item["multi_dimensional_data"]
        allclose_nparray(output, out_expect, 0.001, 0.001)

    # <151, 8>
    in_data = np.load(DATA_DIR + "griffinlim_151x8.npy")[np.newaxis, :]
    out_expect = np.load(DATA_DIR + "griffinlim_151x8_out.npy")
    dataset = ds.NumpySlicesDataset(in_data, column_names=["multi_dimensional_data"], shuffle=False)
    transforms = [audio.GriffinLim(n_fft=300, n_iter=20, win_length=240, hop_length=120, rand_init=False, power=1.2)]
    dataset = dataset.map(operations=transforms, input_columns=["multi_dimensional_data"])
    for item in dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
        output = item["multi_dimensional_data"]
        allclose_nparray(output, out_expect, 0.001, 0.001)

    # <2, 301, 4> hop_length greater than half of win_length
    in_data = np.load(DATA_DIR + "griffinlim_2x301x4.npy")[np.newaxis, :]
    out_expect = np.load(DATA_DIR + "griffinlim_2x301x4_out.npy")
    dataset = ds.NumpySlicesDataset(in_data, column_names=["multi_dimensional_data"], shuffle=False)
    transforms = [audio.GriffinLim(n_fft=600, n_iter=10, win_length=240, hop_length=130, rand_init=False)]
    dataset = dataset.map(operations=transforms, input_columns=["multi_dimensional_data"])
    for item in dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
        output = item["multi_dimensional_data"]
        allclose_nparray(output, out_expect, 0.001, 0.001)


def test_griffin_lim_pipeline_invalid_param_range():
    """
    Feature: GriffinLim
    Description: Test GriffinLim with invalid input parameters
    Expectation: Throw correct error and message
    """
    with pytest.raises(ValueError, match=r"Input n_fft is not within the required interval of \[1, 2147483647\]."):
        audio.GriffinLim(n_fft=-10)

    with pytest.raises(ValueError, match=r"Input n_iter is not within the required interval of \[1, 2147483647\]."):
        audio.GriffinLim(n_fft=300, n_iter=-10)

    with pytest.raises(ValueError, match=r"Input win_length is not within the required interval of \[0, 2147483647\]."):
        audio.GriffinLim(n_fft=300, n_iter=10, win_length=-10)

    with pytest.raises(ValueError,
                       match=r"Input win_length should be no more than n_fft, but got win_length: 400 " +
                             r"and n_fft: 300."):
        audio.GriffinLim(n_fft=300, n_iter=10, win_length=400)

    with pytest.raises(ValueError, match=r"Input hop_length is not within the required interval of \[0, 2147483647\]."):
        audio.GriffinLim(n_fft=300, n_iter=10, win_length=0, hop_length=-10)

    with pytest.raises(ValueError, match=r"Input power is not within the required interval of \(0, 16777216\]."):
        audio.GriffinLim(n_fft=300, n_iter=10, win_length=0, hop_length=0, power=-3)

    with pytest.raises(ValueError, match=r"Input momentum is not within the required interval of \[0, 16777216\]."):
        audio.GriffinLim(n_fft=300, n_iter=10, win_length=0, hop_length=0, power=2, momentum=-10)

    with pytest.raises(ValueError, match=r"Input length is not within the required interval of \[0, 2147483647\]."):
        audio.GriffinLim(n_fft=300, n_iter=10, win_length=0, hop_length=0, power=2, momentum=0.9, length=-2)


def test_griffin_lim_pipeline_invalid_param_constraint():
    """
    Feature: GriffinLim
    Description: Test GriffinLim with invalid input parameters
    Expectation: Throw RuntimeError
    """
    data = np.load(DATA_DIR + "griffinlim_151x8.npy")[np.newaxis, :]
    dataset = ds.NumpySlicesDataset(data, column_names=["multi_dimensional_data"], shuffle=False)

    with pytest.raises(RuntimeError,
                       match=r"map operation: \[GriffinLim\] failed. " +
                             r"GriffinLim: the frequency of the input should equal to n_fft / 2 \+ 1"):
        transforms = [audio.GriffinLim(n_fft=100)]
        dataset = dataset.map(operations=transforms, input_columns=["multi_dimensional_data"])
        for _ in dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
            pass

    with pytest.raises(RuntimeError,
                       match=r"map operation: \[GriffinLim\] failed. " +
                             r"GriffinLim: the frequency of the input should equal to n_fft / 2 \+ 1"):
        transforms = [audio.GriffinLim(n_fft=300, n_iter=10, win_length=0, hop_length=120)]
        dataset = dataset.map(operations=transforms, input_columns=["multi_dimensional_data"])
        for _ in dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
            pass

    with pytest.raises(RuntimeError,
                       match=r"GriffinLim: momentum equal to or greater than 1 can be unstable, " +
                             "but got: 1.000000"):
        transforms = [audio.GriffinLim(n_fft=300, n_iter=10, win_length=0, hop_length=0, power=2, momentum=1)]
        dataset = dataset.map(operations=transforms, input_columns=["multi_dimensional_data"])
        for _ in dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
            pass


def test_griffin_lim_pipeline_invalid_param_type():
    """
    Feature: GriffinLim
    Description: Test GriffinLim with invalid input parameters
    Expectation: Throw correct error and message
    """
    with pytest.raises(TypeError,
                       match=r"Argument window_type with value type is not of type " +
                             r"\[<enum \'WindowType\'>\], but got <class \'str\'>."):
        audio.GriffinLim(n_fft=300, n_iter=10, win_length=0, hop_length=0, window_type="type")

    with pytest.raises(TypeError,
                       match=r"Argument rand_init with value true is not of type \[<class \'bool\'>\], " +
                             r"but got <class \'str\'>."):
        audio.GriffinLim(n_fft=300, n_iter=10, win_length=0, hop_length=0, power=2, momentum=0.9, length=0,
                         rand_init='true')


def test_griffin_lim_eager():
    """
    Feature: GriffinLim
    Description: Test GriffinLim cpp with eager mode
    Expectation: Equal results from Mindspore and benchmark
    """
    # <freq, time>
    spectrogram = np.load(DATA_DIR + "griffinlim_101x6.npy").astype(np.float64)
    out_expect = np.load(DATA_DIR + "griffinlim_101x6_out.npy").astype(np.float64)
    out_ms = audio.GriffinLim(n_fft=200, rand_init=False)(spectrogram)
    allclose_nparray(out_ms, out_expect, 0.001, 0.001)

    # <1, freq, time>
    spectrogram = np.load(DATA_DIR + "griffinlim_1x201x6.npy").astype(np.float64)
    out_expect = np.load(DATA_DIR + "griffinlim_1x201x6_out.npy").astype(np.float64)
    out_ms = audio.GriffinLim(rand_init=False)(spectrogram)
    allclose_nparray(out_ms, out_expect, 0.001, 0.001)

    # <2, freq, time>
    spectrogram = np.load(DATA_DIR + "griffinlim_2x301x6.npy").astype(np.float64)
    out_expect = np.load(DATA_DIR + "griffinlim_2x301x6_out.npy").astype(np.float64)
    out_ms = audio.GriffinLim(n_fft=600, rand_init=False)(spectrogram)
    allclose_nparray(out_ms, out_expect, 0.001, 0.001)


def test_griffin_lim_transform():
    """
    Feature: GriffinLim
    Description: Test GriffinLim with various valid input parameters and data types
    Expectation: The operation completes successfully
    """
    # Test when input data type is float16, GriffinLim interface call is successful
    waveform = np.random.random([201, 6]).astype(np.float16)
    griffin_lim = audio.GriffinLim(n_fft=400, n_iter=1, rand_init=False)
    griffin_lim(waveform)

    # When parameters are set to default values, GriffinLim interface call is successful
    waveform = np.random.random([1, 201, 6]).astype(np.float32)
    griffin_lim = audio.GriffinLim(n_fft=400, win_length=400, hop_length=200, window_type=audio.WindowType.HANN,
                                     power=2.0, momentum=0.99, rand_init=True)
    griffin_lim(waveform)

    # When parameter window_type is KAISER, GriffinLim interface call is successful
    waveform = np.random.random([2, 101, 6]).astype(np.float64)
    griffin_lim = audio.GriffinLim(n_fft=200, win_length=100, hop_length=200, window_type=audio.WindowType.KAISER,
                                     power=0.1, momentum=0.5, rand_init=False)
    griffin_lim(waveform)

    # When input data type is double, GriffinLim interface call is successful
    waveform = np.random.random([2, 101, 6]).astype(np.double)
    griffin_lim = audio.GriffinLim(n_fft=200, win_length=160, hop_length=100, window_type=audio.WindowType.BLACKMAN,
                                     power=1.2, momentum=0.85, rand_init=False)
    griffin_lim(waveform)

    # When input data shape is 3D, GriffinLim interface call is successful
    waveform = np.random.randn(1, 301, 4).astype(np.float64)
    griffin_lim = audio.GriffinLim(n_fft=600, win_length=0, hop_length=120, rand_init=False)
    griffin_lim(waveform)

    # In pipeline mode, GriffinLim interface call is successful

    waveforms = np.random.random([100, 3, 201, 6]).astype(np.float32)
    dataset1 = ds.NumpySlicesDataset(waveforms, column_names=["multi_dimensional_data"])
    griffin_lim = audio.GriffinLim()
    dataset1 = dataset1.map(input_columns=["multi_dimensional_data"], operations=griffin_lim)
    for _ in dataset1.create_dict_iterator(output_numpy=True):
        pass

    # diff dtype
    waveforms = np.random.random([100, 1, 201, 4]).astype(np.float64)
    dataset2 = ds.NumpySlicesDataset(waveforms, column_names=["multi_dimensional_data"])
    griffin_lim = audio.GriffinLim(n_iter=10, window_type=audio.WindowType.BARTLETT, rand_init=False)
    dataset2 = dataset2.map(input_columns=["multi_dimensional_data"], operations=griffin_lim)
    for _ in dataset2.create_dict_iterator(output_numpy=True):
        pass

    # diff dtype
    waveforms = np.random.random([100, 1, 201, 6]).astype(np.float16)
    dataset3 = ds.NumpySlicesDataset(waveforms, column_names=["multi_dimensional_data"])
    griffin_lim = audio.GriffinLim(window_type=audio.WindowType.BLACKMAN, power=3, momentum=0.55)
    dataset3 = dataset3.map(input_columns=["multi_dimensional_data"], operations=griffin_lim)
    for _ in dataset3.create_dict_iterator(output_numpy=True):
        pass

    # diff dtype
    waveforms = np.random.random([100, 1, 201, 6]).astype(np.int_)
    dataset4 = ds.NumpySlicesDataset(waveforms, column_names=["multi_dimensional_data"])
    griffin_lim = audio.GriffinLim(n_iter=30, rand_init=False)
    dataset4 = dataset4.map(input_columns=["multi_dimensional_data"], operations=griffin_lim)
    for _ in dataset4.create_dict_iterator(output_numpy=True):
        pass

    # In eager mode, when parameter power is 167772, GriffinLim interface call is successful
    waveform = np.random.random([1, 201, 6]).astype(np.float32)
    griffin_lim = audio.GriffinLim(power=167772)
    griffin_lim(waveform)

    # When parameter length is 0, GriffinLim interface call is successful
    waveform = np.random.randn(2, 201, 6)
    griffin_lim = audio.GriffinLim(length=0, rand_init=False)
    griffin_lim(waveform)


def test_griffin_lim_param_check():
    """
    Feature: GriffinLim
    Description: Test GriffinLim with invalid input data types and parameters
    Expectation: Correct error types and messages are raised as expected
    """
    # When input data type is list, GriffinLim interface call fails
    waveform = np.random.randn(2, 101, 6).astype(np.float64).tolist()
    griffin_lim = audio.GriffinLim(n_fft=200, rand_init=False)
    with pytest.raises(TypeError, match="Input should be NumPy audio, got <class 'list'>."):
        griffin_lim(waveform)

    # When input data type is string, GriffinLim interface call fails
    waveform = np.random.randn(2, 101, 6).astype(np.float32).tobytes()
    griffin_lim = audio.GriffinLim(n_fft=200, rand_init=False)
    with pytest.raises(TypeError, match="Input should be NumPy audio, got <class 'bytes'>."):
        griffin_lim(waveform)

    # When parameter n_fft is string, GriffinLim interface call fails
    with pytest.raises(TypeError, match="Argument n_fft with value null is not of type \\[<class 'int'>\\],"
                                        " but got <class 'str'>."):
        audio.GriffinLim(n_fft='null', win_length=400, hop_length=200, window_type=audio.WindowType.HANN,
                         power=2.0, momentum=0.99, rand_init=True)

    # When parameter n_fft is bool, GriffinLim interface call fails
    with pytest.raises(TypeError, match="Argument n_fft with value True is not of type \\(<class 'int'>,\\),"
                                        " but got <class 'bool'>."):
        audio.GriffinLim(n_fft=True, window_type=audio.WindowType.HAMMING,
                         power=1.0, momentum=0.99, rand_init=False)

    # When parameter n_fft is 0, GriffinLim interface call fails
    with pytest.raises(ValueError, match="Input n_fft is not within the required interval of \\[1, 2147483647\\]."):
        audio.GriffinLim(n_fft=0)

    # When parameter n_fft is -1, GriffinLim interface call fails
    with pytest.raises(ValueError, match="Input n_fft is not within the required interval of \\[1, 2147483647\\]."):
        audio.GriffinLim(n_fft=-1, rand_init=False)

    # When parameter n_iter is string, GriffinLim interface call fails
    with pytest.raises(TypeError, match="Argument n_iter with value 1 is not of type \\[<class 'int'>\\],"
                                        " but got <class 'str'>."):
        audio.GriffinLim(n_iter='1', win_length=400, hop_length=200, window_type=audio.WindowType.HANN,
                         power=2.0, momentum=0.99, rand_init=True)

    # When parameter n_iter is 0, GriffinLim interface call fails
    with pytest.raises(ValueError, match="Input n_iter is not within the required interval of \\[1, 2147483647\\]."):
        audio.GriffinLim(n_iter=0)

    # When parameter n_iter is -1, GriffinLim interface call fails
    with pytest.raises(ValueError, match="Input n_iter is not within the required interval of \\[1, 2147483647\\]."):
        audio.GriffinLim(n_iter=-1, rand_init=False)

    # When parameter win_length is string, GriffinLim interface call fails
    with pytest.raises(TypeError, match="Argument win_length with value 1 is not of type \\[<class 'int'>\\],"
                                        " but got <class 'str'>."):
        audio.GriffinLim(win_length='1', window_type=audio.WindowType.HANN,
                         power=2.0, momentum=0.99, rand_init=True)

    # When parameter win_length is list, GriffinLim interface call fails
    with pytest.raises(TypeError, match="Argument win_length with value \\[0\\] is not of type \\[<class 'int'>\\],"
                                        " but got <class 'list'>."):
        audio.GriffinLim(win_length=[0])

    # When parameter win_length is -1, GriffinLim interface call fails
    with pytest.raises(ValueError, match="Input win_length is not within the required interval of"
                                         " \\[0, 2147483647\\]."):
        audio.GriffinLim(win_length=-1, rand_init=False)

    # When parameter hop_length is string, GriffinLim interface call fails
    with pytest.raises(TypeError, match="Argument hop_length with value 1 is not of type \\[<class 'int'>\\],"
                                        " but got <class 'str'>."):
        audio.GriffinLim(hop_length='1', window_type=audio.WindowType.BLACKMAN,
                         power=1.0, momentum=0.99, rand_init=True)

    # When parameter hop_length is list, GriffinLim interface call fails
    with pytest.raises(TypeError, match="Argument hop_length with value \\[0\\] is not of type \\[<class 'int'>\\],"
                                        " but got <class 'list'>."):
        audio.GriffinLim(hop_length=[0])

    # When parameter hop_length is negative, GriffinLim interface call fails
    with pytest.raises(ValueError, match="Input hop_length is not within the required interval of "
                                         "\\[0, 2147483647\\]."):
        audio.GriffinLim(hop_length=-2, rand_init=False)

    # When parameter window_type is string, GriffinLim interface call fails
    with pytest.raises(TypeError, match="Argument window_type with value WindowType.BLACKMAN is not of type"
                                        " \\[<enum 'WindowType'>\\], but got <class 'str'>."):
        audio.GriffinLim(window_type="WindowType.BLACKMAN",
                         power=1.0, momentum=0.99, rand_init=True)

    # When parameter window_type is list, GriffinLim interface call fails
    with pytest.raises(TypeError, match="Argument window_type with value True is not of type \\[<enum 'WindowType'>\\],"
                                        " but got <class 'bool'>"):
        audio.GriffinLim(window_type=True)

    # When parameter window_type value is set incorrectly, GriffinLim interface call fails
    with pytest.raises(AttributeError, match="BLACKMEN"):
        audio.GriffinLim(window_type=audio.WindowType.BLACKMEN)

    # When parameter power is string, GriffinLim interface call fails
    with pytest.raises(TypeError, match="Argument power with value 1 is not of type \\[<class 'int'>, <class 'float'>"
                                        "\\], but got <class 'str'>."):
        audio.GriffinLim(power='1', window_type=audio.WindowType.KAISER, momentum=0.99, rand_init=True)

    # When parameter power is list, GriffinLim interface call fails
    with pytest.raises(TypeError, match="Argument power with value \\[0\\] is not of type \\[<class 'int'>,"
                                        " <class 'float'>\\], but got <class 'list'>."):
        audio.GriffinLim(power=[0])

    # When parameter power is negative, GriffinLim interface call fails
    with pytest.raises(ValueError, match="Input power is not within the required interval of \\(0, 16777216\\]."):
        audio.GriffinLim(power=-2, rand_init=False)

    # When parameter momentum is string, GriffinLim interface call fails
    with pytest.raises(TypeError, match="Argument momentum with value 0.99 is not of type \\[<class 'int'>, "
                                        "<class 'float'>\\], but got <class 'str'>."):
        audio.GriffinLim(momentum='0.99', window_type=audio.WindowType.KAISER, rand_init=True)

    # When parameter momentum is negative, GriffinLim interface call fails
    with pytest.raises(ValueError, match="Input momentum is not within the required interval of \\[0, 16777216\\]."):
        audio.GriffinLim(momentum=-1, rand_init=False)

    # When parameter length is string, GriffinLim interface call fails
    with pytest.raises(TypeError, match="Argument length with value 0 is not of type \\[<class 'int'>\\],"
                                        " but got <class 'str'>."):
        audio.GriffinLim(length='0')

    # When parameter length is negative, GriffinLim interface call fails
    with pytest.raises(ValueError, match="Input length is not within the required interval of \\[0, 2147483647\\]."):
        audio.GriffinLim(length=-2, rand_init=False)

    # When parameter rand_init is string, GriffinLim interface call fails
    with pytest.raises(TypeError, match="Argument rand_init with value False is not of type \\[<class 'bool'>\\],"
                                        " but got <class 'str'>."):
        audio.GriffinLim(rand_init='False')

    # When parameter rand_init is int, GriffinLim interface call fails
    with pytest.raises(TypeError, match="Argument rand_init with value 1 is not of type \\[<class 'bool'>\\],"
                                        " but got <class 'int'>."):
        audio.GriffinLim(rand_init=1)

    # When input data freq does not equal n_fft/2 + 1, GriffinLim interface call fails
    with pytest.raises(RuntimeError, match=r"GriffinLim: the frequency of the input should"
                                           " equal to "):
        waveform = np.random.randn(2, 201, 6)
        # 201 does not equal 600/2 + 1
        griffin_lim = audio.GriffinLim(n_fft=600, win_length=400, hop_length=100, window_type=audio.WindowType.HANN,
                                         power=2.0, momentum=0.9, rand_init=False)
        griffin_lim(waveform)

    # When parameter win_length is greater than n_fft, GriffinLim interface call fails
    with pytest.raises(ValueError, match="Input win_length should be no more than n_fft, "
                                         "but got win_length: 480 and n_fft: 400."):
        waveform = np.random.randn(2, 201, 6)
        griffin_lim = audio.GriffinLim(n_fft=400, win_length=480, hop_length=100, window_type=audio.WindowType.HANN,
                                         power=1.0, momentum=0.1, rand_init=False)
        griffin_lim(waveform)

    # When parameter power is 0, GriffinLim interface call fails
    with pytest.raises(ValueError, match=r"Input power is not within the required interval of \(0, 16777216\]."):
        waveform = np.random.randn(2, 201, 6)
        audio.GriffinLim(n_fft=400, power=0)(waveform)

    # When parameter length is 1, GriffinLim interface call fails
    with pytest.raises(RuntimeError) as e:
        waveform = np.random.randn(2, 201, 6)
        # test length is 1, less than n_fft(400)
        griffin_lim = audio.GriffinLim(length=1, rand_init=False)
        griffin_lim(waveform)
    assert "GriffinLim: n_fft must be less than length" in str(e.value)


if __name__ == "__main__":
    test_griffin_lim_pipeline()
    test_griffin_lim_pipeline_invalid_param_range()
    test_griffin_lim_pipeline_invalid_param_constraint()
    test_griffin_lim_pipeline_invalid_param_type()
    test_griffin_lim_eager()
    test_griffin_lim_transform()
    test_griffin_lim_param_check()
