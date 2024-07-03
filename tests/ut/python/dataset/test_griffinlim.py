# Copyright 2022 Huawei Technologies Co., Ltd
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
Testing GriffinLim op in DE
"""
import numpy as np
import pytest

import mindspore.dataset as ds
import mindspore.dataset.audio as c_audio
from mindspore import log as logger

DATA_DIR = "../data/dataset/audiorecord/"


def count_unequal_element(data_expected, data_me, rtol, atol):
    assert data_expected.shape == data_me.shape
    total_count = len(data_expected.flatten())
    error = np.abs(data_expected - data_me)
    greater = np.greater(error, atol + np.abs(data_expected) * rtol)
    loss_count = np.count_nonzero(greater)
    assert (loss_count / total_count) < rtol, \
        "\ndata_expected_std:{0}\ndata_me_error:{1}\nloss:{2}". \
            format(data_expected[greater], data_me[greater], error[greater])


def allclose_nparray(data_expected, data_me, rtol, atol, equal_nan=True):
    if np.any(np.isnan(data_expected)):
        assert np.allclose(data_me, data_expected, rtol, atol, equal_nan=equal_nan)
    elif not np.allclose(data_me, data_expected, rtol, atol, equal_nan=equal_nan):
        count_unequal_element(data_expected, data_me, rtol, atol)


def test_griffin_lim_pipeline():
    """
    Feature: GriffinLim
    Description: Test GriffinLim cpp op in pipeline
    Expectation: Equal results from Mindspore and benchmark
    """
    # <101, 6>
    in_data = np.load(DATA_DIR + "griffinlim_101x6.npy")[np.newaxis, :]
    out_expect = np.load(DATA_DIR + "griffinlim_101x6_out.npy")
    dataset = ds.NumpySlicesDataset(in_data, column_names=["multi_dimensional_data"], shuffle=False)
    transforms = [c_audio.GriffinLim(n_fft=200, rand_init=False)]
    dataset = dataset.map(operations=transforms, input_columns=["multi_dimensional_data"])
    for item in dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
        out_put = item["multi_dimensional_data"]
        allclose_nparray(out_put, out_expect, 0.001, 0.001)

    # <151, 8>
    in_data = np.load(DATA_DIR + "griffinlim_151x8.npy")[np.newaxis, :]
    out_expect = np.load(DATA_DIR + "griffinlim_151x8_out.npy")
    dataset = ds.NumpySlicesDataset(in_data, column_names=["multi_dimensional_data"], shuffle=False)
    transforms = [c_audio.GriffinLim(n_fft=300, n_iter=20, win_length=240, hop_length=120, rand_init=False, power=1.2)]
    dataset = dataset.map(operations=transforms, input_columns=["multi_dimensional_data"])
    for item in dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
        out_put = item["multi_dimensional_data"]
        allclose_nparray(out_put, out_expect, 0.001, 0.001)

    # <2, 301, 4> hop_length greater than half of win_length
    in_data = np.load(DATA_DIR + "griffinlim_2x301x4.npy")[np.newaxis, :]
    out_expect = np.load(DATA_DIR + "griffinlim_2x301x4_out.npy")
    dataset = ds.NumpySlicesDataset(in_data, column_names=["multi_dimensional_data"], shuffle=False)
    transforms = [c_audio.GriffinLim(n_fft=600, n_iter=10, win_length=240, hop_length=130, rand_init=False)]
    dataset = dataset.map(operations=transforms, input_columns=["multi_dimensional_data"])
    for item in dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
        out_put = item["multi_dimensional_data"]
        allclose_nparray(out_put, out_expect, 0.001, 0.001)


def test_griffin_lim_pipeline_invalid_param_range():
    """
    Feature: GriffinLim
    Description: Test GriffinLim with invalid input parameters
    Expectation: Throw correct error and message
    """
    logger.info("test GriffinLim op with default values")
    in_data = np.load(DATA_DIR + "griffinlim_151x8.npy")[np.newaxis, :]
    data1 = ds.NumpySlicesDataset(in_data, column_names=["multi_dimensional_data"], shuffle=False)

    with pytest.raises(ValueError, match=r"Input n_fft is not within the required interval of \[1, 2147483647\]."):
        transforms = [c_audio.GriffinLim(n_fft=-10)]
        data1 = data1.map(operations=transforms, input_columns=["multi_dimensional_data"])
        for item in data1.create_dict_iterator(num_epochs=1, output_numpy=True):
            _ = item["multi_dimensional_data"]

    with pytest.raises(ValueError, match=r"Input n_iter is not within the required interval of \[1, 2147483647\]."):
        transforms = [c_audio.GriffinLim(n_fft=300, n_iter=-10)]
        data1 = data1.map(operations=transforms, input_columns=["multi_dimensional_data"])
        for item in data1.create_dict_iterator(num_epochs=1, output_numpy=True):
            _ = item["multi_dimensional_data"]

    with pytest.raises(ValueError, match=r"Input win_length is not within the required interval of \[0, 2147483647\]."):
        transforms = [c_audio.GriffinLim(n_fft=300, n_iter=10, win_length=-10)]
        data1 = data1.map(operations=transforms, input_columns=["multi_dimensional_data"])
        for item in data1.create_dict_iterator(num_epochs=1, output_numpy=True):
            _ = item["multi_dimensional_data"]

    with pytest.raises(ValueError,
                       match=r"Input win_length should be no more than n_fft, but got win_length: 400 " +
                       r"and n_fft: 300."):
        transforms = [c_audio.GriffinLim(n_fft=300, n_iter=10, win_length=400)]
        data1 = data1.map(operations=transforms, input_columns=["multi_dimensional_data"])
        for item in data1.create_dict_iterator(num_epochs=1, output_numpy=True):
            _ = item["multi_dimensional_data"]

    with pytest.raises(ValueError, match=r"Input hop_length is not within the required interval of \[0, 2147483647\]."):
        transforms = [c_audio.GriffinLim(n_fft=300, n_iter=10, win_length=0, hop_length=-10)]
        data1 = data1.map(operations=transforms, input_columns=["multi_dimensional_data"])
        for item in data1.create_dict_iterator(num_epochs=1, output_numpy=True):
            _ = item["multi_dimensional_data"]

    with pytest.raises(ValueError, match=r"Input power is not within the required interval of \(0, 16777216\]."):
        transforms = [c_audio.GriffinLim(n_fft=300, n_iter=10, win_length=0, hop_length=0, power=-3)]
        data1 = data1.map(operations=transforms, input_columns=["multi_dimensional_data"])
        for item in data1.create_dict_iterator(num_epochs=1, output_numpy=True):
            _ = item["multi_dimensional_data"]

    with pytest.raises(ValueError, match=r"Input momentum is not within the required interval of \[0, 16777216\]."):
        transforms = [c_audio.GriffinLim(n_fft=300, n_iter=10, win_length=0, hop_length=0, power=2, momentum=-10)]
        data1 = data1.map(operations=transforms, input_columns=["multi_dimensional_data"])
        for item in data1.create_dict_iterator(num_epochs=1, output_numpy=True):
            _ = item["multi_dimensional_data"]

    with pytest.raises(ValueError, match=r"Input length is not within the required interval of \[0, 2147483647\]."):
        transforms = [
            c_audio.GriffinLim(n_fft=300, n_iter=10, win_length=0, hop_length=0, power=2, momentum=0.9, length=-2)
        ]
        data1 = data1.map(operations=transforms, input_columns=["multi_dimensional_data"])
        for item in data1.create_dict_iterator(num_epochs=1, output_numpy=True):
            _ = item["multi_dimensional_data"]


def test_griffin_lim_pipeline_invalid_param_constraint():
    """
    Feature: GriffinLim
    Description: Test GriffinLim with invalid input parameters
    Expectation: Throw RuntimeError
    """
    logger.info("test GriffinLim op with default values")
    in_data = np.load(DATA_DIR + "griffinlim_151x8.npy")[np.newaxis, :]
    data1 = ds.NumpySlicesDataset(in_data, column_names=["multi_dimensional_data"], shuffle=False)

    with pytest.raises(RuntimeError,
                       match=r"map operation: \[GriffinLim\] failed. " +
                       r"GriffinLim: the frequency of the input should equal to n_fft / 2 \+ 1"):
        transforms = [c_audio.GriffinLim(n_fft=100)]
        data1 = data1.map(operations=transforms, input_columns=["multi_dimensional_data"])
        for item in data1.create_dict_iterator(num_epochs=1, output_numpy=True):
            _ = item["multi_dimensional_data"]

    with pytest.raises(RuntimeError,
                       match=r"map operation: \[GriffinLim\] failed. " +
                       r"GriffinLim: the frequency of the input should equal to n_fft / 2 \+ 1"):
        transforms = [c_audio.GriffinLim(n_fft=300, n_iter=10, win_length=0, hop_length=120)]
        data1 = data1.map(operations=transforms, input_columns=["multi_dimensional_data"])
        for item in data1.create_dict_iterator(num_epochs=1, output_numpy=True):
            _ = item["multi_dimensional_data"]

    with pytest.raises(RuntimeError,
                       match=r"GriffinLim: momentum equal to or greater than 1 can be unstable, " +
                       "but got: 1.000000"):
        transforms = [c_audio.GriffinLim(n_fft=300, n_iter=10, win_length=0, hop_length=0, power=2, momentum=1)]
        data1 = data1.map(operations=transforms, input_columns=["multi_dimensional_data"])
        for item in data1.create_dict_iterator(num_epochs=1, output_numpy=True):
            _ = item["multi_dimensional_data"]


def test_griffin_lim_pipeline_invalid_param_type():
    """
    Feature: GriffinLim
    Description: Test GriffinLim with invalid input parameters
    Expectation: Throw correct error and message
    """
    logger.info("test GriffinLim op with default values")
    in_data = np.load(DATA_DIR + "griffinlim_151x8.npy")[np.newaxis, :]
    data1 = ds.NumpySlicesDataset(in_data, column_names=["multi_dimensional_data"], shuffle=False)

    with pytest.raises(TypeError,
                       match=r"Argument window_type with value type is not of type " +
                       r"\[<enum \'WindowType\'>\], but got <class \'str\'>."):
        transforms = [c_audio.GriffinLim(n_fft=300, n_iter=10, win_length=0, hop_length=0, window_type="type")]
        data1 = data1.map(operations=transforms, input_columns=["multi_dimensional_data"])
        for item in data1.create_dict_iterator(num_epochs=1, output_numpy=True):
            _ = item["multi_dimensional_data"]

    with pytest.raises(TypeError,
                       match=r"Argument rand_init with value true is not of type \[<class \'bool\'>\], " +
                       r"but got <class \'str\'>."):
        transforms = [
            c_audio.GriffinLim(n_fft=300,
                               n_iter=10,
                               win_length=0,
                               hop_length=0,
                               power=2,
                               momentum=0.9,
                               length=0,
                               rand_init='true')
        ]
        data1 = data1.map(operations=transforms, input_columns=["multi_dimensional_data"])
        for item in data1.create_dict_iterator(num_epochs=1, output_numpy=True):
            _ = item["multi_dimensional_data"]


def test_griffin_lim_eager():
    """
    Feature: GriffinLim
    Description: Test GriffinLim cpp op with eager mode
    Expectation: Equal results from Mindspore and benchmark
    """
    # <freq, time>
    spectrogram = np.load(DATA_DIR + "griffinlim_101x6.npy").astype(np.float64)
    out_expect = np.load(DATA_DIR + "griffinlim_101x6_out.npy").astype(np.float64)
    out_ms = c_audio.GriffinLim(n_fft=200, rand_init=False)(spectrogram)
    allclose_nparray(out_ms, out_expect, 0.001, 0.001)
    # <1, freq, time>
    spectrogram = np.load(DATA_DIR + "griffinlim_1x201x6.npy").astype(np.float64)
    out_expect = np.load(DATA_DIR + "griffinlim_1x201x6_out.npy").astype(np.float64)
    out_ms = c_audio.GriffinLim(rand_init=False)(spectrogram)
    allclose_nparray(out_ms, out_expect, 0.001, 0.001)
    # <2, freq, time>
    spectrogram = np.load(DATA_DIR + "griffinlim_2x301x6.npy").astype(np.float64)
    out_expect = np.load(DATA_DIR + "griffinlim_2x301x6_out.npy").astype(np.float64)
    out_ms = c_audio.GriffinLim(n_fft=600, rand_init=False)(spectrogram)
    allclose_nparray(out_ms, out_expect, 0.001, 0.001)


if __name__ == "__main__":
    test_griffin_lim_pipeline()
    test_griffin_lim_pipeline_invalid_param_range()
    test_griffin_lim_pipeline_invalid_param_constraint()
    test_griffin_lim_pipeline_invalid_param_type()
    test_griffin_lim_eager()
